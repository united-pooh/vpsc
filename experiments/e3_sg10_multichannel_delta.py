"""SG10 multichannel action-conditioned TextWorld event-delta experiment."""

from __future__ import annotations

import argparse
import copy
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import random
import sys
import time
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import torch
from torch import nn
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.e2_f0_fusion_benchmark import (  # noqa: E402
    _environment,
    _percentile,
    _sample_summary,
    _sync,
)
from experiments import e3_sg0_counterfactual_generation as sg0  # noqa: E402
from experiments import e3_tw0_sparse_event_lm as tw0  # noqa: E402
from experiments.e3_sg1_history_generation import build_history_models  # noqa: E402
from experiments.e3_sg4_move_pair_ranking import (  # noqa: E402
    DEFAULT_CORPUS_DIR,
    EXPECTED_DATA_SEEDS,
)
from experiments.e3_sg8_bilinear_closed_form import (  # noqa: E402
    RIDGE_LAMBDAS,
    _outer_features,
    _query_hidden,
)
from experiments.e3_sg9_atomic_event_stream import (  # noqa: E402
    _generic_candidate_hidden,
    _prefill_previous_event,
    _snn_cached_decay_candidate_hidden,
    action_event_token,
)
from vpsc.world_model.cores import (  # noqa: E402
    E3GatedTraceScanCore,
    E3ScanState,
    count_parameters,
    state_nbytes,
)
from vpsc.world_model.event_corpus import load_event_corpus  # noqa: E402
from vpsc.world_model.wikitext import SPLITS, Vocabulary  # noqa: E402


START_EVENT = "<event_start>"
ROOM_LABELS = (
    "<room_no_observation>",
    "<room_novel>",
    "<room_previous>",
    "<room_same>",
)
REWARD_LABELS = ("<reward_zero>", "<reward_positive>")
DONE_LABELS = ("<continue>", "<done>")
EXIT_LABELS = ("<exit_count_0>", "<exit_count_1>", "<exit_count_2>")
CHANNEL_SPECS = (
    ("room_relation", ROOM_LABELS),
    ("reward", REWARD_LABELS),
    ("done", DONE_LABELS),
    ("move_exit_count_after", EXIT_LABELS),
)
CHANNEL_OFFSETS: Dict[str, Tuple[int, int]] = {}
_offset = 0
for _name, _labels in CHANNEL_SPECS:
    CHANNEL_OFFSETS[_name] = (_offset, _offset + len(_labels))
    _offset += len(_labels)
TOTAL_LOGITS = _offset
MODEL_NAMES = sg0.MODEL_NAMES
EXPECTED_COUNTS = {"train": 480, "valid": 60, "test": 60}
EXPECTED_GROUPS = {"train": 160, "valid": 20, "test": 20}
EXPECTED_LENGTH_GROUPS = {
    "train": {length: 32 for length in range(2, 7)},
    "valid": {length: 4 for length in range(2, 7)},
    "test": {length: 4 for length in range(2, 7)},
}


@dataclass(frozen=True)
class MultiChannelExample:
    split: str
    episode_index: int
    game_seed: int
    step_index: int
    candidate_index: int
    source: str
    step_group_id: str
    context_actions: Tuple[str, ...]
    candidate_action: str
    prompt_tokens: Tuple[str, ...]
    prompt_ids: Tuple[int, ...]
    target_indices: Tuple[int, int, int, int]
    target_labels: Tuple[str, str, str, str]
    prompt_unknowns: int

    @property
    def example_id(self) -> str:
        return (
            f"{self.split}:{self.game_seed}:{self.step_index}:"
            f"{self.candidate_index}"
        )

    @property
    def input_length(self) -> int:
        return len(self.prompt_ids)


class MultiChannelBilinearModel(nn.Module):
    def __init__(self, language_model: Any, d_model: int) -> None:
        super().__init__()
        self.language_model = language_model
        self.relation_head = nn.Bilinear(
            d_model, d_model, TOTAL_LOGITS, bias=True
        )


def build_multichannel_models(
    seed: int,
    vocabulary: Vocabulary,
    *,
    d_model: int,
    state_dim: int,
    num_heads: int,
    device: torch.device,
) -> Dict[str, MultiChannelBilinearModel]:
    base_models = build_history_models(
        seed,
        vocabulary,
        d_model=d_model,
        state_dim=state_dim,
        num_heads=num_heads,
        device=device,
    )
    torch.manual_seed(seed + 40_000)
    reference = nn.Bilinear(d_model, d_model, TOTAL_LOGITS, bias=True)
    shared = copy.deepcopy(reference.state_dict())
    models = {}
    for name, base in base_models.items():
        model = MultiChannelBilinearModel(base, d_model)
        model.relation_head.load_state_dict(shared)
        models[name] = model.to(device)
    return models


def _room_feature(corpus: Any, text: str) -> Optional[str]:
    normalized = sg0.normalize_textworld_observation(text)
    rooms = tuple(
        value
        for value in sg0._world_features(corpus.tokenizer.tokenize(normalized))
        if value.startswith("room:")
    )
    if len(rooms) > 1:
        raise ValueError(f"multiple room features in observation: {rooms}")
    return rooms[0] if rooms else None


def _room_relation_label(
    corpus: Any,
    next_observation: str,
    current_observation: str,
    prior_observations: Sequence[str],
) -> str:
    room = _room_feature(corpus, next_observation)
    if room is None:
        return ROOM_LABELS[0]
    current = _room_feature(corpus, current_observation)
    if room == current:
        return ROOM_LABELS[3]
    prior_rooms = {
        value
        for value in (
            _room_feature(corpus, observation)
            for observation in prior_observations
        )
        if value is not None
    }
    if room in prior_rooms:
        return ROOM_LABELS[2]
    return ROOM_LABELS[1]


def _move_exit_label(actions: Sequence[str]) -> str:
    count = sum(sg0._action_type(str(action)) == "move" for action in actions)
    if count >= len(EXIT_LABELS):
        raise ValueError(f"unsupported move exit count {count}")
    return EXIT_LABELS[count]


def build_multichannel_examples(
    corpus_root: Path,
    corpus: Any,
) -> Tuple[Dict[str, Tuple[MultiChannelExample, ...]], Vocabulary]:
    raw = {}
    for split in SPLITS:
        values = []
        path = corpus_root / split / "episodes.jsonl"
        for episode_index, line in enumerate(
            path.read_text(encoding="utf-8").splitlines()
        ):
            episode = json.loads(line)
            if episode.get("split") != split:
                raise ValueError(f"multichannel episode split mismatch: {path}")
            steps = episode["steps"]
            factual_actions = tuple(str(step["action"]) for step in steps)
            observations = tuple(str(step["observation"]) for step in steps)
            for step_index, step in enumerate(steps):
                if bool(step["done"]):
                    factual_after: Sequence[str] = ()
                elif step_index + 1 < len(steps):
                    factual_after = steps[step_index + 1]["admissible_actions"]
                else:
                    raise ValueError("nonterminal factual final step lacks after state")
                candidates = [
                    {
                        "source": "factual",
                        "action": str(step["action"]),
                        "next_obs": str(step["next_obs"]),
                        "reward": float(step["reward"]),
                        "done": bool(step["done"]),
                        "admissible_actions_after": tuple(factual_after),
                    }
                ]
                candidates.extend(
                    {
                        "source": "counterfactual",
                        "action": str(counterfactual["action"]),
                        "next_obs": str(counterfactual["next_obs"]),
                        "reward": float(counterfactual["reward"]),
                        "done": bool(counterfactual["done"]),
                        "admissible_actions_after": tuple(
                            counterfactual["admissible_actions_after"]
                        ),
                    }
                    for counterfactual in step["counterfactuals"]
                )
                if len(candidates) != 3:
                    raise ValueError("SG10 requires one factual plus two counterfactuals")
                step_group_id = f"{split}:{episode['seed']}:{step_index}"
                context_actions = factual_actions[:step_index]
                context_tokens = (
                    START_EVENT,
                    *(action_event_token(action) for action in context_actions),
                )
                for candidate_index, candidate in enumerate(candidates):
                    target_labels = (
                        _room_relation_label(
                            corpus,
                            candidate["next_obs"],
                            observations[step_index],
                            observations[:step_index],
                        ),
                        (
                            REWARD_LABELS[1]
                            if candidate["reward"] > 0.0
                            else REWARD_LABELS[0]
                        ),
                        DONE_LABELS[1] if candidate["done"] else DONE_LABELS[0],
                        _move_exit_label(candidate["admissible_actions_after"]),
                    )
                    target_indices = tuple(
                        labels.index(label)
                        for (_channel, labels), label in zip(
                            CHANNEL_SPECS, target_labels
                        )
                    )
                    candidate_action = str(candidate["action"])
                    values.append(
                        {
                            "split": split,
                            "episode_index": episode_index,
                            "game_seed": int(episode["seed"]),
                            "step_index": step_index,
                            "candidate_index": candidate_index,
                            "source": str(candidate["source"]),
                            "step_group_id": step_group_id,
                            "context_actions": context_actions,
                            "candidate_action": candidate_action,
                            "prompt_tokens": context_tokens
                            + (action_event_token(candidate_action),),
                            "target_indices": target_indices,
                            "target_labels": target_labels,
                        }
                    )
        raw[split] = tuple(values)

    vocabulary = Vocabulary.build(
        token
        for record in raw["train"]
        for token in record["prompt_tokens"]
    )
    examples = {}
    for split in SPLITS:
        encoded = []
        for record in raw[split]:
            prompt_ids = vocabulary.encode(record["prompt_tokens"])
            encoded.append(
                MultiChannelExample(
                    split=record["split"],
                    episode_index=record["episode_index"],
                    game_seed=record["game_seed"],
                    step_index=record["step_index"],
                    candidate_index=record["candidate_index"],
                    source=record["source"],
                    step_group_id=record["step_group_id"],
                    context_actions=record["context_actions"],
                    candidate_action=record["candidate_action"],
                    prompt_tokens=record["prompt_tokens"],
                    prompt_ids=prompt_ids,
                    target_indices=record["target_indices"],
                    target_labels=record["target_labels"],
                    prompt_unknowns=sum(
                        token_id == vocabulary.unk_id for token_id in prompt_ids
                    ),
                )
            )
        examples[split] = tuple(encoded)
    return examples, vocabulary


def audit_multichannel_examples(
    examples: Mapping[str, Sequence[MultiChannelExample]],
    vocabulary: Vocabulary,
    *,
    expected_counts: Mapping[str, int],
    expected_groups: Mapping[str, int],
) -> Dict[str, Any]:
    length_count = len(EXPECTED_LENGTH_GROUPS["train"])
    if any(expected_groups[split] % length_count for split in SPLITS):
        raise ValueError(
            "SG10 expected step groups must divide the five event lengths"
        )
    expected_length_groups = {
        split: {
            length: expected_groups[split] // length_count
            for length in EXPECTED_LENGTH_GROUPS["train"]
        }
        for split in SPLITS
    }
    splits = {}
    all_valid = True
    for split, values in examples.items():
        groups: Dict[str, list[MultiChannelExample]] = defaultdict(list)
        input_outputs: Dict[Tuple[int, ...], set[Tuple[int, ...]]] = defaultdict(set)
        for example in values:
            groups[example.step_group_id].append(example)
            input_outputs[example.prompt_ids].add(example.target_indices)
        channel_counts = {
            name: Counter(example.target_labels[index] for example in values)
            for index, (name, _labels) in enumerate(CHANNEL_SPECS)
        }
        length_groups = Counter(
            len(group[0].prompt_ids) for group in groups.values()
        )
        exact_vectors = Counter(example.target_labels for example in values)
        group_valid = all(
            len(group) == 3
            and {example.candidate_index for example in group} == {0, 1, 2}
            and Counter(example.source for example in group)
            == Counter({"factual": 1, "counterfactual": 2})
            and len({example.prompt_ids[:-1] for example in group}) == 1
            for group in groups.values()
        )
        ambiguity_count = sum(len(outputs) > 1 for outputs in input_outputs.values())
        record = {
            "example_count": len(values),
            "step_group_count": len(groups),
            "game_count": len({example.game_seed for example in values}),
            "prompt_length": sg0._length_summary(
                [len(example.prompt_ids) for example in values]
            ),
            "prompt_unknown_count": sum(
                example.prompt_unknowns for example in values
            ),
            "channel_label_counts": {
                name: dict(sorted(counts.items()))
                for name, counts in channel_counts.items()
            },
            "exact_vector_count": len(exact_vectors),
            "exact_vector_majority_accuracy": max(exact_vectors.values())
            / len(values),
            "length_group_counts": dict(sorted(length_groups.items())),
            "ambiguous_input_count": ambiguity_count,
            "groups_valid": group_valid,
        }
        valid = (
            len(values) == expected_counts[split]
            and len(groups) == expected_groups[split]
            and dict(length_groups) == expected_length_groups[split]
            and record["prompt_unknown_count"] == 0
            and ambiguity_count == 0
            and group_valid
            and all(
                set(channel_counts[name]) == set(labels)
                for name, labels in CHANNEL_SPECS
            )
        )
        record["passed"] = valid
        splits[split] = record
        all_valid = all_valid and valid

    train_inputs = {example.prompt_ids for example in examples["train"]}
    heldout_coverage = {
        split: {
            "covered": sum(
                example.prompt_ids in train_inputs for example in examples[split]
            ),
            "total": len(examples[split]),
        }
        for split in ("valid", "test")
    }
    return {
        "task": "multichannel action-conditioned real TextWorld event delta",
        "channels": {
            name: labels for name, labels in CHANNEL_SPECS
        },
        "label_sources": {
            "room_relation": "normalized next_obs room vs current/prior rooms",
            "reward": "recorded transition reward",
            "done": "recorded transition done",
            "move_exit_count_after": (
                "counterfactual admissible_actions_after or next factual step"
            ),
        },
        "expected_counts": dict(expected_counts),
        "expected_groups": dict(expected_groups),
        "expected_length_group_counts": expected_length_groups,
        "splits": splits,
        "heldout_full_input_coverage": heldout_coverage,
        "passed": all_valid,
    }


def build_length_stratified_schedule(
    examples: Sequence[MultiChannelExample],
    *,
    epochs: int,
    batch_groups: int,
    seed: int,
) -> Tuple[Tuple[int, ...], ...]:
    groups: Dict[str, list[int]] = defaultdict(list)
    for index, example in enumerate(examples):
        groups[example.step_group_id].append(index)
    strata: Dict[int, list[Tuple[int, ...]]] = defaultdict(list)
    for group_id in sorted(groups):
        indices = tuple(
            sorted(groups[group_id], key=lambda index: examples[index].candidate_index)
        )
        if len(indices) != 3:
            raise ValueError(f"SG10 group {group_id} must contain three candidates")
        lengths = {len(examples[index].prompt_ids) for index in indices}
        if len(lengths) != 1:
            raise ValueError(f"SG10 group {group_id} has mixed lengths")
        strata[next(iter(lengths))].append(indices)
    if any(len(values) % batch_groups for values in strata.values()):
        raise ValueError("every SG10 length stratum must divide batch_groups")
    generator = random.Random(seed)
    schedule = []
    for _epoch in range(epochs):
        lengths = sorted(strata)
        generator.shuffle(lengths)
        for length in lengths:
            shuffled = list(range(len(strata[length])))
            generator.shuffle(shuffled)
            for start in range(0, len(shuffled), batch_groups):
                schedule.append(
                    tuple(
                        example_index
                        for group_index in shuffled[start : start + batch_groups]
                        for example_index in strata[length][group_index]
                    )
                )
    return tuple(schedule)


def _batch_tensors(
    examples: Sequence[MultiChannelExample],
    indices: Sequence[int],
    *,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    selected = tuple(examples[index] for index in indices)
    lengths = {len(example.prompt_ids) for example in selected}
    if len(lengths) != 1:
        raise ValueError("SG10 batch must be length homogeneous")
    length = next(iter(lengths))
    input_ids = torch.tensor(
        [example.prompt_ids for example in selected],
        dtype=torch.long,
        device=device,
    )
    query_indices = torch.tensor(
        (length - 2, length - 1), dtype=torch.long, device=device
    )
    targets = torch.tensor(
        [example.target_indices for example in selected],
        dtype=torch.long,
        device=device,
    )
    return input_ids, query_indices, targets


def multichannel_logits(
    model: MultiChannelBilinearModel,
    input_ids: torch.Tensor,
    query_indices: torch.Tensor,
    *,
    use_eligibility: bool,
    detach_state: bool,
) -> Tuple[torch.Tensor, Any]:
    hidden, state = _query_hidden(
        model.language_model,
        input_ids,
        query_indices,
        use_eligibility=use_eligibility,
        detach_state=detach_state,
    )
    return model.relation_head(hidden[:, 0], hidden[:, 1]), state


def build_class_weights(
    train: Sequence[MultiChannelExample],
    *,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    weights = {}
    for channel_index, (name, labels) in enumerate(CHANNEL_SPECS):
        counts = Counter(
            example.target_indices[channel_index] for example in train
        )
        if set(counts) != set(range(len(labels))):
            raise ValueError(f"SG10 train missing class in {name}: {counts}")
        values = torch.tensor(
            [len(train) / (len(labels) * counts[index]) for index in range(len(labels))],
            dtype=torch.float32,
            device=device,
        )
        weights[name] = values
    return weights


def _weighted_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    class_weights: Mapping[str, torch.Tensor],
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    losses = {}
    for channel_index, (name, _labels) in enumerate(CHANNEL_SPECS):
        start, stop = CHANNEL_OFFSETS[name]
        losses[name] = F.cross_entropy(
            logits[:, start:stop],
            targets[:, channel_index],
            weight=class_weights[name],
        )
    return torch.stack(tuple(losses.values())).mean(), losses


def _prediction_matrix(logits: torch.Tensor) -> torch.Tensor:
    return torch.stack(
        tuple(
            logits[:, CHANNEL_OFFSETS[name][0] : CHANNEL_OFFSETS[name][1]].argmax(
                dim=-1
            )
            for name, _labels in CHANNEL_SPECS
        ),
        dim=1,
    )


def _metric_accumulator() -> Dict[str, Any]:
    return {
        "count": 0,
        "exact_correct": 0,
        "channel_correct": Counter(),
        "class_correct": {name: Counter() for name, _labels in CHANNEL_SPECS},
        "class_count": {name: Counter() for name, _labels in CHANNEL_SPECS},
        "step_exact": defaultdict(list),
    }


def _accumulate_predictions(
    accumulator: Dict[str, Any],
    predictions: torch.Tensor,
    targets: torch.Tensor,
    group_ids: Sequence[str],
) -> None:
    correct = predictions == targets
    exact = correct.all(dim=1)
    accumulator["count"] += targets.shape[0]
    accumulator["exact_correct"] += int(exact.sum().item())
    for channel_index, (name, _labels) in enumerate(CHANNEL_SPECS):
        accumulator["channel_correct"][name] += int(
            correct[:, channel_index].sum().item()
        )
        for row in range(targets.shape[0]):
            target = int(targets[row, channel_index].item())
            accumulator["class_count"][name][target] += 1
            accumulator["class_correct"][name][target] += int(
                correct[row, channel_index].item()
            )
    for row, group_id in enumerate(group_ids):
        accumulator["step_exact"][group_id].append(bool(exact[row].item()))


def _finalize_metrics(accumulator: Mapping[str, Any]) -> Dict[str, Any]:
    count = accumulator["count"]
    channel_accuracy = {
        name: accumulator["channel_correct"][name] / count
        for name, _labels in CHANNEL_SPECS
    }
    class_recall = {
        name: {
            labels[index]: (
                accumulator["class_correct"][name][index]
                / accumulator["class_count"][name][index]
            )
            for index in range(len(labels))
        }
        for name, labels in CHANNEL_SPECS
    }
    return {
        "exact_vector_accuracy": accumulator["exact_correct"] / count,
        "macro_channel_accuracy": sg0._mean(channel_accuracy.values()),
        "channel_accuracy": channel_accuracy,
        "class_recall": class_recall,
        "step_group_exact_consistency": sum(
            all(values) for values in accumulator["step_exact"].values()
        )
        / len(accumulator["step_exact"]),
        "example_count": count,
        "step_group_count": len(accumulator["step_exact"]),
    }


def evaluate_multichannel(
    model: MultiChannelBilinearModel,
    examples: Sequence[MultiChannelExample],
    class_weights: Mapping[str, torch.Tensor],
    *,
    device: torch.device,
    include_records: bool,
) -> Dict[str, Any]:
    model.eval()
    accumulator = _metric_accumulator()
    loss_sums = Counter()
    with torch.inference_mode():
        by_length: Dict[int, list[int]] = defaultdict(list)
        for index, example in enumerate(examples):
            by_length[len(example.prompt_ids)].append(index)
        for length in sorted(by_length):
            values = by_length[length]
            for start_index in range(0, len(values), 96):
                indices = tuple(values[start_index : start_index + 96])
                input_ids, query_indices, targets = _batch_tensors(
                    examples, indices, device=device
                )
                logits, _state = multichannel_logits(
                    model,
                    input_ids,
                    query_indices,
                    use_eligibility=False,
                    detach_state=True,
                )
                _loss, channel_losses = _weighted_loss(
                    logits, targets, class_weights
                )
                for name, value in channel_losses.items():
                    loss_sums[name] += float(value.item()) * len(indices)
                predictions = _prediction_matrix(logits)
                _accumulate_predictions(
                    accumulator,
                    predictions,
                    targets,
                    tuple(examples[index].step_group_id for index in indices),
                )
    metrics = _finalize_metrics(accumulator)
    metrics["weighted_channel_nll"] = {
        name: loss_sums[name] / len(examples) for name, _labels in CHANNEL_SPECS
    }
    metrics["mean_weighted_channel_nll"] = sg0._mean(
        metrics["weighted_channel_nll"].values()
    )

    timings = []
    state_sizes = []
    records = []
    with torch.inference_mode():
        for index, example in enumerate(examples):
            input_ids, query_indices, targets = _batch_tensors(
                examples, (index,), device=device
            )
            _sync(device)
            started = time.perf_counter_ns()
            logits, state = multichannel_logits(
                model,
                input_ids,
                query_indices,
                use_eligibility=False,
                detach_state=True,
            )
            logits.sum().item()
            _sync(device)
            timings.append((time.perf_counter_ns() - started) / 1e6)
            state_sizes.append(state_nbytes(state))
            if include_records:
                predicted = _prediction_matrix(logits)[0]
                records.append(
                    {
                        "example_id": example.example_id,
                        "targets": example.target_labels,
                        "predictions": tuple(
                            labels[int(predicted[channel_index].item())]
                            for channel_index, (_name, labels) in enumerate(
                                CHANNEL_SPECS
                            )
                        ),
                        "exact_correct": bool(
                            (predicted == targets[0]).all().item()
                        ),
                    }
                )
    metrics["timing"] = {
        **_sample_summary(timings, 1),
        "p99_ms": _percentile(timings, 0.99),
        "state_bytes_max": max(state_sizes),
        "state_bytes_mean": sg0._mean(state_sizes),
    }
    metrics["records"] = records if include_records else None
    return metrics


def train_multichannel(
    name: str,
    model: MultiChannelBilinearModel,
    examples: Sequence[MultiChannelExample],
    schedule: Sequence[Sequence[int]],
    class_weights: Mapping[str, torch.Tensor],
    *,
    epochs: int,
    batches_per_epoch: int,
    device: torch.device,
) -> Dict[str, Any]:
    model.train(True)
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        parameters, lr=1e-3, weight_decay=0.01, fused=True
    )
    batch_timings = []
    example_timings = []
    losses = []
    example_exposures = 0
    epoch_loss = 0.0
    epoch_examples = 0
    epoch_records = []
    started_all = time.perf_counter_ns()
    for update, indices in enumerate(schedule):
        input_ids, query_indices, targets = _batch_tensors(
            examples, indices, device=device
        )
        _sync(device)
        started = time.perf_counter_ns()
        optimizer.zero_grad(set_to_none=True)
        logits, _state = multichannel_logits(
            model,
            input_ids,
            query_indices,
            use_eligibility=name in ("snn_at1", "snn_ra0"),
            detach_state=True,
        )
        loss, _channel_losses = _weighted_loss(logits, targets, class_weights)
        if not torch.isfinite(loss):
            raise FloatingPointError(
                f"non-finite SG10 loss for {name} at update {update + 1}"
            )
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            parameters, 1.0, foreach=True
        )
        optimizer.step()
        _sync(device)
        elapsed_ms = (time.perf_counter_ns() - started) / 1e6
        batch_size = len(indices)
        batch_timings.append(elapsed_ms)
        example_timings.append(elapsed_ms / batch_size)
        value = float(loss.detach().item())
        losses.append(value)
        example_exposures += batch_size
        epoch_loss += value * batch_size
        epoch_examples += batch_size
        if (update + 1) % batches_per_epoch == 0:
            epoch_records.append(
                {
                    "epoch": len(epoch_records) + 1,
                    "weighted_nll": epoch_loss / epoch_examples,
                    "example_exposures": epoch_examples,
                    "last_gradient_norm": float(gradient_norm),
                }
            )
            epoch_loss = 0.0
            epoch_examples = 0
    elapsed_seconds = (time.perf_counter_ns() - started_all) / 1e9
    warmup = len(batch_timings) // 5
    return {
        "epochs": epochs,
        "updates": len(schedule),
        "batches_per_epoch": batches_per_epoch,
        "batch_examples": len(schedule[0]),
        "example_exposures": example_exposures,
        "loss_first": losses[0],
        "loss_last": losses[-1],
        "loss_last_epoch": epoch_records[-1]["weighted_nll"],
        "batch_timing": {
            **_sample_summary(batch_timings[warmup:], 1),
            "warmup_updates_excluded": warmup,
        },
        "example_equivalent_timing": {
            **_sample_summary(example_timings[warmup:], 1),
            "warmup_updates_excluded": warmup,
        },
        "elapsed_seconds": elapsed_seconds,
        "examples_per_second_total": example_exposures / elapsed_seconds,
        "epoch_records": epoch_records,
    }


def _extract_features(
    language_model: Any,
    examples: Sequence[MultiChannelExample],
    *,
    device: torch.device,
) -> Dict[str, Any]:
    language_model.eval()
    features = []
    targets = []
    groups = []
    started = time.perf_counter_ns()
    by_length: Dict[int, list[int]] = defaultdict(list)
    for index, example in enumerate(examples):
        by_length[len(example.prompt_ids)].append(index)
    with torch.inference_mode():
        for length in sorted(by_length):
            values = by_length[length]
            for start_index in range(0, len(values), 96):
                indices = tuple(values[start_index : start_index + 96])
                input_ids, query_indices, target = _batch_tensors(
                    examples, indices, device=device
                )
                hidden, _state = _query_hidden(
                    language_model,
                    input_ids,
                    query_indices,
                    use_eligibility=False,
                    detach_state=True,
                )
                features.append(_outer_features(hidden).to(torch.float64))
                targets.append(target)
                groups.extend(examples[index].step_group_id for index in indices)
    _sync(device)
    return {
        "features": torch.cat(features),
        "targets": torch.cat(targets),
        "group_ids": tuple(groups),
        "elapsed_seconds": (time.perf_counter_ns() - started) / 1e9,
    }


def _ridge_target_code(targets: torch.Tensor) -> torch.Tensor:
    code = -torch.ones(
        targets.shape[0], TOTAL_LOGITS, dtype=torch.float64, device=targets.device
    )
    for channel_index, (name, _labels) in enumerate(CHANNEL_SPECS):
        start, _stop = CHANNEL_OFFSETS[name]
        code[
            torch.arange(targets.shape[0], device=targets.device),
            start + targets[:, channel_index],
        ] = 1.0
    return code


def _ridge_multichannel_metrics(
    scores: torch.Tensor,
    targets: torch.Tensor,
    group_ids: Sequence[str],
) -> Dict[str, Any]:
    predictions = _prediction_matrix(scores)
    accumulator = _metric_accumulator()
    _accumulate_predictions(accumulator, predictions, targets, group_ids)
    result = _finalize_metrics(accumulator)
    result["mse"] = float(
        F.mse_loss(scores, _ridge_target_code(targets)).item()
    )
    return result


def fit_multichannel_ridge(
    language_model: Any,
    examples: Mapping[str, Sequence[MultiChannelExample]],
    *,
    device: torch.device,
    lambdas: Sequence[float],
) -> Dict[str, Any]:
    extracted = {
        split: _extract_features(language_model, examples[split], device=device)
        for split in SPLITS
    }
    train_x = extracted["train"]["features"]
    train_targets = extracted["train"]["targets"]
    train_y = _ridge_target_code(train_targets)
    mean = train_x[:, 1:].mean(dim=0)
    scale = train_x[:, 1:].std(dim=0, unbiased=False).clamp_min(1e-8)

    def transform(values: torch.Tensor) -> torch.Tensor:
        return torch.cat(
            (values[:, :1], (values[:, 1:] - mean) / scale), dim=1
        )

    x_train = transform(train_x)
    x_valid = transform(extracted["valid"]["features"])
    x_test = transform(extracted["test"]["features"])
    identity = torch.eye(x_train.shape[0], dtype=torch.float64, device=device)
    gram = x_train @ x_train.T
    candidates = []
    weights_by_lambda = {}
    started_fit = time.perf_counter_ns()
    for ridge_lambda in lambdas:
        alpha = torch.linalg.solve(
            gram + float(ridge_lambda) * identity, train_y
        )
        weights = x_train.T @ alpha
        weights_by_lambda[float(ridge_lambda)] = weights
        candidates.append(
            {
                "lambda": float(ridge_lambda),
                "valid": _ridge_multichannel_metrics(
                    x_valid @ weights,
                    extracted["valid"]["targets"],
                    extracted["valid"]["group_ids"],
                ),
            }
        )
    selected = min(
        candidates,
        key=lambda record: (
            -record["valid"]["exact_vector_accuracy"],
            -record["valid"]["macro_channel_accuracy"],
            record["valid"]["mse"],
            record["lambda"],
        ),
    )
    weights = weights_by_lambda[selected["lambda"]]
    _sync(device)
    fit_seconds = (time.perf_counter_ns() - started_fit) / 1e9
    feature_seconds = sum(
        extracted[split]["elapsed_seconds"] for split in ("train", "valid")
    )
    return {
        "feature_dimension": x_train.shape[1],
        "output_dimension": TOTAL_LOGITS,
        "readout_parameter_count": weights.numel(),
        "selected_lambda": selected["lambda"],
        "lambda_candidates": tuple(float(value) for value in lambdas),
        "selection_rule": (
            "max valid exact, max valid macro, min MSE, min lambda"
        ),
        "validation_candidates": candidates,
        "train": _ridge_multichannel_metrics(
            x_train @ weights,
            train_targets,
            extracted["train"]["group_ids"],
        ),
        "valid": selected["valid"],
        "test": _ridge_multichannel_metrics(
            x_test @ weights,
            extracted["test"]["targets"],
            extracted["test"]["group_ids"],
        ),
        "feature_extraction_seconds": {
            split: extracted[split]["elapsed_seconds"] for split in SPLITS
        },
        "fit_seconds": fit_seconds,
        "training_wall_seconds": feature_seconds + fit_seconds,
    }


def evaluate_cached_multichannel(
    model: MultiChannelBilinearModel,
    examples: Sequence[MultiChannelExample],
    *,
    device: torch.device,
    use_cached_decay: bool,
    timing_repeats: int,
    timing_warmup_repeats: int,
) -> Dict[str, Any]:
    model.eval()
    groups: Dict[str, list[MultiChannelExample]] = defaultdict(list)
    for example in examples:
        groups[example.step_group_id].append(example)
    accumulator = _metric_accumulator()
    timings = []
    prefix_timings = []
    state_sizes = []
    max_difference = 0.0
    decays = None
    if use_cached_decay:
        core = model.language_model.core
        if not isinstance(core, E3GatedTraceScanCore):
            raise TypeError("SG10 cached decay is SNN-only")
        decays = core.decays()
    with torch.inference_mode():
        for group_id in sorted(groups):
            group = sorted(groups[group_id], key=lambda value: value.candidate_index)
            context = group[0].prompt_ids[:-1]
            if any(example.prompt_ids[:-1] != context for example in group):
                raise ValueError("SG10 cached group context mismatch")
            prefix_input = torch.tensor([context], dtype=torch.long, device=device)
            prefix_query = torch.tensor(
                (len(context) - 1,), dtype=torch.long, device=device
            )
            _sync(device)
            prefix_started = time.perf_counter_ns()
            hidden, prefix_state = _query_hidden(
                model.language_model,
                prefix_input,
                prefix_query,
                use_eligibility=False,
                detach_state=True,
            )
            previous_hidden = hidden[:, 0]
            previous_hidden.sum().item()
            _sync(device)
            prefix_timings.append(
                (time.perf_counter_ns() - prefix_started) / 1e6
            )
            state_sizes.append(state_nbytes(prefix_state))
            for example in group:
                candidate_id = example.prompt_ids[-1]

                def forward_candidate() -> torch.Tensor:
                    if use_cached_decay:
                        if not isinstance(prefix_state, E3ScanState):
                            raise TypeError("SG10 SNN prefix returned invalid state")
                        candidate_hidden, _next = _snn_cached_decay_candidate_hidden(
                            model.language_model,
                            candidate_id,
                            prefix_state,
                            decays,  # type: ignore[arg-type]
                            device=device,
                        )
                    else:
                        candidate_hidden, _next = _generic_candidate_hidden(
                            model.language_model,
                            candidate_id,
                            prefix_state,
                            device=device,
                        )
                    return model.relation_head(previous_hidden, candidate_hidden)

                logits = forward_candidate()
                prediction = _prediction_matrix(logits)
                target = torch.tensor(
                    [example.target_indices], dtype=torch.long, device=device
                )
                _accumulate_predictions(
                    accumulator, prediction, target, (example.step_group_id,)
                )
                for repeat in range(timing_warmup_repeats + timing_repeats):
                    _sync(device)
                    started = time.perf_counter_ns()
                    timed_logits = forward_candidate()
                    timed_logits.sum().item()
                    _sync(device)
                    if repeat >= timing_warmup_repeats:
                        timings.append((time.perf_counter_ns() - started) / 1e6)
                full_input, full_query, _targets = _batch_tensors(
                    examples, (examples.index(example),), device=device
                )
                full_logits, _state = multichannel_logits(
                    model,
                    full_input,
                    full_query,
                    use_eligibility=False,
                    detach_state=True,
                )
                max_difference = max(
                    max_difference,
                    float((logits - full_logits).abs().max().item()),
                )
    result = _finalize_metrics(accumulator)
    result.update(
        {
            "mode": (
                "snn_cached_decay" if use_cached_decay else "generic_cached_state"
            ),
            "max_full_logit_abs_difference": max_difference,
            "prefix_timing": {
                **_sample_summary(prefix_timings, 1),
                "p99_ms": _percentile(prefix_timings, 0.99),
            },
            "candidate_timing": {
                **_sample_summary(timings, 1),
                "p99_ms": _percentile(timings, 0.99),
            },
            "candidate_timing_sample_count": len(timings),
            "prefix_state_bytes_max": max(state_sizes),
        }
    )
    return result


def _meets_task(metrics: Mapping[str, Any]) -> bool:
    return (
        metrics["exact_vector_accuracy"] >= 0.90
        and all(value >= 0.95 for value in metrics["channel_accuracy"].values())
        and metrics["class_recall"]["reward"][REWARD_LABELS[1]] >= 0.90
        and metrics["class_recall"]["done"][DONE_LABELS[1]] >= 0.90
    )


def _decision(
    data_audit: Mapping[str, Any],
    seeds: Sequence[Mapping[str, Any]],
    *,
    quick: bool,
) -> Dict[str, Any]:
    if quick:
        return {
            "data_gate": "PASS" if data_audit["passed"] else "FAIL",
            "task_gate": "SMOKE",
            "snn_quality_gate": "SMOKE",
            "ridge_quality_gate": "SMOKE",
            "training_speed_gate": "SMOKE",
            "cached_stream_gate": "SMOKE",
            "overall": "SMOKE",
            "next_route": "formal_sg10_multichannel_delta",
        }
    mean_metrics = {}
    for name in MODEL_NAMES:
        mean_metrics[name] = {
            "exact_vector_accuracy": sg0._mean(
                seed["post"][name]["test"]["exact_vector_accuracy"]
                for seed in seeds
            ),
            "macro_channel_accuracy": sg0._mean(
                seed["post"][name]["test"]["macro_channel_accuracy"]
                for seed in seeds
            ),
            "channel_accuracy": {
                channel: sg0._mean(
                    seed["post"][name]["test"]["channel_accuracy"][channel]
                    for seed in seeds
                )
                for channel, _labels in CHANNEL_SPECS
            },
            "reward_positive_recall": sg0._mean(
                seed["post"][name]["test"]["class_recall"]["reward"]
                [REWARD_LABELS[1]]
                for seed in seeds
            ),
            "done_positive_recall": sg0._mean(
                seed["post"][name]["test"]["class_recall"]["done"]
                [DONE_LABELS[1]]
                for seed in seeds
            ),
        }
    qualified_ann = tuple(
        name
        for name in ("lstm", "transformer")
        if _meets_task(
            {
                **mean_metrics[name],
                "class_recall": {
                    "reward": {
                        REWARD_LABELS[1]: mean_metrics[name][
                            "reward_positive_recall"
                        ]
                    },
                    "done": {
                        DONE_LABELS[1]: mean_metrics[name]["done_positive_recall"]
                    },
                },
            }
        )
    )
    task_pass = bool(qualified_ann)
    best_ann_exact = max(
        mean_metrics[name]["exact_vector_accuracy"]
        for name in ("lstm", "transformer")
    )
    best_ann_macro = max(
        mean_metrics[name]["macro_channel_accuracy"]
        for name in ("lstm", "transformer")
    )
    ra0 = mean_metrics["snn_ra0"]
    snn_quality = (
        ra0["exact_vector_accuracy"] >= 0.90
        and ra0["macro_channel_accuracy"] >= 0.95
        and all(value >= 0.95 for value in ra0["channel_accuracy"].values())
        and ra0["reward_positive_recall"] >= 0.90
        and ra0["done_positive_recall"] >= 0.90
        and ra0["exact_vector_accuracy"] >= best_ann_exact - 0.02
        and ra0["macro_channel_accuracy"] >= best_ann_macro - 0.02
    )
    ridge_mean = {
        "exact_vector_accuracy": sg0._mean(
            seed["closed_form_ridge"]["test"]["exact_vector_accuracy"]
            for seed in seeds
        ),
        "macro_channel_accuracy": sg0._mean(
            seed["closed_form_ridge"]["test"]["macro_channel_accuracy"]
            for seed in seeds
        ),
        "channel_accuracy": {
            channel: sg0._mean(
                seed["closed_form_ridge"]["test"]["channel_accuracy"][channel]
                for seed in seeds
            )
            for channel, _labels in CHANNEL_SPECS
        },
        "reward_positive_recall": sg0._mean(
            seed["closed_form_ridge"]["test"]["class_recall"]["reward"]
            [REWARD_LABELS[1]]
            for seed in seeds
        ),
        "done_positive_recall": sg0._mean(
            seed["closed_form_ridge"]["test"]["class_recall"]["done"]
            [DONE_LABELS[1]]
            for seed in seeds
        ),
    }
    ridge_quality = (
        ridge_mean["exact_vector_accuracy"] >= 0.90
        and ridge_mean["macro_channel_accuracy"] >= 0.95
        and all(value >= 0.95 for value in ridge_mean["channel_accuracy"].values())
        and ridge_mean["reward_positive_recall"] >= 0.90
        and ridge_mean["done_positive_recall"] >= 0.90
        and ridge_mean["exact_vector_accuracy"] >= best_ann_exact - 0.02
        and ridge_mean["macro_channel_accuracy"] >= best_ann_macro - 0.02
    )
    mean_train_p50 = {
        name: sg0._mean(
            seed["training"][name]["example_equivalent_timing"]["p50_ms"]
            for seed in seeds
        )
        for name in MODEL_NAMES
    }
    mean_train_wall = {
        name: sg0._mean(seed["training"][name]["elapsed_seconds"] for seed in seeds)
        for name in MODEL_NAMES
    }
    ann_for_speed = qualified_ann or ("lstm", "transformer")
    fastest_ann_p50 = min(mean_train_p50[name] for name in ann_for_speed)
    fastest_ann_wall = min(mean_train_wall[name] for name in ann_for_speed)
    ridge_wall = sg0._mean(
        seed["closed_form_ridge"]["training_wall_seconds"] for seed in seeds
    )
    training_speed = (
        mean_train_p50["snn_ra0"] <= fastest_ann_p50
        and mean_train_wall["snn_ra0"] <= fastest_ann_wall
    ) or ridge_wall <= fastest_ann_wall
    per_seed_stream = []
    for seed in seeds:
        ra0_stream = seed["cached_stream"]["snn_ra0"]["cached_decay"]
        ann_p50 = min(
            seed["cached_stream"][name]["generic"]["candidate_timing"][
                "p50_ms"
            ]
            for name in ann_for_speed
        )
        ann_p95 = min(
            seed["cached_stream"][name]["generic"]["candidate_timing"][
                "p95_ms"
            ]
            for name in ann_for_speed
        )
        per_seed_stream.append(
            {
                "seed": seed["seed"],
                "ra0_p50_ms": ra0_stream["candidate_timing"]["p50_ms"],
                "ra0_p95_ms": ra0_stream["candidate_timing"]["p95_ms"],
                "fastest_ann_p50_ms": ann_p50,
                "fastest_ann_p95_ms": ann_p95,
                "quality_passed": _meets_task(ra0_stream),
                "equivalent": ra0_stream["max_full_logit_abs_difference"] <= 1e-5,
                "passed": _meets_task(ra0_stream)
                and ra0_stream["max_full_logit_abs_difference"] <= 1e-5
                and ra0_stream["candidate_timing"]["p50_ms"] <= ann_p50
                and ra0_stream["candidate_timing"]["p95_ms"] <= ann_p95,
            }
        )
    cached_stream = all(record["passed"] for record in per_seed_stream)
    gates = {
        "data_gate": bool(data_audit["passed"]),
        "task_gate": task_pass,
        "snn_quality_gate": snn_quality,
        "ridge_quality_gate": ridge_quality,
        "training_speed_gate": training_speed,
        "cached_stream_gate": cached_stream,
    }
    overall = "PASS" if all(gates.values()) else "FAIL"
    return {
        **{name: "PASS" if value else "FAIL" for name, value in gates.items()},
        "overall": overall,
        "mean_test_metrics": mean_metrics,
        "mean_ridge_test_metrics": ridge_mean,
        "qualified_ann_models": qualified_ann,
        "best_ann_exact_vector_accuracy": best_ann_exact,
        "best_ann_macro_channel_accuracy": best_ann_macro,
        "mean_training_example_p50_ms": mean_train_p50,
        "mean_training_wall_seconds": mean_train_wall,
        "mean_ridge_training_wall_seconds": ridge_wall,
        "per_seed_stream_comparison": per_seed_stream,
        "next_route": (
            "sg11_recursive_block_woodbury_closed_loop_rollout"
            if overall == "PASS"
            else "sg11_multiscale_state_or_compiled_event_step"
        ),
    }


def run_experiment(args: argparse.Namespace) -> Dict[str, Any]:
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    device = torch.device(
        "cuda"
        if args.device == "cuda" or args.device == "auto" and torch.cuda.is_available()
        else "cpu"
    )
    if device.type == "cpu":
        torch.set_num_threads(args.threads)
    corpus_root = args.corpus_dir.expanduser().resolve()
    expected_data_seeds = {
        "train": tuple(args.expected_train_seeds),
        "valid": tuple(args.expected_valid_seeds),
        "test": tuple(args.expected_test_seeds),
    }
    manifest = tw0._manifest_provenance(
        corpus_root, expected_seeds_by_split=expected_data_seeds
    )
    corpus = load_event_corpus(corpus_root)
    examples, vocabulary = build_multichannel_examples(corpus_root, corpus)
    expected_counts = dict(zip(SPLITS, args.expected_counts))
    expected_groups = dict(zip(SPLITS, args.expected_groups))
    data_audit = audit_multichannel_examples(
        examples,
        vocabulary,
        expected_counts=expected_counts,
        expected_groups=expected_groups,
    )
    if not data_audit["passed"]:
        raise AssertionError("SG10 multichannel data audit failed")
    class_weights = build_class_weights(examples["train"], device=device)
    batches_per_epoch = 10

    seed_results = []
    for seed in args.seeds:
        models = build_multichannel_models(
            10_200_000 + 100 * seed,
            vocabulary,
            d_model=args.d_model,
            state_dim=args.state_dim,
            num_heads=args.num_heads,
            device=device,
        )
        ridge_reservoir = copy.deepcopy(models["snn_ra0"].language_model)
        parameter_counts = {
            name: {
                "total": count_parameters(model),
                "core": count_parameters(model.language_model.core),
                "relation_head": count_parameters(model.relation_head),
            }
            for name, model in models.items()
        }
        totals = tuple(record["total"] for record in parameter_counts.values())
        parameter_spread = (max(totals) - min(totals)) / sg0._mean(totals)
        if parameter_spread > 0.03:
            raise AssertionError(f"SG10 parameter spread failed: {parameter_counts}")
        schedule = build_length_stratified_schedule(
            examples["train"],
            epochs=args.epochs,
            batch_groups=args.batch_groups,
            seed=10_201_000 + seed,
        )
        if len(schedule) != batches_per_epoch * args.epochs:
            raise AssertionError("SG10 schedule update count mismatch")
        training = {
            name: train_multichannel(
                name,
                model,
                examples["train"],
                schedule,
                class_weights,
                epochs=args.epochs,
                batches_per_epoch=batches_per_epoch,
                device=device,
            )
            for name, model in models.items()
        }
        post = {
            name: {
                split: evaluate_multichannel(
                    model,
                    examples[split],
                    class_weights,
                    device=device,
                    include_records=split == "test",
                )
                for split in ("train", "valid", "test")
            }
            for name, model in models.items()
        }
        cached_stream = {}
        for name, model in models.items():
            cached_stream[name] = {
                "generic": evaluate_cached_multichannel(
                    model,
                    examples["test"],
                    device=device,
                    use_cached_decay=False,
                    timing_repeats=args.timing_repeats,
                    timing_warmup_repeats=args.timing_warmup_repeats,
                )
            }
            if name.startswith("snn_"):
                cached_stream[name]["cached_decay"] = (
                    evaluate_cached_multichannel(
                        model,
                        examples["test"],
                        device=device,
                        use_cached_decay=True,
                        timing_repeats=args.timing_repeats,
                        timing_warmup_repeats=args.timing_warmup_repeats,
                    )
                )
        closed_form = fit_multichannel_ridge(
            ridge_reservoir,
            examples,
            device=device,
            lambdas=args.ridge_lambdas,
        )
        seed_results.append(
            {
                "seed": seed,
                "parameter_counts": parameter_counts,
                "parameter_relative_spread": parameter_spread,
                "training": training,
                "post": post,
                "cached_stream": cached_stream,
                "closed_form_ridge": closed_form,
            }
        )
    decision = _decision(data_audit, seed_results, quick=args.quick)
    return {
        "schema_version": 1,
        "experiment": "E3-SG10 multichannel TextWorld event delta",
        "formal": not args.quick,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "environment": _environment(device),
        "research_hypothesis": {
            "epistemic_status": "event-driven multichannel extension",
            "statement": (
                "A bilinear persistent SNN event state can jointly predict "
                "spatial, value, terminal, and affordance deltas."
            ),
            "what_if": (
                "What if one sparse candidate event can update several world "
                "channels without sacrificing closed-form training or cached speed?"
            ),
        },
        "configuration": {
            "corpus_dir": str(corpus_root),
            "expected_data_seeds": expected_data_seeds,
            "expected_counts": expected_counts,
            "expected_groups": expected_groups,
            "epochs": args.epochs,
            "seeds": tuple(args.seeds),
            "threads": args.threads if device.type == "cpu" else None,
            "d_model": args.d_model,
            "state_dim": args.state_dim,
            "num_heads": args.num_heads,
            "batch_groups": args.batch_groups,
            "batch_examples": args.batch_groups * 3,
            "batches_per_epoch": batches_per_epoch,
            "optimizer_updates_per_model": batches_per_epoch * args.epochs,
            "example_exposures_per_model": len(examples["train"]) * args.epochs,
            "timing_repeats_per_candidate": args.timing_repeats,
            "timing_warmup_repeats_per_candidate": args.timing_warmup_repeats,
            "class_weights": {
                name: tuple(float(value) for value in tensor.cpu())
                for name, tensor in class_weights.items()
            },
            "ridge_lambdas": tuple(args.ridge_lambdas),
            "learning_rate": 1e-3,
            "weight_decay": 0.01,
            "gradient_clip": 1.0,
        },
        "dataset": {
            "synthetic": False,
            "manifest_provenance": manifest,
            "corpus_provenance": tw0._corpus_provenance(corpus),
            "vocabulary": {
                "size": len(vocabulary),
                "fingerprint": vocabulary.fingerprint,
                "source_split": "train",
                "source_fields": ("prior_factual_events", "candidate_event"),
            },
            "audit": data_audit,
        },
        "seeds": seed_results,
        "decision": decision,
    }


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-dir", type=Path, default=DEFAULT_CORPUS_DIR)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/e3_scan/e3_sg10_multichannel_delta.json"),
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=(0, 1, 2))
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--d-model", type=int, default=tw0.D_MODEL)
    parser.add_argument("--state-dim", type=int, default=tw0.STATE_DIM)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--batch-groups", type=int, default=16)
    parser.add_argument("--timing-repeats", type=int, default=128)
    parser.add_argument("--timing-warmup-repeats", type=int, default=16)
    parser.add_argument(
        "--ridge-lambdas", nargs="+", type=float, default=RIDGE_LAMBDAS
    )
    parser.add_argument(
        "--expected-counts",
        nargs=3,
        type=int,
        default=tuple(EXPECTED_COUNTS[split] for split in SPLITS),
    )
    parser.add_argument(
        "--expected-groups",
        nargs=3,
        type=int,
        default=tuple(EXPECTED_GROUPS[split] for split in SPLITS),
    )
    parser.add_argument(
        "--expected-train-seeds",
        nargs="+",
        type=int,
        default=EXPECTED_DATA_SEEDS["train"],
    )
    parser.add_argument(
        "--expected-valid-seeds",
        nargs="+",
        type=int,
        default=EXPECTED_DATA_SEEDS["valid"],
    )
    parser.add_argument(
        "--expected-test-seeds",
        nargs="+",
        type=int,
        default=EXPECTED_DATA_SEEDS["test"],
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args(argv)
    if min(
        args.epochs,
        args.threads,
        args.d_model,
        args.state_dim,
        args.num_heads,
        args.batch_groups,
        args.timing_repeats,
        *args.ridge_lambdas,
        *args.expected_counts,
        *args.expected_groups,
        *args.expected_train_seeds,
        *args.expected_valid_seeds,
        *args.expected_test_seeds,
    ) <= 0:
        parser.error("all numeric experiment controls must be positive")
    if args.timing_warmup_repeats < 0:
        parser.error("timing-warmup-repeats must be nonnegative")
    if args.d_model % args.num_heads:
        parser.error("d-model must be divisible by num-heads")
    if args.quick:
        args.seeds = args.seeds[:1]
        args.epochs = 2
        args.timing_repeats = 1
        args.timing_warmup_repeats = 0
    return args


def main() -> None:
    args = _parse_args()
    result = run_experiment(args)
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result["decision"], indent=2, sort_keys=True))
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
