"""SG0 action-conditioned TextWorld counterfactual sequence generation."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import random
import re
import sys
import time
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple

import torch
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
from experiments import e3_tw0_sparse_event_lm as tw0  # noqa: E402
from experiments.e3_ra0_reverse_adjoint import (  # noqa: E402
    build_textworld_models,
)
from vpsc.world_model.cores import (  # noqa: E402
    E3GatedTraceScanCore,
    E3ScanState,
    count_parameters,
    state_nbytes,
)
from vpsc.world_model.event_corpus import (  # noqa: E402
    TextWorldEventCorpus,
    load_event_corpus,
)
from vpsc.world_model.lm import CausalLanguageModel  # noqa: E402
from vpsc.world_model.wikitext import SPLITS, Vocabulary  # noqa: E402


MODEL_NAMES = ("snn_bptt", "snn_at1", "snn_ra0", "lstm", "transformer")
MAX_GENERATION_TOKENS = 80
EXPECTED_COUNTS = {"train": 40, "valid": 10, "test": 10}
EXPECTED_PAIRS = {"train": 20, "valid": 5, "test": 5}
_ROOM_HEADER = re.compile(r"^-=.+=-$")
_CONTENT_WORD = re.compile(r"[A-Za-z]{2}")
_DIRECTIONS = frozenset(("north", "south", "east", "west"))


@dataclass(frozen=True)
class CounterfactualExample:
    split: str
    episode_index: int
    game_seed: int
    step_index: int
    counterfactual_index: int
    pair_id: str
    action: str
    action_type: str
    observation_text: str
    target_text: str
    observation_ids: Tuple[int, ...]
    prompt_ids: Tuple[int, ...]
    target_ids: Tuple[int, ...]
    prompt_unknowns: int
    target_unknowns: int

    @property
    def example_id(self) -> str:
        return (
            f"{self.split}:{self.game_seed}:{self.step_index}:"
            f"{self.counterfactual_index}"
        )

    @property
    def input_length(self) -> int:
        return len(self.prompt_ids) + len(self.target_ids) - 1


@dataclass(frozen=True)
class _RawCounterfactualExample:
    split: str
    episode_index: int
    game_seed: int
    step_index: int
    counterfactual_index: int
    pair_id: str
    action: str
    action_type: str
    observation_text: str
    target_text: str
    observation_tokens: Tuple[str, ...]
    prompt_tokens: Tuple[str, ...]
    target_tokens: Tuple[str, ...]


def normalize_textworld_observation(text: str) -> str:
    """Remove UI-only lines while retaining world-state language."""

    lines = []
    for raw in text.splitlines():
        line = " ".join(raw.split())
        if not line or line.startswith(">"):
            continue
        lines.append(line)
    for index, line in enumerate(lines):
        if _ROOM_HEADER.fullmatch(line):
            lines = lines[index:]
            break
    return "\n".join(lines)


def _action_type(action: str) -> str:
    return "move" if action.startswith("go ") else action.split()[0]


def _prompt_tokens(
    corpus: TextWorldEventCorpus,
    observation: str,
    action: str,
) -> Tuple[str, ...]:
    parts = (
        ("<bos>",),
        corpus.tokenizer.tokenize("observation:"),
        corpus.tokenizer.tokenize(observation),
        ("<eos>",),
        corpus.tokenizer.tokenize("action:"),
        corpus.tokenizer.tokenize(action),
        ("<eos>",),
        corpus.tokenizer.tokenize("next observation:"),
    )
    return tuple(token for part in parts for token in part)


def build_counterfactual_examples(
    root: Path,
    corpus: TextWorldEventCorpus,
) -> Tuple[Dict[str, Tuple[CounterfactualExample, ...]], Vocabulary]:
    raw_by_split: Dict[str, Tuple[_RawCounterfactualExample, ...]] = {}
    for split in SPLITS:
        path = root / split / "episodes.jsonl"
        raw_records = []
        for episode_index, line in enumerate(
            path.read_text(encoding="utf-8").splitlines()
        ):
            episode = json.loads(line)
            if episode.get("split") != split:
                raise ValueError(f"counterfactual episode split mismatch: {path}")
            game_seed = int(episode["seed"])
            for step_index, step in enumerate(episode["steps"]):
                observation = normalize_textworld_observation(step["observation"])
                if not observation:
                    raise ValueError(
                        f"empty normalized observation in {split} seed {game_seed}"
                    )
                for counterfactual_index, counterfactual in enumerate(
                    step["counterfactuals"]
                ):
                    target = normalize_textworld_observation(
                        counterfactual["next_obs"]
                    )
                    if not target:
                        raise ValueError(
                            f"empty normalized target in {split} seed {game_seed}"
                        )
                    action = str(counterfactual["action"])
                    prompt_tokens = _prompt_tokens(corpus, observation, action)
                    target_tokens = corpus.tokenizer.tokenize(target) + ("<eos>",)
                    observation_tokens = corpus.tokenizer.tokenize(observation)
                    raw_records.append(
                        _RawCounterfactualExample(
                            split=split,
                            episode_index=episode_index,
                            game_seed=game_seed,
                            step_index=step_index,
                            counterfactual_index=counterfactual_index,
                            pair_id=f"{split}:{game_seed}:{step_index}",
                            action=action,
                            action_type=_action_type(action),
                            observation_text=observation,
                            target_text=target,
                            observation_tokens=observation_tokens,
                            prompt_tokens=prompt_tokens,
                            target_tokens=target_tokens,
                        )
                    )
        raw_by_split[split] = tuple(raw_records)

    # The event corpus vocabulary tokenizes raw JSON strings, where escaped
    # newlines can join otherwise separate natural-language tokens.  SG0 uses
    # normalized observations, so its vocabulary must be rebuilt from the
    # normalized TRAIN prompts and targets only.  Validation/test never
    # contribute token identities or frequencies.
    vocabulary = Vocabulary.build(
        token
        for record in raw_by_split["train"]
        for token in record.prompt_tokens + record.target_tokens
    )
    result: Dict[str, Tuple[CounterfactualExample, ...]] = {}
    for split in SPLITS:
        records = []
        for raw in raw_by_split[split]:
            prompt_ids = vocabulary.encode(raw.prompt_tokens)
            target_ids = vocabulary.encode(raw.target_tokens)
            observation_ids = vocabulary.encode(raw.observation_tokens)
            records.append(
                CounterfactualExample(
                    split=raw.split,
                    episode_index=raw.episode_index,
                    game_seed=raw.game_seed,
                    step_index=raw.step_index,
                    counterfactual_index=raw.counterfactual_index,
                    pair_id=raw.pair_id,
                    action=raw.action,
                    action_type=raw.action_type,
                    observation_text=raw.observation_text,
                    target_text=raw.target_text,
                    observation_ids=observation_ids,
                    prompt_ids=prompt_ids,
                    target_ids=target_ids,
                    prompt_unknowns=sum(
                        value == vocabulary.unk_id for value in prompt_ids
                    ),
                    target_unknowns=sum(
                        value == vocabulary.unk_id for value in target_ids[:-1]
                    ),
                )
            )
        result[split] = tuple(records)
    return result, vocabulary


def _mean(values: Iterable[float]) -> float:
    values = tuple(float(value) for value in values)
    return math.fsum(values) / len(values)


def _length_summary(values: Sequence[int]) -> Dict[str, float | int]:
    ordered = sorted(values)
    return {
        "min": ordered[0],
        "mean": _mean(ordered),
        "p50": _percentile(ordered, 0.50),
        "p95": _percentile(ordered, 0.95),
        "max": ordered[-1],
    }


def _strip_eos(values: Sequence[int], eos_id: int) -> Tuple[int, ...]:
    result = []
    for value in values:
        if value == eos_id:
            break
        result.append(int(value))
    return tuple(result)


def _edit_distance(left: Sequence[int], right: Sequence[int]) -> int:
    previous = list(range(len(right) + 1))
    for left_index, left_value in enumerate(left, start=1):
        current = [left_index]
        for right_index, right_value in enumerate(right, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[right_index] + 1,
                    previous[right_index - 1] + int(left_value != right_value),
                )
            )
        previous = current
    return previous[-1]


def _lcs_length(left: Sequence[int], right: Sequence[int]) -> int:
    previous = [0] * (len(right) + 1)
    for left_value in left:
        current = [0]
        for index, right_value in enumerate(right, start=1):
            current.append(
                previous[index - 1] + 1
                if left_value == right_value
                else max(previous[index], current[-1])
            )
        previous = current
    return previous[-1]


def _token_f1(left: Sequence[int], right: Sequence[int]) -> float:
    left_counts = Counter(left)
    right_counts = Counter(right)
    overlap = sum((left_counts & right_counts).values())
    if not left and not right:
        return 1.0
    if overlap == 0:
        return 0.0
    precision = overlap / len(left)
    recall = overlap / len(right)
    return 2.0 * precision * recall / (precision + recall)


def _world_features(tokens: Sequence[str]) -> frozenset[str]:
    lowered = tuple(token.lower() for token in tokens)
    features = {f"direction:{token}" for token in lowered if token in _DIRECTIONS}
    if "coin" in lowered:
        features.add("object:coin")
    if "carrying" in lowered and "nothing" in lowered:
        features.add("inventory:empty")
    if "end" in lowered or "restart" in lowered or "scored" in lowered:
        features.add("terminal")
    for index in range(len(tokens) - 4):
        if tokens[index : index + 2] != ("-", "="):
            continue
        for stop in range(index + 3, len(tokens) - 1):
            if tokens[stop : stop + 2] == ("=", "-"):
                room = " ".join(token.lower() for token in tokens[index + 2 : stop])
                if room:
                    features.add(f"room:{room}")
                return frozenset(features)
    return frozenset(features)


def _feature_f1(
    prediction: frozenset[str], target: frozenset[str]
) -> float:
    if not prediction and not target:
        return 1.0
    overlap = len(prediction & target)
    if overlap == 0:
        return 0.0
    precision = overlap / len(prediction)
    recall = overlap / len(target)
    return 2.0 * precision * recall / (precision + recall)


def evaluate_predictions(
    examples: Sequence[CounterfactualExample],
    predictions: Mapping[str, Sequence[int]],
    vocabulary: Vocabulary,
    *,
    include_records: bool,
) -> Dict[str, Any]:
    records = []
    by_action: Dict[str, list[Mapping[str, float]]] = defaultdict(list)
    generated_by_pair: Dict[str, list[Tuple[Tuple[int, ...], Tuple[int, ...]]]] = (
        defaultdict(list)
    )
    for example in examples:
        prediction_with_eos = tuple(int(value) for value in predictions[example.example_id])
        target_with_eos = example.target_ids
        prediction = _strip_eos(prediction_with_eos, vocabulary.eos_id)
        target = _strip_eos(target_with_eos, vocabulary.eos_id)
        denominator = max(len(prediction), len(target), 1)
        edit_similarity = 1.0 - _edit_distance(prediction, target) / denominator
        lcs = _lcs_length(prediction, target)
        lcs_f1 = 1.0 if not prediction and not target else (
            2.0 * lcs / (len(prediction) + len(target))
            if prediction or target
            else 0.0
        )
        prediction_tokens = vocabulary.decode(prediction)
        target_tokens = vocabulary.decode(target)
        target_features = _world_features(target_tokens)
        prediction_features = _world_features(prediction_tokens)
        room_targets = {value for value in target_features if value.startswith("room:")}
        room_predictions = {
            value for value in prediction_features if value.startswith("room:")
        }
        metrics = {
            "exact": float(prediction_with_eos == target_with_eos),
            "edit_similarity": edit_similarity,
            "lcs_f1": lcs_f1,
            "token_f1": _token_f1(prediction, target),
            "feature_f1": _feature_f1(prediction_features, target_features),
            "room_accuracy": (
                float(room_predictions == room_targets) if room_targets else float("nan")
            ),
            "generated_length": float(len(prediction)),
            "target_length": float(len(target)),
            "stopped_on_eos": float(
                vocabulary.eos_id in prediction_with_eos
            ),
        }
        by_action[example.action_type].append(metrics)
        generated_by_pair[example.pair_id].append((prediction, target))
        if include_records:
            records.append(
                {
                    "example_id": example.example_id,
                    "pair_id": example.pair_id,
                    "action": example.action,
                    "action_type": example.action_type,
                    "target_tokens": target_tokens,
                    "prediction_tokens": prediction_tokens,
                    "target_features": sorted(target_features),
                    "prediction_features": sorted(prediction_features),
                    **metrics,
                }
            )

    flat = [metrics for values in by_action.values() for metrics in values]

    def aggregate(values: Sequence[Mapping[str, float]]) -> Dict[str, Any]:
        room_values = [value["room_accuracy"] for value in values if not math.isnan(value["room_accuracy"])]
        return {
            "exact": _mean(value["exact"] for value in values),
            "edit_similarity": _mean(value["edit_similarity"] for value in values),
            "lcs_f1": _mean(value["lcs_f1"] for value in values),
            "token_f1": _mean(value["token_f1"] for value in values),
            "feature_f1": _mean(value["feature_f1"] for value in values),
            "room_accuracy": _mean(room_values) if room_values else None,
            "mean_generated_length": _mean(
                value["generated_length"] for value in values
            ),
            "mean_target_length": _mean(value["target_length"] for value in values),
            "eos_rate": _mean(value["stopped_on_eos"] for value in values),
            "example_count": len(values),
        }

    sensitive = 0
    different_targets = 0
    for values in generated_by_pair.values():
        if len(values) != 2:
            raise AssertionError("each counterfactual pair must contain exactly two examples")
        predictions_pair = (values[0][0], values[1][0])
        targets_pair = (values[0][1], values[1][1])
        if targets_pair[0] != targets_pair[1]:
            different_targets += 1
            sensitive += int(predictions_pair[0] != predictions_pair[1])
    return {
        **aggregate(flat),
        "action_types": {
            name: aggregate(values) for name, values in sorted(by_action.items())
        },
        "paired_different_target_count": different_targets,
        "paired_action_sensitivity": (
            sensitive / different_targets if different_targets else 1.0
        ),
        "records": records if include_records else None,
    }


def _majority_targets(
    train: Sequence[CounterfactualExample],
) -> Dict[str, Tuple[int, ...]]:
    by_action: Dict[str, Counter[Tuple[int, ...]]] = defaultdict(Counter)
    for example in train:
        by_action[example.action_type][example.target_ids] += 1
    return {
        action_type: sorted(
            counts.items(), key=lambda item: (-item[1], item[0])
        )[0][0]
        for action_type, counts in by_action.items()
    }


def audit_examples(
    examples: Mapping[str, Sequence[CounterfactualExample]],
    vocabulary: Vocabulary,
) -> Dict[str, Any]:
    targets = {
        split: {example.target_text for example in values}
        for split, values in examples.items()
    }
    split_records = {}
    for split, values in examples.items():
        target_token_count = sum(len(value.target_ids) - 1 for value in values)
        split_records[split] = {
            "example_count": len(values),
            "pair_count": len({value.pair_id for value in values}),
            "action_types": dict(Counter(value.action_type for value in values)),
            "unique_target_count": len(targets[split]),
            "prompt_length": _length_summary(
                [len(value.prompt_ids) for value in values]
            ),
            "target_length_with_eos": _length_summary(
                [len(value.target_ids) for value in values]
            ),
            "input_length": _length_summary([value.input_length for value in values]),
            "prompt_unknown_count": sum(value.prompt_unknowns for value in values),
            "target_unknown_count": sum(value.target_unknowns for value in values),
            "target_token_count_without_eos": target_token_count,
            "target_unknown_ratio": (
                sum(value.target_unknowns for value in values) / target_token_count
            ),
            "format_only_target_count": sum(
                not _CONTENT_WORD.search(value.target_text) for value in values
            ),
        }
    overlap = {
        "valid_in_train": len(targets["valid"] & targets["train"]),
        "test_in_train": len(targets["test"] & targets["train"]),
        "valid_test": len(targets["valid"] & targets["test"]),
    }
    majority = _majority_targets(examples["train"])
    test_predictions = {
        "copy_observation": {
            value.example_id: value.observation_ids + (vocabulary.eos_id,)
            for value in examples["test"]
        },
        "action_majority": {
            value.example_id: majority[value.action_type]
            for value in examples["test"]
        },
    }
    baselines = {
        name: evaluate_predictions(
            examples["test"], predictions, vocabulary, include_records=False
        )
        for name, predictions in test_predictions.items()
    }
    data_pass = (
        all(len(examples[split]) == EXPECTED_COUNTS[split] for split in SPLITS)
        and all(
            len({value.pair_id for value in examples[split]}) == EXPECTED_PAIRS[split]
            for split in SPLITS
        )
        and max(
            len(value.prompt_ids)
            for values in examples.values()
            for value in values
        )
        <= 80
        and max(
            len(value.target_ids)
            for values in examples.values()
            for value in values
        )
        <= 70
        and split_records["valid"]["target_unknown_ratio"] < 0.10
        and split_records["test"]["target_unknown_ratio"] < 0.10
        and all(
            record["format_only_target_count"] == 0
            for record in split_records.values()
        )
        and overlap["test_in_train"] / len(targets["test"]) <= 0.20
    )
    return {
        "normalization": {
            "collapse_internal_whitespace": True,
            "remove_empty_lines": True,
            "remove_hud_lines_starting_with_prompt": True,
            "drop_prefix_before_first_room_header": True,
            "target_truncation": False,
        },
        "splits": split_records,
        "target_overlap": overlap,
        "baselines": baselines,
        "passed": data_pass,
    }


def _example_tensors(
    example: CounterfactualExample,
    *,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    sequence = example.prompt_ids + example.target_ids
    input_ids = torch.tensor(sequence[:-1], dtype=torch.long, device=device).unsqueeze(0)
    first_query = len(example.prompt_ids) - 1
    query_indices = torch.arange(
        first_query,
        first_query + len(example.target_ids),
        dtype=torch.long,
        device=device,
    )
    targets = torch.tensor(
        example.target_ids, dtype=torch.long, device=device
    ).unsqueeze(0)
    if query_indices.numel() != targets.numel():  # pragma: no cover
        raise AssertionError("counterfactual query/target alignment failed")
    return input_ids, query_indices, targets


def evaluate_teacher(
    model: CausalLanguageModel[Any],
    examples: Sequence[CounterfactualExample],
    *,
    device: torch.device,
) -> Dict[str, Any]:
    model.eval()
    total_loss = 0.0
    total_count = 0
    total_correct = 0
    action_loss: Dict[str, float] = defaultdict(float)
    action_count: Counter[str] = Counter()
    action_correct: Counter[str] = Counter()
    with torch.inference_mode():
        for example in examples:
            input_ids, query_indices, targets = _example_tensors(
                example, device=device
            )
            output = model(input_ids, None, detach_state=True)
            logits = output.logits.index_select(1, query_indices)
            losses = F.cross_entropy(
                logits.reshape(-1, model.vocab_size),
                targets.reshape(-1),
                reduction="none",
            )
            predictions = logits.argmax(dim=-1).reshape(-1)
            target_flat = targets.reshape(-1)
            count = target_flat.numel()
            total_loss += float(losses.sum().item())
            total_count += count
            correct = int((predictions == target_flat).sum().item())
            total_correct += correct
            action_loss[example.action_type] += float(losses.sum().item())
            action_count[example.action_type] += count
            action_correct[example.action_type] += correct
    nll = total_loss / total_count
    return {
        "nll": nll,
        "ppl": math.exp(min(nll, 80.0)),
        "accuracy": total_correct / total_count,
        "target_count": total_count,
        "action_types": {
            name: {
                "nll": action_loss[name] / action_count[name],
                "accuracy": action_correct[name] / action_count[name],
                "target_count": action_count[name],
            }
            for name in sorted(action_count)
        },
    }


def _training_schedule(count: int, epochs: int, seed: int) -> Tuple[int, ...]:
    generator = random.Random(seed)
    schedule = []
    for _ in range(epochs):
        indices = list(range(count))
        generator.shuffle(indices)
        schedule.extend(indices)
    return tuple(schedule)


def train_model(
    name: str,
    model: CausalLanguageModel[Any],
    examples: Sequence[CounterfactualExample],
    schedule: Sequence[int],
    *,
    epochs: int,
    device: torch.device,
) -> Dict[str, Any]:
    model.train(True)
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        parameters, lr=1e-3, weight_decay=0.01, fused=True
    )
    timings = []
    losses = []
    input_tokens = 0
    target_tokens = 0
    epoch_loss = 0.0
    epoch_targets = 0
    epoch_records = []
    started_all = time.perf_counter_ns()
    for update, example_index in enumerate(schedule):
        example = examples[example_index]
        input_ids, query_indices, targets = _example_tensors(
            example, device=device
        )
        _sync(device)
        started = time.perf_counter_ns()
        optimizer.zero_grad(set_to_none=True)
        logits, _state = tw0._sparse_forward(
            model,
            input_ids,
            query_indices,
            None,
            use_eligibility=name in ("snn_at1", "snn_ra0"),
            detach_state=True,
        )
        loss = F.cross_entropy(
            logits.reshape(-1, model.vocab_size), targets.reshape(-1)
        )
        if not torch.isfinite(loss):
            raise FloatingPointError(
                f"non-finite SG0 loss for {name} at update {update + 1}"
            )
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            parameters, 1.0, foreach=True
        )
        optimizer.step()
        _sync(device)
        timings.append((time.perf_counter_ns() - started) / 1e6)
        value = float(loss.detach().item())
        losses.append(value)
        count = targets.numel()
        input_tokens += input_ids.numel()
        target_tokens += count
        epoch_loss += value * count
        epoch_targets += count
        if (update + 1) % len(examples) == 0:
            epoch_records.append(
                {
                    "epoch": len(epoch_records) + 1,
                    "target_nll": epoch_loss / epoch_targets,
                    "target_count": epoch_targets,
                    "last_gradient_norm": float(gradient_norm),
                }
            )
            epoch_loss = 0.0
            epoch_targets = 0
    elapsed_seconds = (time.perf_counter_ns() - started_all) / 1e9
    warmup_updates = len(timings) // 5
    steady = timings[warmup_updates:]
    timing = _sample_summary(steady, 1)
    timing.update(
        {
            "warmup_updates_excluded": warmup_updates,
            "input_tokens_per_second_total": input_tokens / elapsed_seconds,
            "target_tokens_per_second_total": target_tokens / elapsed_seconds,
        }
    )
    return {
        "epochs": epochs,
        "updates": len(schedule),
        "input_tokens": input_tokens,
        "target_tokens": target_tokens,
        "loss_first": losses[0],
        "loss_last": losses[-1],
        "loss_last_epoch": epoch_records[-1]["target_nll"],
        "timing": timing,
        "elapsed_seconds": elapsed_seconds,
        "epoch_records": epoch_records,
    }


class _CachedSNNGenerationStepper:
    def __init__(
        self, model: CausalLanguageModel[Any], state: E3ScanState
    ) -> None:
        if not isinstance(model.core, E3GatedTraceScanCore):  # pragma: no cover
            raise TypeError("cached generation requires gated trace core")
        self.model = model
        self.core = model.core
        projected = tw0._project_chunk_state(model, state)
        self.excitatory = projected.layers[0].excitatory
        self.inhibitory = projected.layers[0].inhibitory
        self.decay_e, self.decay_i = self.core.decays()

    def step(self, token_id: int) -> torch.Tensor:
        token = torch.tensor(
            [token_id], dtype=torch.long, device=self.excitatory.device
        )
        embedded = self.model.input_dropout(self.model.embedding(token))
        output = self.core.forward_step_tensors_cached_decay(
            embedded,
            self.excitatory,
            self.inhibitory,
            self.decay_e,
            self.decay_i,
        )
        self.excitatory, self.inhibitory = output[1], output[2]
        hidden = self.model.output_norm(self.model.output_dropout(output[0]))
        return self.model.lm_head(hidden)

    def state_bytes(self) -> int:
        return sum(
            value.numel() * value.element_size()
            for value in (self.excitatory, self.inhibitory)
        )


class _GenericGenerationStepper:
    def __init__(self, model: CausalLanguageModel[Any], state: Any) -> None:
        self.model = model
        self.state = state

    def step(self, token_id: int) -> torch.Tensor:
        token = torch.tensor(
            [token_id], dtype=torch.long, device=self.model.embedding.weight.device
        )
        result = self.model.step(token, self.state, detach_state=True)
        self.state = result.state
        return result.logits

    def state_bytes(self) -> int:
        return state_nbytes(self.state)


def generate_model(
    model: CausalLanguageModel[Any],
    examples: Sequence[CounterfactualExample],
    vocabulary: Vocabulary,
    *,
    max_tokens: int,
    device: torch.device,
    include_records: bool,
) -> Dict[str, Any]:
    model.eval()
    predictions: Dict[str, Tuple[int, ...]] = {}
    prefill_samples = []
    token_samples = []
    state_sizes = []
    with torch.inference_mode():
        for example in examples:
            prompt = torch.tensor(
                example.prompt_ids, dtype=torch.long, device=device
            ).unsqueeze(0)
            _sync(device)
            started = time.perf_counter_ns()
            output = model(prompt, None, detach_state=True)
            logits = output.logits[:, -1]
            state = output.state
            logits.sum().item()
            _sync(device)
            prefill_samples.append((time.perf_counter_ns() - started) / 1e6)
            if isinstance(model.core, E3GatedTraceScanCore):
                if not isinstance(state, E3ScanState):  # pragma: no cover
                    raise TypeError("gated trace prefill returned invalid state")
                stepper: Any = _CachedSNNGenerationStepper(model, state)
            else:
                stepper = _GenericGenerationStepper(model, state)
            generated = []
            next_id = int(logits.argmax(dim=-1).item())
            for _ in range(max_tokens):
                generated.append(next_id)
                if next_id == vocabulary.eos_id:
                    break
                _sync(device)
                started = time.perf_counter_ns()
                logits = stepper.step(next_id)
                logits.sum().item()
                _sync(device)
                token_samples.append((time.perf_counter_ns() - started) / 1e6)
                next_id = int(logits.argmax(dim=-1).item())
            state_sizes.append(stepper.state_bytes())
            predictions[example.example_id] = tuple(generated)
    metrics = evaluate_predictions(
        examples, predictions, vocabulary, include_records=include_records
    )
    metrics["timing"] = {
        "prefill": {
            **_sample_summary(prefill_samples, 1),
            "p99_ms": _percentile(prefill_samples, 0.99),
        },
        "generated_token": {
            **_sample_summary(token_samples, 1),
            "p99_ms": _percentile(token_samples, 0.99),
        },
        "state_bytes_max": max(state_sizes),
        "state_bytes_mean": _mean(state_sizes),
    }
    return metrics


def _decision(
    *,
    data_audit: Mapping[str, Any],
    seed_results: Sequence[Mapping[str, Any]],
    quick: bool,
) -> Dict[str, Any]:
    if quick:
        return {
            "data_gate": "PASS" if data_audit["passed"] else "FAIL",
            "task_gate": "SMOKE",
            "quality_gate": "SMOKE",
            "speed_gate": "SMOKE",
            "stream_gate": "SMOKE",
            "overall": "SMOKE",
            "next_route": "formal_sg0",
        }
    task_improvement = all(
        seed["post_teacher"][name]["test"]["nll"]
        <= seed["pre_teacher"][name]["test"]["nll"] - 0.10
        for seed in seed_results
        for name in ("lstm", "transformer")
    )
    baseline_edit = max(
        data_audit["baselines"][name]["edit_similarity"]
        for name in ("copy_observation", "action_majority")
    )
    mean_nll = {
        name: _mean(
            seed["post_teacher"][name]["test"]["nll"]
            for seed in seed_results
        )
        for name in MODEL_NAMES
    }
    mean_edit = {
        name: _mean(
            seed["generation"][name]["edit_similarity"]
            for seed in seed_results
        )
        for name in MODEL_NAMES
    }
    best_ann_nll = min(mean_nll["lstm"], mean_nll["transformer"])
    best_ann_edit = max(mean_edit["lstm"], mean_edit["transformer"])
    task_generation = best_ann_edit >= baseline_edit + 0.05
    ra0_improvement = all(
        seed["post_teacher"]["snn_ra0"]["test"]["nll"]
        <= seed["pre_teacher"]["snn_ra0"]["test"]["nll"] - 0.10
        for seed in seed_results
    )
    nll_gap_bptt = abs(mean_nll["snn_ra0"] - mean_nll["snn_bptt"])
    nll_gap_at1 = abs(mean_nll["snn_ra0"] - mean_nll["snn_at1"])
    edit_gap_bptt = abs(mean_edit["snn_ra0"] - mean_edit["snn_bptt"])
    edit_gap_at1 = abs(mean_edit["snn_ra0"] - mean_edit["snn_at1"])
    paired_sensitivity = _mean(
        seed["generation"]["snn_ra0"]["paired_action_sensitivity"]
        for seed in seed_results
    )
    quality_pass = (
        ra0_improvement
        and mean_nll["snn_ra0"] <= best_ann_nll + 0.25
        and nll_gap_bptt <= 0.10
        and nll_gap_at1 <= 0.10
        and mean_edit["snn_ra0"] >= best_ann_edit - 0.10
        and edit_gap_bptt <= 0.05
        and edit_gap_at1 <= 0.05
        and mean_edit["snn_ra0"] >= baseline_edit + 0.05
        and paired_sensitivity >= 0.50
    )
    mean_training_p50 = {
        name: _mean(
            seed["training"][name]["timing"]["p50_ms"]
            for seed in seed_results
        )
        for name in MODEL_NAMES
    }
    at1_speedup = mean_training_p50["snn_at1"] / mean_training_p50["snn_ra0"]
    bptt_speedup = mean_training_p50["snn_bptt"] / mean_training_p50["snn_ra0"]
    speed_pass = (
        at1_speedup >= 1.25
        and bptt_speedup >= 1.25
        and mean_training_p50["snn_ra0"] <= mean_training_p50["lstm"]
    )
    stream_pass = all(
        seed["generation"]["snn_ra0"]["timing"]["generated_token"]["p50_ms"]
        <= seed["generation"]["lstm"]["timing"]["generated_token"]["p50_ms"]
        and seed["generation"]["snn_ra0"]["timing"]["generated_token"]["p95_ms"]
        <= seed["generation"]["lstm"]["timing"]["generated_token"]["p95_ms"]
        and seed["generation"]["snn_ra0"]["timing"]["prefill"]["p50_ms"]
        <= seed["generation"]["lstm"]["timing"]["prefill"]["p50_ms"]
        for seed in seed_results
    )
    gates = {
        "data_gate": bool(data_audit["passed"]),
        "task_gate": task_improvement and task_generation,
        "quality_gate": quality_pass,
        "speed_gate": speed_pass,
        "stream_gate": stream_pass,
    }
    overall = "PASS" if all(gates.values()) else "FAIL"
    return {
        **{name: "PASS" if passed else "FAIL" for name, passed in gates.items()},
        "overall": overall,
        "mean_test_teacher_nll": mean_nll,
        "mean_test_greedy_edit_similarity": mean_edit,
        "best_ann_nll": best_ann_nll,
        "best_ann_edit_similarity": best_ann_edit,
        "best_non_neural_edit_similarity": baseline_edit,
        "ra0_vs_bptt_nll_gap": nll_gap_bptt,
        "ra0_vs_at1_nll_gap": nll_gap_at1,
        "ra0_vs_bptt_edit_gap": edit_gap_bptt,
        "ra0_vs_at1_edit_gap": edit_gap_at1,
        "ra0_paired_action_sensitivity": paired_sensitivity,
        "mean_training_p50_ms": mean_training_p50,
        "ra0_vs_at1_training_speedup": at1_speedup,
        "ra0_vs_bptt_training_speedup": bptt_speedup,
        "next_route": (
            "online_closed_loop"
            if overall == "PASS"
            else "byte_or_adaptive_spike_or_native_scan"
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
    manifest = tw0._manifest_provenance(corpus_root)
    corpus = load_event_corpus(corpus_root)
    examples, vocabulary = build_counterfactual_examples(corpus_root, corpus)
    data_audit = audit_examples(examples, vocabulary)
    if not data_audit["passed"]:
        raise AssertionError("SG0 data audit failed; refusing model experiment")

    seed_results = []
    for seed in args.seeds:
        models = build_textworld_models(
            9_400_000 + 100 * seed, vocabulary, device=device
        )
        parameter_counts = {
            name: {
                "total": count_parameters(model),
                "core": count_parameters(model.core),
            }
            for name, model in models.items()
        }
        totals = tuple(record["total"] for record in parameter_counts.values())
        parameter_spread = (max(totals) - min(totals)) / _mean(totals)
        if parameter_spread > 0.02:
            raise AssertionError(f"SG0 parameter spread failed: {parameter_counts}")
        pre_teacher = {
            name: {
                split: evaluate_teacher(model, examples[split], device=device)
                for split in ("valid", "test")
            }
            for name, model in models.items()
        }
        schedule = _training_schedule(
            len(examples["train"]), args.epochs, 9_401_000 + seed
        )
        training = {
            name: train_model(
                name,
                model,
                examples["train"],
                schedule,
                epochs=args.epochs,
                device=device,
            )
            for name, model in models.items()
        }
        post_teacher = {
            name: {
                split: evaluate_teacher(model, examples[split], device=device)
                for split in ("train", "valid", "test")
            }
            for name, model in models.items()
        }
        generation = {
            name: generate_model(
                model,
                examples["test"],
                vocabulary,
                max_tokens=args.max_generation_tokens,
                device=device,
                include_records=True,
            )
            for name, model in models.items()
        }
        seed_results.append(
            {
                "seed": seed,
                "parameter_counts": parameter_counts,
                "parameter_relative_spread": parameter_spread,
                "pre_teacher": pre_teacher,
                "training": training,
                "post_teacher": post_teacher,
                "generation": generation,
            }
        )
    decision = _decision(
        data_audit=data_audit, seed_results=seed_results, quick=args.quick
    )
    return {
        "schema_version": 1,
        "experiment": "E3-SG0 TextWorld counterfactual sequence generation",
        "formal": not args.quick,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "environment": _environment(device),
        "configuration": {
            "corpus_dir": str(corpus_root),
            "seeds": tuple(args.seeds),
            "epochs": args.epochs,
            "threads": args.threads if device.type == "cpu" else None,
            "d_model": tw0.D_MODEL,
            "state_dim": tw0.STATE_DIM,
            "learning_rate": 1e-3,
            "weight_decay": 0.01,
            "gradient_clip": 1.0,
            "optimizer_fused": True,
            "gradient_foreach": True,
            "max_generation_tokens": args.max_generation_tokens,
            "state_reset_per_example": True,
            "target_truncation": False,
            "vocabulary_source": "normalized_train_prompt_and_target_only",
        },
        "dataset": {
            "synthetic": False,
            "manifest_provenance": manifest,
            "corpus_provenance": tw0._corpus_provenance(corpus),
            "generation_vocabulary": {
                "size": len(vocabulary),
                "fingerprint": vocabulary.fingerprint,
                "tokenizer": corpus.tokenizer.metadata(),
                "source_split": "train",
                "source_fields": ("normalized_prompt", "normalized_target"),
            },
            "audit": data_audit,
        },
        "seeds": seed_results,
        "decision": decision,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-dir", type=Path, default=tw0.DEFAULT_CORPUS_DIR)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/e3_scan/e3_sg0_counterfactual_generation.json"),
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=(0, 1, 2))
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument(
        "--max-generation-tokens", type=int, default=MAX_GENERATION_TOKENS
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    if min(args.epochs, args.threads, args.max_generation_tokens) <= 0:
        parser.error("epochs, threads, and max generation tokens must be positive")
    if args.quick:
        args.seeds = args.seeds[:1]
        args.epochs = 2
        args.max_generation_tokens = min(args.max_generation_tokens, 32)
    return args


def main() -> None:
    args = _parse_args()
    result = run_experiment(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(result["decision"], indent=2, sort_keys=True))
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
