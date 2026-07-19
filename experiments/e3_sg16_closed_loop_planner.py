"""SG16 real TextWorld closed-loop candidate planning benchmark.

The strict phase-isolated spike kernel, LSTM, and Transformer see the same
typed action-event state and predict the same four next-state channels.  A
fixed semantic planner executes the selected command in the official
TextWorld interpreter and reconciles model state only after the real
observation arrives.  No walkthrough or counterfactual clone is exposed to
the planner.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple

import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.e2_f0_fusion_benchmark import (  # noqa: E402
    _environment,
    _percentile,
    _sync,
)
from experiments import e3_sg0_counterfactual_generation as sg0  # noqa: E402
from experiments import e3_sg10_multichannel_delta as sg10  # noqa: E402
from experiments import e3_tw0_sparse_event_lm as tw0  # noqa: E402
from experiments.e3_sg9_atomic_event_stream import (  # noqa: E402
    _generic_candidate_hidden,
    _prefill_previous_event,
    action_event_token,
)
from experiments.e3_sg12_spike_delay_rls import (  # noqa: E402
    PRIMARY_ORDER,
    action_index,
    build_action_alphabet,
)
from experiments.e3_sg13_suffix_spike_kernel import (  # noqa: E402
    _padded_history_key,
    block_schur_kernel_fit,
    extract_kernel_records,
    suffix_spike_kernel,
)
from experiments.e3_sg15_phase_isolated_kernel import (  # noqa: E402
    PRIMARY_SPEC,
)
from vpsc.world_model.cores import count_parameters, state_nbytes  # noqa: E402
from vpsc.world_model.event_corpus import load_event_corpus  # noqa: E402
from vpsc.world_model.textworld import open_textworld  # noqa: E402
from vpsc.world_model.wikitext import (  # noqa: E402
    SPLITS,
    Vocabulary,
    file_sha256,
)


DEFAULT_CORPUS = Path("results/e3_scan/textworld_sg15r_l5")
DEFAULT_OUTPUT = Path("results/e3_scan/e3_sg16_closed_loop_planner.json")
DEFAULT_SG15R_REFERENCE = Path(
    "results/e3_scan/e3_sg15r_fourth_fresh_confirmation.json"
)
DEFAULT_SG16_REFERENCE = Path(
    "results/e3_scan/e3_sg16_closed_loop_planner.json"
)
SG15R_REFERENCE_SHA256 = (
    "A0599E48C13E3FFC1171DD5FCF08B175F4110FE884CD33EF0ED6B916A1698ACE"
)
SG15R_EXPERIMENT = "E3-SG15R fourth-fresh strict phase suffix confirmation"
SG16_EXPERIMENT = "E3-SG16 real TextWorld closed-loop candidate planner"
FROZEN_LAMBDA = 1e-6
ANN_MODEL_NAMES = ("lstm", "transformer")
EXPECTED_COUNTS = {"train": 480, "valid": 120, "test": 120}
EXPECTED_GROUPS = {"train": 160, "valid": 40, "test": 40}
MECHANISM_SEEDS = {
    "train": tuple(range(20260801, 20260833)),
    "valid": tuple(range(20261101, 20261109)),
    "test": tuple(range(20261109, 20261117)),
}
CONFIRMATION_SEEDS = {
    "train": tuple(range(20260801, 20260833)),
    "valid": tuple(range(20261201, 20261209)),
    "test": tuple(range(20261209, 20261217)),
}
ROOM_PRIORITY = {
    sg10.ROOM_LABELS[0]: 0,
    sg10.ROOM_LABELS[2]: 1,
    sg10.ROOM_LABELS[3]: 2,
    sg10.ROOM_LABELS[1]: 3,
}


@dataclass(frozen=True)
class CandidateBranch:
    action: str
    scores: torch.Tensor
    elapsed_ms: float
    next_hidden: Optional[torch.Tensor] = None
    next_state: Any = None


@dataclass(frozen=True)
class PredictionSummary:
    labels: Tuple[str, str, str, str]
    semantic_priority: Tuple[int, int, int]
    confidence_margin: float


@dataclass(frozen=True)
class AnnPrefix:
    previous_hidden: torch.Tensor
    state: Any


def _load_reference(
    path: Path, expected_sha: str, expected_experiment: str
) -> Tuple[Dict[str, Any], str]:
    payload = path.read_bytes()
    digest = hashlib.sha256(payload).hexdigest().upper()
    if digest != expected_sha.upper():
        raise ValueError(
            f"reference SHA mismatch for {path}: expected {expected_sha}, got {digest}"
        )
    result = json.loads(payload)
    if result.get("experiment") != expected_experiment:
        raise ValueError(
            f"unexpected reference experiment {result.get('experiment')!r}"
        )
    return result, digest


def _artifact_hashes(corpus_root: Path, split: str) -> Dict[str, str]:
    return {
        name: file_sha256(corpus_root / split / name).upper()
        for name in ("manifest.json", "episodes.jsonl", "token_events.txt")
    }


def decode_prediction(scores: torch.Tensor) -> PredictionSummary:
    values = scores.detach().reshape(-1)
    if values.numel() != sg10.TOTAL_LOGITS:
        raise ValueError(
            f"candidate scores must have {sg10.TOTAL_LOGITS} logits"
        )
    labels = []
    confidence = 0.0
    for name, channel_labels in sg10.CHANNEL_SPECS:
        start, stop = sg10.CHANNEL_OFFSETS[name]
        channel = values[start:stop]
        index = int(channel.argmax().item())
        labels.append(channel_labels[index])
        top = torch.topk(channel, k=min(2, channel.numel())).values
        if top.numel() == 2:
            confidence += float((top[0] - top[1]).item())
    label_tuple = tuple(labels)
    reward_positive = int(label_tuple[1] == sg10.REWARD_LABELS[1])
    done = int(label_tuple[2] == sg10.DONE_LABELS[1])
    return PredictionSummary(
        labels=label_tuple,  # type: ignore[arg-type]
        semantic_priority=(
            reward_positive,
            done,
            ROOM_PRIORITY[label_tuple[0]],
        ),
        confidence_margin=confidence,
    )


def select_candidate(
    branches: Sequence[CandidateBranch],
) -> Tuple[CandidateBranch, Dict[str, PredictionSummary]]:
    if not branches:
        raise ValueError("closed-loop planner received no candidate actions")
    summaries = {branch.action: decode_prediction(branch.scores) for branch in branches}
    best_semantic = max(
        summary.semantic_priority for summary in summaries.values()
    )
    semantic_ties = tuple(
        branch
        for branch in branches
        if summaries[branch.action].semantic_priority == best_semantic
    )
    best_confidence = max(
        summaries[branch.action].confidence_margin for branch in semantic_ties
    )
    confidence_ties = tuple(
        branch
        for branch in semantic_ties
        if summaries[branch.action].confidence_margin == best_confidence
    )
    selected = min(confidence_ties, key=lambda branch: branch.action)
    return selected, summaries


def reconcile_topology(
    room_stack: Sequence[str], next_room: Optional[str]
) -> Tuple[str, Tuple[str, ...]]:
    if not room_stack:
        raise ValueError("room stack cannot be empty")
    rooms = tuple(room_stack)
    if next_room is None:
        return sg10.ROOM_LABELS[0], rooms
    if next_room == rooms[-1]:
        return sg10.ROOM_LABELS[3], rooms
    if next_room in rooms:
        target = max(index for index, room in enumerate(rooms) if room == next_room)
        return sg10.ROOM_LABELS[2], rooms[: target + 1]
    return sg10.ROOM_LABELS[1], rooms + (next_room,)


class KernelPlannerBackend:
    name = "strict_phase_snn"

    def __init__(
        self,
        *,
        alphabet_index: Mapping[str, int],
        prototype_keys: torch.Tensor,
        prototype_phases: torch.Tensor,
        alpha: torch.Tensor,
        device: torch.device,
    ) -> None:
        self.alphabet_index = alphabet_index
        self.prototype_keys = prototype_keys.to(device=device)
        self.prototype_phases = prototype_phases.to(device=device)
        self.alpha = alpha.to(device=device, dtype=torch.float32)
        self.device = device
        self.pad_index = len(alphabet_index)

    def reset(self) -> float:
        return 0.0

    def score(
        self, context_actions: Sequence[str], actions: Sequence[str]
    ) -> Tuple[CandidateBranch, ...]:
        branches = []
        with torch.inference_mode():
            for action in actions:
                _sync(self.device)
                started = time.perf_counter_ns()
                key = torch.tensor(
                    [
                        _padded_history_key(
                            context_actions,
                            action,
                            alphabet_index=self.alphabet_index,
                            pad_index=self.pad_index,
                        )
                    ],
                    dtype=torch.long,
                    device=self.device,
                )
                phase = torch.tensor(
                    (len(context_actions),), dtype=torch.long, device=self.device
                )
                kernel = suffix_spike_kernel(
                    key,
                    self.prototype_keys,
                    phase,
                    self.prototype_phases,
                    PRIMARY_SPEC,
                    dtype=torch.float32,
                )
                scores = kernel @ self.alpha
                scores.sum().item()
                _sync(self.device)
                branches.append(
                    CandidateBranch(
                        action=action,
                        scores=scores,
                        elapsed_ms=(time.perf_counter_ns() - started) / 1e6,
                    )
                )
        return tuple(branches)

    def on_novel(self, selected: CandidateBranch) -> None:
        return None

    def on_rollback(self, depth: int) -> None:
        return None

    def state_bytes(self) -> int:
        return PRIMARY_ORDER * len(self.alphabet_index) + 8


class AnnPlannerBackend:
    def __init__(
        self,
        name: str,
        model: sg10.MultiChannelBilinearModel,
        vocabulary: Vocabulary,
        *,
        device: torch.device,
    ) -> None:
        if name not in ANN_MODEL_NAMES:
            raise ValueError(f"unsupported ANN planner backend {name!r}")
        self.name = name
        self.model = model
        self.vocabulary = vocabulary
        self.device = device
        self.prefix_stack: list[AnnPrefix] = []

    def reset(self) -> float:
        self.model.eval()
        token_id = self.vocabulary.token_id(sg10.START_EVENT)
        if token_id == self.vocabulary.unk_id:
            raise KeyError("START_EVENT is outside ANN vocabulary")
        with torch.inference_mode():
            _sync(self.device)
            started = time.perf_counter_ns()
            hidden, state = _prefill_previous_event(
                self.model.language_model, token_id, device=self.device
            )
            hidden.sum().item()
            _sync(self.device)
        self.prefix_stack = [AnnPrefix(hidden, state)]
        return (time.perf_counter_ns() - started) / 1e6

    def score(
        self, context_actions: Sequence[str], actions: Sequence[str]
    ) -> Tuple[CandidateBranch, ...]:
        if len(self.prefix_stack) != len(context_actions) + 1:
            raise AssertionError("ANN cached stack and topological depth diverged")
        prefix = self.prefix_stack[-1]
        branches = []
        with torch.inference_mode():
            for action in actions:
                token = action_event_token(action)
                token_id = self.vocabulary.token_id(token)
                if token_id == self.vocabulary.unk_id:
                    raise KeyError(f"ANN action event is outside train vocabulary: {token}")
                _sync(self.device)
                started = time.perf_counter_ns()
                hidden, state = _generic_candidate_hidden(
                    self.model.language_model,
                    token_id,
                    prefix.state,
                    device=self.device,
                )
                scores = self.model.relation_head(prefix.previous_hidden, hidden)
                scores.sum().item()
                _sync(self.device)
                branches.append(
                    CandidateBranch(
                        action=action,
                        scores=scores,
                        elapsed_ms=(time.perf_counter_ns() - started) / 1e6,
                        next_hidden=hidden,
                        next_state=state,
                    )
                )
        return tuple(branches)

    def on_novel(self, selected: CandidateBranch) -> None:
        if selected.next_hidden is None or selected.next_state is None:
            raise AssertionError("ANN novel transition lacks cached candidate state")
        self.prefix_stack.append(
            AnnPrefix(selected.next_hidden, selected.next_state)
        )

    def on_rollback(self, depth: int) -> None:
        if depth < 0 or depth >= len(self.prefix_stack):
            raise ValueError(f"invalid ANN rollback depth {depth}")
        del self.prefix_stack[depth + 1 :]

    def state_bytes(self) -> int:
        return sum(
            state_nbytes(prefix.state)
            + prefix.previous_hidden.numel() * prefix.previous_hidden.element_size()
            for prefix in self.prefix_stack
        )


def _prediction_record(summary: PredictionSummary) -> Dict[str, Any]:
    return {
        "labels": {
            name: label
            for (name, _channel_labels), label in zip(
                sg10.CHANNEL_SPECS, summary.labels
            )
        },
        "semantic_priority": summary.semantic_priority,
        "confidence_margin": summary.confidence_margin,
    }


def run_closed_loop_game(
    backend: Any,
    game: Mapping[str, Any],
    corpus: Any,
    *,
    max_actions: int,
) -> Dict[str, Any]:
    game_file = Path(str(game["game_file"]))
    expected_sha = str(game["game_sha256"]).lower()
    actual_sha = file_sha256(game_file)
    if actual_sha != expected_sha:
        raise ValueError(
            f"closed-loop game SHA mismatch for {game_file}: "
            f"expected {expected_sha}, got {actual_sha}"
        )
    adapter = open_textworld(game_file, extras=())
    records = []
    candidate_timings = []
    decision_timings = []
    environment_timings = []
    state_bytes = []
    total_reward = 0.0
    done = False
    won = False
    try:
        initial_observation = adapter.reset()
        initial_room = sg10._room_feature(corpus, initial_observation)
        if initial_room is None:
            raise ValueError("closed-loop initial observation has no room feature")
        room_stack: Tuple[str, ...] = (initial_room,)
        context_actions: Tuple[str, ...] = ()
        initialization_ms = float(backend.reset())
        state_bytes.append(int(backend.state_bytes()))
        for decision_index in range(max_actions):
            phase_before = len(context_actions)
            actions = tuple(sorted(set(adapter.admissible_actions)))
            if not actions:
                raise RuntimeError("official TextWorld state has no admissible actions")
            decision_started = time.perf_counter_ns()
            branches = backend.score(context_actions, actions)
            selected, summaries = select_candidate(branches)
            decision_elapsed = (time.perf_counter_ns() - decision_started) / 1e6
            decision_timings.append(decision_elapsed)
            candidate_timings.extend(branch.elapsed_ms for branch in branches)
            environment_started = time.perf_counter_ns()
            transition = adapter.step(selected.action)
            environment_timings.append(
                (time.perf_counter_ns() - environment_started) / 1e6
            )
            total_reward += float(transition.reward)
            next_room = sg10._room_feature(corpus, transition.next_observation)
            actual_relation, reconciled_rooms = reconcile_topology(
                room_stack, next_room
            )
            if actual_relation == sg10.ROOM_LABELS[1]:
                context_actions = context_actions + (selected.action,)
                backend.on_novel(selected)
            elif actual_relation == sg10.ROOM_LABELS[2]:
                target_depth = len(reconciled_rooms) - 1
                context_actions = context_actions[:target_depth]
                backend.on_rollback(target_depth)
            room_stack = reconciled_rooms
            if len(context_actions) != len(room_stack) - 1:
                raise AssertionError("topological actions and room stack diverged")
            state_bytes.append(int(backend.state_bytes()))
            info = dict(transition.info)
            done = bool(transition.done)
            won = bool(info.get("won", False))
            records.append(
                {
                    "decision": decision_index,
                    "phase_before": phase_before,
                    "admissible_actions": actions,
                    "selected_action": selected.action,
                    "selected_prediction": _prediction_record(
                        summaries[selected.action]
                    ),
                    "candidate_predictions": {
                        branch.action: _prediction_record(summaries[branch.action])
                        for branch in branches
                    },
                    "actual": {
                        "room_relation": actual_relation,
                        "reward": float(transition.reward),
                        "done": done,
                        "won": won,
                    },
                    "path_depth_after": len(context_actions),
                    "candidate_elapsed_ms": {
                        branch.action: branch.elapsed_ms for branch in branches
                    },
                    "decision_elapsed_ms": decision_elapsed,
                    "environment_elapsed_ms": environment_timings[-1],
                }
            )
            if done:
                break
    finally:
        adapter.close()
    return {
        "seed": int(game["seed"]),
        "game_file": str(game_file),
        "game_sha256": actual_sha.upper(),
        "official_core_api": True,
        "walkthrough_requested_or_read": False,
        "counterfactual_clone_used": False,
        "won": won,
        "done": done,
        "action_count": len(records),
        "within_action_budget": done and len(records) <= max_actions,
        "optimal_five_action_win": won and len(records) == 5,
        "total_reward": total_reward,
        "initialization_ms": initialization_ms,
        "candidate_timings_ms": candidate_timings,
        "decision_timings_ms": decision_timings,
        "environment_timings_ms": environment_timings,
        "state_bytes_max": max(state_bytes),
        "trajectory": records,
    }


def _timing_summary(values: Sequence[float]) -> Dict[str, float]:
    if not values:
        raise ValueError("timing summary requires samples")
    return {
        "sample_count": len(values),
        "mean_ms": sg0._mean(values),
        "min_ms": min(values),
        "max_ms": max(values),
        "p50_ms": _percentile(values, 0.50),
        "p95_ms": _percentile(values, 0.95),
        "p99_ms": _percentile(values, 0.99),
    }


def run_closed_loop_suite(
    backend: Any,
    games: Sequence[Mapping[str, Any]],
    corpus: Any,
    *,
    max_actions: int,
) -> Dict[str, Any]:
    results = tuple(
        run_closed_loop_game(
            backend, game, corpus, max_actions=max_actions
        )
        for game in games
    )
    candidate_timings = tuple(
        value for game in results for value in game["candidate_timings_ms"]
    )
    decision_timings = tuple(
        value for game in results for value in game["decision_timings_ms"]
    )
    environment_timings = tuple(
        value for game in results for value in game["environment_timings_ms"]
    )
    return {
        "backend": backend.name,
        "game_count": len(results),
        "win_count": sum(bool(game["won"]) for game in results),
        "win_rate": sum(bool(game["won"]) for game in results) / len(results),
        "mean_action_count": sg0._mean(game["action_count"] for game in results),
        "optimal_five_action_win_rate": sum(
            bool(game["optimal_five_action_win"]) for game in results
        )
        / len(results),
        "mean_total_reward": sg0._mean(game["total_reward"] for game in results),
        "all_within_action_budget": all(
            bool(game["within_action_budget"]) for game in results
        ),
        "candidate_timing": _timing_summary(candidate_timings),
        "decision_timing": _timing_summary(decision_timings),
        "environment_timing": _timing_summary(environment_timings),
        "initialization_ms": _timing_summary(
            tuple(game["initialization_ms"] for game in results)
        ),
        "state_bytes_max": max(game["state_bytes_max"] for game in results),
        "games": results,
    }


def _game_records(
    corpus_root: Path,
    expected_test_seeds: Sequence[int],
    *,
    split: str = "test",
) -> Tuple[Dict[str, Any], ...]:
    if split not in SPLITS:
        raise ValueError(f"unknown game-record split {split!r}")
    manifest = json.loads(
        (corpus_root / split / "manifest.json").read_text(encoding="utf-8")
    )
    episodes = {
        int(value["seed"]): value
        for value in (
            json.loads(line)
            for line in (corpus_root / split / "episodes.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
        )
    }
    games = []
    for record in manifest["games"]:
        seed = int(record["seed"])
        episode = episodes[seed]
        games.append(
            {
                "seed": seed,
                "game_file": str(record["game_file"]),
                "game_sha256": str(episode["game_sha256"]),
            }
        )
    if tuple(game["seed"] for game in games) != tuple(expected_test_seeds):
        raise ValueError(
            f"closed-loop game seeds do not match frozen {split} seeds"
        )
    return tuple(games)


def _action_coverage(
    corpus_root: Path,
    vocabulary: Vocabulary,
    alphabet_index: Mapping[str, int],
) -> Dict[str, Any]:
    actions = set()
    for line in (corpus_root / "test" / "episodes.jsonl").read_text(
        encoding="utf-8"
    ).splitlines():
        episode = json.loads(line)
        for step in episode["steps"]:
            actions.update(str(action) for action in step["admissible_actions"])
    missing_alphabet = []
    missing_vocabulary = []
    for action in sorted(actions):
        try:
            action_index(action, alphabet_index)
        except KeyError:
            missing_alphabet.append(action)
        token = action_event_token(action)
        if vocabulary.token_id(token) == vocabulary.unk_id:
            missing_vocabulary.append(action)
    return {
        "test_admissible_actions": tuple(sorted(actions)),
        "missing_from_snn_alphabet": tuple(missing_alphabet),
        "missing_from_ann_vocabulary": tuple(missing_vocabulary),
        "passed": not missing_alphabet and not missing_vocabulary,
    }


def _fit_spike_kernel(
    records: Mapping[str, Mapping[str, Any]],
    examples: Mapping[str, Sequence[sg10.MultiChannelExample]],
    *,
    batch_groups: int,
    device: torch.device,
) -> Tuple[Dict[str, Any], Dict[str, torch.Tensor]]:
    schedule = sg10.build_length_stratified_schedule(
        examples["train"],
        epochs=1,
        batch_groups=batch_groups,
        seed=10_216_000,
    )
    alpha, online = block_schur_kernel_fit(
        records,
        PRIMARY_SPEC,
        schedule,
        ridge_lambda=FROZEN_LAMBDA,
        device=device,
    )
    train = records["train"]
    audit_started = time.perf_counter_ns()
    train_kernel = suffix_spike_kernel(
        train["keys"],
        train["keys"],
        train["phases"],
        train["phases"],
        PRIMARY_SPEC,
    )
    identity = torch.eye(
        train_kernel.shape[0], dtype=torch.float64, device=device
    )
    batch_alpha = torch.linalg.solve(
        train_kernel + FROZEN_LAMBDA * identity,
        train["target_code"],
    )
    batch_score_difference = float(
        ((train_kernel @ alpha) - (train_kernel @ batch_alpha)).abs().max().item()
    )
    audit_seconds = (time.perf_counter_ns() - audit_started) / 1e9
    if batch_score_difference > 1e-6:
        raise AssertionError("SG16 online spike kernel diverged from batch solve")
    metrics = {}
    for split in SPLITS:
        kernel = suffix_spike_kernel(
            records[split]["keys"],
            train["keys"],
            records[split]["phases"],
            train["phases"],
            PRIMARY_SPEC,
        )
        scores = kernel @ alpha
        metrics[split] = sg10._ridge_multichannel_metrics(
            scores,
            records[split]["targets"],
            records[split]["group_ids"],
        )
    logical_model_bytes = (
        train["keys"].numel()
        + train["phases"].numel()
        + alpha.numel() * 4
    )
    result = {
        "model": "strict_phase_snn",
        "kernel": PRIMARY_SPEC.name,
        "ridge_lambda": FROZEN_LAMBDA,
        "training": {
            **online,
            "train_state_encoding_seconds": train["elapsed_seconds"],
            "deployment_training_wall_seconds": (
                train["elapsed_seconds"] + online["elapsed_seconds"]
            ),
            "batch_equivalence_audit_seconds_excluded": audit_seconds,
            "batch_train_score_max_abs_difference": batch_score_difference,
        },
        "offline": metrics,
        "logical_model_storage_bytes_uint8_keys_phase_float32_alpha": logical_model_bytes,
        "persistent_state_bytes": PRIMARY_ORDER
        * int(train["keys"].max().item())
        + 8,
    }
    runtime = {
        "alpha": alpha.detach(),
        "prototype_keys": train["keys"].detach(),
        "prototype_phases": train["phases"].detach(),
    }
    return result, runtime


def _frozen_protocol(args: argparse.Namespace) -> Dict[str, Any]:
    return {
        "primary_kernel": PRIMARY_SPEC.name,
        "ridge_lambda": FROZEN_LAMBDA,
        "spike_delay_order": PRIMARY_ORDER,
        "planner_semantic_order": (
            "reward_positive",
            "done",
            "room_novel",
            "room_same",
            "room_previous",
            "room_no_observation",
        ),
        "planner_tie_break": "sum channel top1-top2 margins, then lexical action",
        "state_reconciliation": (
            "real observation room stack: novel push, prior rollback, same/no-room hold"
        ),
        "ann_models": ANN_MODEL_NAMES,
        "ann_epochs": args.epochs,
        "ann_training_seeds": tuple(args.seeds),
        "d_model": args.d_model,
        "state_dim": args.state_dim,
        "num_heads": args.num_heads,
        "batch_groups": args.batch_groups,
        "max_actions": args.max_actions,
        "snn_online_schedule_seed": 10_216_000,
    }


def _mean_suite_metric(
    replications: Sequence[Mapping[str, Any]], model: str, key: str
) -> float:
    return sg0._mean(replication["closed_loop"][model][key] for replication in replications)


def _decision(
    data_audit: Mapping[str, Any],
    action_coverage: Mapping[str, Any],
    spike: Mapping[str, Any],
    replications: Sequence[Mapping[str, Any]],
    *,
    fresh_confirmation: bool,
    quick: bool,
) -> Dict[str, Any]:
    if quick:
        return {
            "data_gate": "PASS" if data_audit["passed"] else "FAIL",
            "action_coverage_gate": "PASS" if action_coverage["passed"] else "FAIL",
            "offline_model_gate": "SMOKE",
            "closed_loop_task_gate": "SMOKE",
            "snn_closed_loop_quality_gate": "SMOKE",
            "quality_pareto_gate": "SMOKE",
            "training_speed_gate": "SMOKE",
            "response_speed_gate": "SMOKE",
            "storage_gate": "SMOKE",
            "overall": "SMOKE",
            "next_route": "formal_sg16_closed_loop_candidate_planner",
        }
    offline_model = (
        spike["offline"]["test"]["exact_vector_accuracy"] >= 0.98
        and all(
            replication["offline"]["transformer"]["exact_vector_accuracy"]
            >= 0.90
            for replication in replications
        )
        and all(
            replication["offline"]["lstm"]["exact_vector_accuracy"] >= 0.85
            for replication in replications
        )
    )
    closed_loop_task = all(
        max(
            replication["closed_loop"]["lstm"]["win_rate"],
            replication["closed_loop"]["transformer"]["win_rate"],
        )
        >= 0.75
        for replication in replications
    )
    snn_quality = all(
        replication["closed_loop"]["strict_phase_snn"]["win_rate"] == 1.0
        and replication["closed_loop"]["strict_phase_snn"]["mean_action_count"]
        <= 5.0
        and replication["closed_loop"]["strict_phase_snn"][
            "all_within_action_budget"
        ]
        for replication in replications
    )
    quality_pareto = all(
        replication["closed_loop"]["strict_phase_snn"]["win_rate"]
        >= max(
            replication["closed_loop"]["lstm"]["win_rate"],
            replication["closed_loop"]["transformer"]["win_rate"],
        )
        and replication["closed_loop"]["strict_phase_snn"]["mean_action_count"]
        <= min(
            replication["closed_loop"]["lstm"]["mean_action_count"],
            replication["closed_loop"]["transformer"]["mean_action_count"],
        )
        for replication in replications
    )
    spike_training = spike["training"]["deployment_training_wall_seconds"]
    training_speed = all(
        spike_training < replication["training"][name]["elapsed_seconds"]
        for replication in replications
        for name in ANN_MODEL_NAMES
    )
    response_comparisons = []
    for replication in replications:
        snn = replication["closed_loop"]["strict_phase_snn"]
        for name in ANN_MODEL_NAMES:
            ann = replication["closed_loop"][name]
            record = {
                "seed": replication["seed"],
                "ann_model": name,
                "snn_candidate_p50_ms": snn["candidate_timing"]["p50_ms"],
                "ann_candidate_p50_ms": ann["candidate_timing"]["p50_ms"],
                "snn_candidate_p95_ms": snn["candidate_timing"]["p95_ms"],
                "ann_candidate_p95_ms": ann["candidate_timing"]["p95_ms"],
                "snn_decision_p50_ms": snn["decision_timing"]["p50_ms"],
                "ann_decision_p50_ms": ann["decision_timing"]["p50_ms"],
                "snn_decision_p95_ms": snn["decision_timing"]["p95_ms"],
                "ann_decision_p95_ms": ann["decision_timing"]["p95_ms"],
            }
            record["passed"] = (
                record["snn_candidate_p50_ms"] <= record["ann_candidate_p50_ms"]
                and record["snn_candidate_p95_ms"] <= record["ann_candidate_p95_ms"]
                and record["snn_decision_p50_ms"] <= record["ann_decision_p50_ms"]
                and record["snn_decision_p95_ms"] <= record["ann_decision_p95_ms"]
            )
            response_comparisons.append(record)
    response_speed = all(record["passed"] for record in response_comparisons)
    spike_bytes = spike[
        "logical_model_storage_bytes_uint8_keys_phase_float32_alpha"
    ]
    storage = all(
        spike_bytes <= replication["parameter_counts"][name] * 4
        for replication in replications
        for name in ANN_MODEL_NAMES
    )
    gates = {
        "data_gate": bool(data_audit["passed"]),
        "action_coverage_gate": bool(action_coverage["passed"]),
        "offline_model_gate": offline_model,
        "closed_loop_task_gate": closed_loop_task,
        "snn_closed_loop_quality_gate": snn_quality,
        "quality_pareto_gate": quality_pareto,
        "training_speed_gate": training_speed,
        "response_speed_gate": response_speed,
        "storage_gate": storage,
    }
    overall = "PASS" if all(gates.values()) else "FAIL"
    if overall == "PASS":
        next_route = (
            "sg17_two_step_counterfactual_rollout"
            if fresh_confirmation
            else "sg16r_fifth_fresh_closed_loop_confirmation"
        )
    elif offline_model and not snn_quality:
        next_route = "sg16_planner_horizon_or_observation_state"
    elif not response_speed:
        next_route = "sg16_vectorized_unique_prototype_kernel"
    else:
        next_route = "sg16_observation_reservoir_times_strict_phase_kernel"
    return {
        **{name: "PASS" if value else "FAIL" for name, value in gates.items()},
        "overall": overall,
        "fresh_confirmation": fresh_confirmation,
        "independent_confirmation_required": not fresh_confirmation,
        "mean_closed_loop": {
            model: {
                "win_rate": _mean_suite_metric(replications, model, "win_rate"),
                "mean_action_count": _mean_suite_metric(
                    replications, model, "mean_action_count"
                ),
                "optimal_five_action_win_rate": _mean_suite_metric(
                    replications, model, "optimal_five_action_win_rate"
                ),
            }
            for model in ("strict_phase_snn", *ANN_MODEL_NAMES)
        },
        "snn_training_wall_seconds": spike_training,
        "response_comparisons": response_comparisons,
        "next_route": next_route,
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
    expected_seeds = CONFIRMATION_SEEDS if args.fresh_confirmation else MECHANISM_SEEDS
    sg15_reference, sg15_digest = _load_reference(
        args.sg15r_reference.expanduser().resolve(),
        SG15R_REFERENCE_SHA256,
        SG15R_EXPERIMENT,
    )
    if (
        sg15_reference["decision"]["overall"] != "PASS"
        or sg15_reference["configuration"]["primary_kernel"] != PRIMARY_SPEC.name
        or sg15_reference["kernel_results"][PRIMARY_SPEC.name]["selected_lambda"]
        != FROZEN_LAMBDA
    ):
        raise ValueError("SG15R reference did not freeze the expected passing kernel")
    sg16_reference = None
    sg16_digest = None
    if args.fresh_confirmation:
        if not args.sg16_reference_sha:
            raise ValueError("fresh confirmation requires --sg16-reference-sha")
        sg16_reference, sg16_digest = _load_reference(
            args.sg16_reference.expanduser().resolve(),
            args.sg16_reference_sha,
            SG16_EXPERIMENT,
        )
        if sg16_reference["decision"]["overall"] != "PASS":
            raise ValueError("SG16 mechanism reference must pass before confirmation")
    manifest = tw0._manifest_provenance(
        corpus_root, expected_seeds_by_split=expected_seeds
    )
    corpus = load_event_corpus(corpus_root)
    examples, vocabulary = sg10.build_multichannel_examples(corpus_root, corpus)
    data_audit = sg10.audit_multichannel_examples(
        examples,
        vocabulary,
        expected_counts=EXPECTED_COUNTS,
        expected_groups=EXPECTED_GROUPS,
    )
    if not data_audit["passed"]:
        raise AssertionError("SG16 multichannel data audit failed")
    alphabet = build_action_alphabet(examples)
    alphabet_index = {token: index for index, token in enumerate(alphabet)}
    action_coverage = _action_coverage(corpus_root, vocabulary, alphabet_index)
    if not action_coverage["passed"]:
        raise AssertionError("SG16 live action coverage failed")
    train_hashes = _artifact_hashes(corpus_root, "train")
    frozen_protocol = _frozen_protocol(args)
    confirmation_reproduction = None
    if sg16_reference is not None:
        reference_protocol = sg16_reference["configuration"]["frozen_protocol"]
        current_protocol = json.loads(json.dumps(frozen_protocol))
        confirmation_reproduction = {
            "protocol_equal": current_protocol == reference_protocol,
            "train_artifacts_equal": train_hashes
            == sg16_reference["dataset"]["train_artifact_sha256"],
            "vocabulary_fingerprint_equal": vocabulary.fingerprint
            == sg16_reference["dataset"]["vocabulary_fingerprint"],
            "action_alphabet_equal": list(alphabet)
            == sg16_reference["dataset"]["action_alphabet"],
        }
        if not all(confirmation_reproduction.values()):
            raise AssertionError("SG16R frozen protocol reproduction failed")
    records = {
        split: extract_kernel_records(
            examples[split], alphabet_index=alphabet_index, device=device
        )
        for split in SPLITS
    }
    spike_result, spike_runtime = _fit_spike_kernel(
        records,
        examples,
        batch_groups=args.batch_groups,
        device=device,
    )
    games = _game_records(corpus_root, expected_seeds["test"])
    if args.game_limit:
        games = games[: args.game_limit]
    class_weights = sg10.build_class_weights(examples["train"], device=device)
    replications = []
    for seed in args.seeds:
        build_started = time.perf_counter_ns()
        all_models = sg10.build_multichannel_models(
            10_200_000 + 100 * seed,
            vocabulary,
            d_model=args.d_model,
            state_dim=args.state_dim,
            num_heads=args.num_heads,
            device=device,
        )
        models = {name: all_models[name] for name in ANN_MODEL_NAMES}
        model_build_seconds = (time.perf_counter_ns() - build_started) / 1e9
        schedule = sg10.build_length_stratified_schedule(
            examples["train"],
            epochs=args.epochs,
            batch_groups=args.batch_groups,
            seed=10_201_000 + seed,
        )
        training = {
            name: sg10.train_multichannel(
                name,
                model,
                examples["train"],
                schedule,
                class_weights,
                epochs=args.epochs,
                batches_per_epoch=10,
                device=device,
            )
            for name, model in models.items()
        }
        offline = {
            name: sg10.evaluate_multichannel(
                model,
                examples["test"],
                class_weights,
                device=device,
                include_records=False,
            )
            for name, model in models.items()
        }
        parameter_counts = {
            name: count_parameters(model) for name, model in models.items()
        }
        snn_backend = KernelPlannerBackend(
            alphabet_index=alphabet_index,
            prototype_keys=spike_runtime["prototype_keys"],
            prototype_phases=spike_runtime["prototype_phases"],
            alpha=spike_runtime["alpha"],
            device=device,
        )
        closed_loop = {
            "strict_phase_snn": run_closed_loop_suite(
                snn_backend, games, corpus, max_actions=args.max_actions
            )
        }
        for name, model in models.items():
            closed_loop[name] = run_closed_loop_suite(
                AnnPlannerBackend(name, model, vocabulary, device=device),
                games,
                corpus,
                max_actions=args.max_actions,
            )
        replications.append(
            {
                "seed": seed,
                "model_build_seconds_excluded_from_training": model_build_seconds,
                "parameter_counts": parameter_counts,
                "training": training,
                "offline": offline,
                "closed_loop": closed_loop,
            }
        )
        del models
        del all_models
    decision = _decision(
        data_audit,
        action_coverage,
        spike_result,
        replications,
        fresh_confirmation=args.fresh_confirmation,
        quick=args.quick,
    )
    return {
        "schema_version": 1,
        "experiment": (
            "E3-SG16R fifth-fresh closed-loop candidate planner confirmation"
            if args.fresh_confirmation
            else SG16_EXPERIMENT
        ),
        "formal": not args.quick,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "environment": _environment(device),
        "research_hypothesis": {
            "epistemic_status": (
                "independent fifth procedural game confirmation"
                if args.fresh_confirmation
                else "closed-loop mechanism on already observed fourth games"
            ),
            "statement": (
                "A strict phase-isolated spike suffix memory can drive a real "
                "observation-corrected candidate planner with ANN-level task "
                "quality and lower training/response cost."
            ),
            "what_if": (
                "What if a recoverable spike suffix is a tiny cognitive map "
                "whose predicted deltas are sufficient for real-time action?"
            ),
        },
        "references": {
            "sg15r_frozen_kernel": {
                "path": str(args.sg15r_reference.expanduser().resolve()),
                "sha256": sg15_digest,
            },
            "sg16_frozen_planner": (
                {
                    "path": str(args.sg16_reference.expanduser().resolve()),
                    "sha256": sg16_digest,
                }
                if sg16_reference is not None
                else None
            ),
        },
        "configuration": {
            "corpus_dir": str(corpus_root),
            "fresh_confirmation": args.fresh_confirmation,
            "threads": args.threads if device.type == "cpu" else None,
            "frozen_protocol": frozen_protocol,
            "actual_game_count_per_model": len(games),
        },
        "dataset": {
            "synthetic": False,
            "manifest_provenance": manifest,
            "train_artifact_sha256": train_hashes,
            "vocabulary_fingerprint": vocabulary.fingerprint,
            "action_alphabet": alphabet,
            "audit": data_audit,
            "action_coverage": action_coverage,
        },
        "spike_kernel": spike_result,
        "confirmation_reproduction": confirmation_reproduction,
        "replications": replications,
        "decision": decision,
    }


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-dir", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument(
        "--sg15r-reference", type=Path, default=DEFAULT_SG15R_REFERENCE
    )
    parser.add_argument("--sg16-reference", type=Path, default=DEFAULT_SG16_REFERENCE)
    parser.add_argument("--sg16-reference-sha", type=str, default="")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seeds", nargs="+", type=int, default=(0, 1, 2))
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--d-model", type=int, default=tw0.D_MODEL)
    parser.add_argument("--state-dim", type=int, default=tw0.STATE_DIM)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--batch-groups", type=int, default=16)
    parser.add_argument("--max-actions", type=int, default=15)
    parser.add_argument("--game-limit", type=int, default=0)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--fresh-confirmation", action="store_true")
    args = parser.parse_args(argv)
    if min(
        args.epochs,
        args.threads,
        args.d_model,
        args.state_dim,
        args.num_heads,
        args.batch_groups,
        args.max_actions,
        *args.seeds,
    ) < 0 or min(
        args.epochs,
        args.threads,
        args.d_model,
        args.state_dim,
        args.num_heads,
        args.batch_groups,
        args.max_actions,
    ) <= 0:
        parser.error("numeric experiment controls must be nonnegative seeds or positive")
    if args.d_model % args.num_heads:
        parser.error("d-model must be divisible by num-heads")
    if args.game_limit < 0:
        parser.error("game-limit must be nonnegative")
    if args.game_limit and not args.quick:
        parser.error("game-limit is smoke-only")
    if args.quick:
        args.seeds = args.seeds[:1]
        args.epochs = 2
        args.game_limit = args.game_limit or 2
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
