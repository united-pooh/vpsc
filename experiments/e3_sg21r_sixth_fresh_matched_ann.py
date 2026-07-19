"""SG21R sixth-fresh SNN versus matched-input LSTM/Transformer confirmation."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import torch
from torch import nn
from torch.nn import functional as F


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.e2_f0_fusion_benchmark import _environment, _sync  # noqa: E402
from experiments import e3_sg10_multichannel_delta as sg10  # noqa: E402
from experiments import e3_sg16_closed_loop_planner as sg16  # noqa: E402
from experiments import e3_sg17_two_step_rollout as sg17  # noqa: E402
from experiments import e3_sg18_affordance_weighted_krr as sg18  # noqa: E402
from experiments import e3_sg19_plan_edge_spikes as sg19  # noqa: E402
from experiments import e3_sg21_episodic_edge_graph as sg21  # noqa: E402
from experiments import e3_tw0_sparse_event_lm as tw0  # noqa: E402
from experiments.e3_sg12_spike_delay_rls import (  # noqa: E402
    build_action_alphabet,
)
from vpsc.world_model.event_corpus import load_event_corpus  # noqa: E402
from vpsc.world_model.wikitext import SPLITS, file_sha256  # noqa: E402


DEFAULT_CORPUS = Path("results/e3_scan/textworld_sg21r_l5")
DEFAULT_OUTPUT = Path(
    "results/e3_scan/e3_sg21r_sixth_fresh_matched_ann.json"
)
DEFAULT_CACHE = Path("results/e3_scan/e3_sg21r_fresh_exhaustive_tree_cache.json")
DEFAULT_SG21_REFERENCE = Path("results/e3_scan/e3_sg21_episodic_edge_graph.json")
SG21_REFERENCE_SHA256 = (
    "F1F319E420C3A892B0B66538FA1060BF02BD937BD0F98F2CE089C6D8343EA0C3"
)
SG21_EXPERIMENT = "E3-SG21 episodic edge spikes and causal output projection"
EXPECTED_SEEDS = sg17.CONFIRMATION_SEEDS
EXPECTED_EXHAUSTIVE_COUNTS = {"train": 640, "valid": 160, "test": 160}
TRAIN_ARTIFACT_SHA256 = {
    "manifest.json": "31CCA6B192FFD9C73C35401A51E781C6872E6A8BCAF51C1C415D38D228DA0D48",
    "episodes.jsonl": "2399F060AAC410CE11216E059918FC3DBB6F00EA9C814A2E5140D654C4E6997D",
    "token_events.txt": "638E27D86E9B06332D804EE1FB10F7B70360A24CEA3D869202D74FD734292C22",
}
MODEL_NAMES = ("lstm", "transformer")
TRAINING_SEEDS = (0, 1, 2)
D_MODEL = 32
TRANSFORMER_HEADS = 4
TRANSFORMER_FFN = 64
EPOCHS = 50
BATCH_SIZE = 64
LEARNING_RATE = 3e-3
WEIGHT_DECAY = 1e-4
MAX_PHASE = 6


@dataclass(frozen=True)
class FeatureVocabulary:
    tokens: Tuple[str, ...]
    token_to_id: Mapping[str, int]


def build_feature_vocabulary(
    action_alphabet: Sequence[str], action_order: Sequence[str]
) -> FeatureVocabulary:
    actions = tuple(sorted(set(str(value) for value in action_alphabet)))
    values = {"<feature_pad>"}
    values.update(f"phase:{phase}" for phase in range(MAX_PHASE + 1))
    for position in range(3):
        values.add(f"history:{position}:<pad>")
        values.update(f"history:{position}:{action}" for action in actions)
    values.update(f"candidate:{action}" for action in action_order)
    for slot in ("current", "next"):
        values.add(f"plan_{slot}:{sg19.PLAN_PAD}")
        values.update(f"plan_{slot}:{action}" for action in actions)
    values.update(("return:0", "return:1"))
    for action in action_order:
        values.update((f"mask:{action}:0", f"mask:{action}:1"))
    tokens = tuple(sorted(values))
    return FeatureVocabulary(
        tokens=tokens,
        token_to_id={token: index for index, token in enumerate(tokens)},
    )


def feature_tokens(
    context_actions: Sequence[str],
    current_mask: Sequence[int],
    candidate_action: str,
    plan: Sequence[str],
    last_move: Optional[str],
    action_order: Sequence[str],
) -> Tuple[str, ...]:
    phase = len(context_actions)
    if phase > MAX_PHASE:
        raise ValueError(f"matched feature phase {phase} exceeds {MAX_PHASE}")
    history = ("<pad>",) * max(0, 3 - len(context_actions)) + tuple(
        context_actions[-3:]
    )
    plan_current, plan_next = sg19._plan_slots(plan, phase)
    values = [f"phase:{phase}"]
    values.extend(
        f"history:{position}:{action}"
        for position, action in enumerate(history)
    )
    values.extend(
        (
            f"candidate:{candidate_action}",
            f"plan_current:{plan_current}",
            f"plan_next:{plan_next}",
            f"return:{sg19.return_edge_spike(last_move, candidate_action)}",
        )
    )
    values.extend(
        f"mask:{action}:{int(bit)}"
        for action, bit in zip(action_order, current_mask)
    )
    if len(values) != 16:
        raise AssertionError(f"matched feature length is {len(values)}, not 16")
    return tuple(values)


def encode_feature_tokens(
    tokens: Sequence[str], vocabulary: FeatureVocabulary
) -> Tuple[int, ...]:
    try:
        return tuple(vocabulary.token_to_id[token] for token in tokens)
    except KeyError as exc:
        raise KeyError(f"matched ANN feature is OOV: {exc.args[0]}") from exc


def tensorize_matched_features(
    records: Sequence[Mapping[str, Any]],
    plans: Mapping[int, Sequence[str]],
    vocabulary: FeatureVocabulary,
    action_order: Sequence[str],
    *,
    device: torch.device,
) -> Tuple[torch.Tensor, float]:
    started = time.perf_counter_ns()
    rows = []
    for record in records:
        context = tuple(str(action) for action in record["context_actions"])
        candidate = str(record["candidate_action"])
        rows.append(
            encode_feature_tokens(
                feature_tokens(
                    context,
                    record["current_mask"],
                    candidate,
                    plans[int(record["game_seed"])],
                    sg19._last_move(context),
                    action_order,
                ),
                vocabulary,
            )
        )
    tensor = torch.tensor(rows, dtype=torch.long, device=device)
    return tensor, (time.perf_counter_ns() - started) / 1e9


class MatchedLSTM(nn.Module):
    def __init__(self, vocabulary_size: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocabulary_size, D_MODEL)
        self.core = nn.LSTM(D_MODEL, D_MODEL, num_layers=1, batch_first=True)
        self.norm = nn.LayerNorm(D_MODEL)
        self.head = nn.Linear(D_MODEL, sg18.NEXT_MASK_OFFSET + 8)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(token_ids)
        output, _state = self.core(embedded)
        return self.head(self.norm(output[:, -1]))


class MatchedTransformer(nn.Module):
    def __init__(self, vocabulary_size: int, sequence_length: int = 16) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocabulary_size, D_MODEL)
        self.position = nn.Parameter(torch.zeros(sequence_length, D_MODEL))
        layer = nn.TransformerEncoderLayer(
            d_model=D_MODEL,
            nhead=TRANSFORMER_HEADS,
            dim_feedforward=TRANSFORMER_FFN,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=False,
        )
        self.core = nn.TransformerEncoder(layer, num_layers=1)
        self.norm = nn.LayerNorm(D_MODEL)
        self.head = nn.Linear(D_MODEL, sg18.NEXT_MASK_OFFSET + 8)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(token_ids) + self.position[None, :, :]
        output = self.core(embedded)
        return self.head(self.norm(output[:, -1]))


def build_matched_model(name: str, vocabulary_size: int, seed: int) -> nn.Module:
    torch.manual_seed(seed + (100_000 if name == "lstm" else 200_000))
    if name == "lstm":
        return MatchedLSTM(vocabulary_size)
    if name == "transformer":
        return MatchedTransformer(vocabulary_size)
    raise KeyError(name)


def _parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def _raw_metrics(
    logits: torch.Tensor,
    tensors: Mapping[str, Any],
    action_order: Sequence[str],
) -> Dict[str, Any]:
    return {
        "delta": sg10._ridge_multichannel_metrics(
            logits[:, : sg10.TOTAL_LOGITS].to(torch.float64),
            tensors["targets"],
            tensors["group_ids"],
        ),
        "next_affordance": sg18._mask_metrics(
            logits[:, sg18.NEXT_MASK_OFFSET :].to(torch.float64),
            tensors["next_masks"],
            action_order,
        ),
    }


def _nineteen_bit_accuracy(
    logits: torch.Tensor, target_code: torch.Tensor
) -> float:
    targets = target_code > 0.0
    return float(((logits > 0.0) == targets).to(torch.float64).mean().item())


def train_matched_model(
    name: str,
    seed: int,
    train_tokens: torch.Tensor,
    train_target_code: torch.Tensor,
    train_tensors: Mapping[str, Any],
    action_order: Sequence[str],
    vocabulary_size: int,
    feature_encoding_seconds: float,
    *,
    device: torch.device,
) -> Tuple[nn.Module, Dict[str, Any]]:
    model = build_matched_model(name, vocabulary_size, seed).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    targets = ((train_target_code + 1.0) * 0.5).to(torch.float32)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed + 300_000)
    losses = []
    _sync(device)
    started = time.perf_counter_ns()
    model.train()
    for _epoch in range(EPOCHS):
        permutation = torch.randperm(
            train_tokens.shape[0], generator=generator
        ).to(device)
        epoch_loss = 0.0
        for offset in range(0, train_tokens.shape[0], BATCH_SIZE):
            indices = permutation[offset : offset + BATCH_SIZE]
            optimizer.zero_grad(set_to_none=True)
            logits = model(train_tokens.index_select(0, indices))
            loss = F.binary_cross_entropy_with_logits(
                logits, targets.index_select(0, indices)
            )
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.detach().item()) * int(indices.numel())
        losses.append(epoch_loss / train_tokens.shape[0])
    _sync(device)
    optimization_seconds = (time.perf_counter_ns() - started) / 1e9
    model.eval()
    with torch.no_grad():
        train_logits = model(train_tokens)
    raw = _raw_metrics(train_logits, train_tensors, action_order)
    bit_accuracy = _nineteen_bit_accuracy(train_logits, train_target_code)
    return model, {
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "optimizer": "AdamW",
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "loss": "BCEWithLogitsLoss on 19 +/-1 targets",
        "feature_encoding_seconds": feature_encoding_seconds,
        "optimization_wall_seconds": optimization_seconds,
        "deployment_training_wall_seconds": (
            feature_encoding_seconds + optimization_seconds
        ),
        "initial_epoch_loss": losses[0],
        "final_epoch_loss": losses[-1],
        "parameter_count": _parameter_count(model),
        "parameter_bytes": _parameter_count(model) * 4,
        "train_nineteen_bit_accuracy": bit_accuracy,
        "train_delta_exact_accuracy": raw["delta"]["exact_vector_accuracy"],
        "train_next_mask_exact_accuracy": raw["next_affordance"][
            "exact_mask_accuracy"
        ],
        "valid_baseline": (
            bit_accuracy >= 0.98
            and raw["delta"]["exact_vector_accuracy"] >= 0.95
        ),
    }


class MatchedAnnScorer:
    def __init__(
        self,
        model: nn.Module,
        vocabulary: FeatureVocabulary,
        action_order: Sequence[str],
        device: torch.device,
    ) -> None:
        self.model = model
        self.vocabulary = vocabulary
        self.action_order = tuple(action_order)
        self.device = device

    def __call__(
        self,
        context_actions: Sequence[str],
        current_mask: Sequence[int],
        candidate_action: str,
        plan: Sequence[str],
        last_move: Optional[str],
    ) -> Tuple[torch.Tensor, Tuple[int, ...], float]:
        _sync(self.device)
        started = time.perf_counter_ns()
        token_ids = encode_feature_tokens(
            feature_tokens(
                context_actions,
                current_mask,
                candidate_action,
                plan,
                last_move,
                self.action_order,
            ),
            self.vocabulary,
        )
        tensor = torch.tensor(
            (token_ids,), dtype=torch.long, device=self.device
        )
        with torch.no_grad():
            scores = self.model(tensor)
        scores.sum().item()
        _sync(self.device)
        mask = tuple(
            int(value)
            for value in (
                scores[0, sg18.NEXT_MASK_OFFSET :] > 0.0
            ).cpu().tolist()
        )
        return scores, mask, (time.perf_counter_ns() - started) / 1e6


class SnnEndToEndScorer:
    def __init__(
        self,
        alphabet_index: Mapping[str, int],
        unique: Mapping[str, Any],
        coefficients: torch.Tensor,
        device: torch.device,
    ) -> None:
        self.alphabet_index = alphabet_index
        self.unique = {
            name: tensor.to(
                device=device,
                dtype=(torch.float32 if name == "masks" else tensor.dtype),
            )
            for name, tensor in unique.items()
            if isinstance(tensor, torch.Tensor)
            and name
            in (
                "keys",
                "phases",
                "masks",
                "plan_current",
                "plan_next",
                "return_edges",
            )
        }
        self.coefficients = coefficients.to(device=device, dtype=torch.float32)
        self.device = device

    def __call__(
        self,
        context_actions: Sequence[str],
        current_mask: Sequence[int],
        candidate_action: str,
        plan: Sequence[str],
        last_move: Optional[str],
    ) -> Tuple[torch.Tensor, Tuple[int, ...], float]:
        _sync(self.device)
        started = time.perf_counter_ns()
        scores, mask, _model_only = sg19._runtime_score(
            context_actions,
            current_mask,
            candidate_action,
            plan,
            last_move,
            alphabet_index=self.alphabet_index,
            unique=self.unique,
            coefficients=self.coefficients,
            device=self.device,
        )
        return scores, mask, (time.perf_counter_ns() - started) / 1e6


def evaluate_logits_with_graph(
    logits: torch.Tensor,
    tensors: Mapping[str, Any],
    records: Sequence[Mapping[str, Any]],
    snapshots: Mapping[Tuple[int, int], sg21.GraphState],
    action_order: Sequence[str],
    split_name: str,
    plans: Optional[Mapping[int, Sequence[str]]] = None,
    enforce_plan_path_constraint: bool = False,
) -> Dict[str, Any]:
    predictions = []
    masks = []
    kinds: Dict[str, int] = {}
    for index, record in enumerate(records):
        base_indices = sg21._indices_from_scores(logits[index : index + 1])
        predicted_mask = tuple(
            int(value)
            for value in (
                logits[index, sg18.NEXT_MASK_OFFSET :] > 0.0
            ).tolist()
        )
        seed = int(record["game_seed"])
        root_step = int(record["root_step"])
        prediction, mask, _graph, audit = sg21.project_graph_transition(
            base_indices,
            predicted_mask,
            snapshots[(seed, root_step)],
            str(record["candidate_action"]),
            action_order,
            branch_tag=f"{split_name}:{seed}:{root_step}",
            plan=(plans[seed] if plans is not None else None),
            phase=len(record["context_actions"]),
            enforce_plan_path_constraint=enforce_plan_path_constraint,
        )
        predictions.append(prediction)
        masks.append(mask)
        kind = str(audit["kind"])
        kinds[kind] = kinds.get(kind, 0) + 1
    targets = tuple(
        tuple(int(value) for value in row) for row in tensors["targets"]
    )
    mask_scores = torch.tensor(masks, dtype=torch.float64) * 2.0 - 1.0
    return {
        "delta": sg17._rollout_metrics(predictions, targets),
        "next_affordance": sg18._mask_metrics(
            mask_scores, tensors["next_masks"], action_order
        ),
        "projection_kind_counts": kinds,
    }


def _artifact_hashes(corpus_root: Path) -> Dict[str, Dict[str, str]]:
    return {
        split: {
            name: file_sha256(corpus_root / split / name).upper()
            for name in ("manifest.json", "episodes.jsonl", "token_events.txt")
        }
        for split in SPLITS
    }


def _cache_identity(
    corpus_root: Path,
    artifact_hashes: Mapping[str, Any],
    expected_seeds: Mapping[str, Sequence[int]] = EXPECTED_SEEDS,
) -> Dict[str, Any]:
    return {
        "schema_version": 1,
        "corpus_dir": str(corpus_root),
        "artifact_hashes": artifact_hashes,
        "expected_seeds": {
            split: list(expected_seeds[split]) for split in SPLITS
        },
    }


def collect_or_load_fresh_cache(
    args: argparse.Namespace,
    corpus_root: Path,
    corpus: Any,
    action_order: Sequence[str],
    artifact_hashes: Mapping[str, Any],
    expected_seeds: Mapping[str, Sequence[int]] = EXPECTED_SEEDS,
) -> Tuple[Dict[str, Any], str, bool, float]:
    cache_path = args.cache.expanduser().resolve()
    identity = _cache_identity(corpus_root, artifact_hashes, expected_seeds)
    if cache_path.is_file() and not args.refresh_cache:
        payload = json.loads(cache_path.read_text(encoding="utf-8"))
        if payload.get("identity") != identity:
            raise ValueError("SG21R cache identity mismatch")
        return payload, file_sha256(cache_path).upper(), True, 0.0
    started = time.perf_counter_ns()
    games = {
        split: sg16._game_records(
            corpus_root, expected_seeds[split], split=split
        )
        for split in SPLITS
    }
    exhaustive = {
        split: sg18.collect_exhaustive_split(
            corpus_root, corpus, split, games[split], action_order
        )
        for split in SPLITS
    }
    tree = sg17.collect_two_step_tree(corpus_root, corpus, games["test"])
    elapsed = (time.perf_counter_ns() - started) / 1e9
    payload = {
        "identity": identity,
        "collection_wall_seconds": elapsed,
        "exhaustive": exhaustive,
        "branch_tree": tree,
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return payload, file_sha256(cache_path).upper(), False, elapsed


def _data_gate(
    manifest: Mapping[str, Any],
    data_audit: Mapping[str, Any],
    artifact_hashes: Mapping[str, Mapping[str, str]],
    cache: Mapping[str, Any],
    graph_audit: Mapping[str, Any],
    fresh_game_hash_overlap_count: int,
) -> Tuple[bool, Dict[str, Any]]:
    exhaustive = cache["exhaustive"]
    tree = cache["branch_tree"]
    train_equal = all(
        artifact_hashes["train"][name] == expected
        for name, expected in TRAIN_ARTIFACT_SHA256.items()
    )
    exhaustive_counts = {
        split: int(exhaustive[split]["record_count"]) for split in SPLITS
    }
    exhaustive_ok = exhaustive_counts == EXPECTED_EXHAUSTIVE_COUNTS and all(
        exhaustive[split]["all_live_factual_won"]
        and exhaustive[split]["all_counterfactuals_non_mutating"]
        for split in SPLITS
    )
    tree_ok = (
        tree["game_count"] == 8
        and tree["root_count"] == 40
        and tree["first_branch_count"] == 160
        and tree["second_pair_count"] == 616
        and tree["all_live_factual_won"]
        and tree["all_counterfactuals_non_mutating"]
    )
    graph_ok = (
        graph_audit["all_masks_match_exhaustive_cache"]
        and graph_audit["all_binding_steps_precede_snapshot_root"]
        and graph_audit["all_rooms_unique_and_present"]
        and graph_audit["no_edge_conflicts"]
    )
    passed = bool(
        manifest
        and data_audit["passed"]
        and train_equal
        and fresh_game_hash_overlap_count == 0
        and exhaustive_ok
        and tree_ok
        and graph_ok
    )
    return passed, {
        "train_artifacts_equal_fifth": train_equal,
        "fresh_valid_test_game_hash_overlap_with_fifth": (
            fresh_game_hash_overlap_count
        ),
        "exhaustive_counts": exhaustive_counts,
        "exhaustive_audit_passed": exhaustive_ok,
        "tree_audit_passed": tree_ok,
        "graph_audit_passed": graph_ok,
    }


def _decision(
    data_passed: bool,
    snn: Mapping[str, Any],
    ann: Sequence[Mapping[str, Any]],
    snn_bytes: int,
    max_graph_bytes: int,
) -> Dict[str, Any]:
    snn_test = snn["graph_split_metrics"]["test"]
    snn_rollout = snn["rollout"]
    snn_quality = (
        snn_test["delta"]["exact_vector_accuracy"] == 1.0
        and all(
            value == 1.0
            for value in snn_test["delta"]["channel_accuracy"].values()
        )
        and snn_test["next_affordance"]["bit_accuracy"] >= 0.98
        and snn_test["next_affordance"]["exact_mask_accuracy"] >= 0.95
        and snn_rollout["teacher_forced_second"]["exact_vector_accuracy"] == 1.0
        and snn_rollout["self_rollout_second"]["exact_vector_accuracy"] == 1.0
        and all(
            value == 1.0
            for value in snn_rollout["self_rollout_second"][
                "channel_accuracy"
            ].values()
        )
        and snn_rollout["first_routing_accuracy"] == 1.0
        and snn_rollout["premature_stop_first_branch_count"] == 0
    )
    ann_valid = all(record["training"]["valid_baseline"] for record in ann)
    matched_quality = ann_valid and all(
        snn_test["next_affordance"]["exact_mask_accuracy"]
        >= record["graph_split_metrics"]["test"]["next_affordance"][
            "exact_mask_accuracy"
        ]
        and snn_rollout["self_rollout_second"]["exact_vector_accuracy"]
        >= record["rollout"]["self_rollout_second"]["exact_vector_accuracy"]
        for record in ann
    )
    training_comparisons = []
    response_comparisons = []
    storage_comparisons = []
    for record in ann:
        training_comparisons.append(
            {
                "seed": record["seed"],
                "model": record["model"],
                "snn_seconds": snn["training_wall_seconds"],
                "ann_seconds": record["training"][
                    "graph_plus_model_training_wall_seconds"
                ],
                "passed": snn["training_wall_seconds"]
                < record["training"]["graph_plus_model_training_wall_seconds"],
            }
        )
        response_comparisons.append(
            {
                "seed": record["seed"],
                "model": record["model"],
                "snn_p50_ms": snn_rollout["teacher_pair_timing"]["p50_ms"],
                "ann_p50_ms": record["rollout"]["teacher_pair_timing"][
                    "p50_ms"
                ],
                "snn_p95_ms": snn_rollout["teacher_pair_timing"]["p95_ms"],
                "ann_p95_ms": record["rollout"]["teacher_pair_timing"][
                    "p95_ms"
                ],
                "passed": (
                    snn_rollout["teacher_pair_timing"]["p50_ms"]
                    < record["rollout"]["teacher_pair_timing"]["p50_ms"]
                    and snn_rollout["teacher_pair_timing"]["p95_ms"]
                    < record["rollout"]["teacher_pair_timing"]["p95_ms"]
                ),
            }
        )
        ann_bytes = record["training"]["parameter_bytes"] + max_graph_bytes
        storage_comparisons.append(
            {
                "seed": record["seed"],
                "model": record["model"],
                "snn_bytes": snn_bytes,
                "ann_bytes": ann_bytes,
                "passed": snn_bytes <= ann_bytes,
            }
        )
    training = all(value["passed"] for value in training_comparisons)
    response = all(value["passed"] for value in response_comparisons)
    storage = all(value["passed"] for value in storage_comparisons)
    gates = {
        "data_no_leak_gate": data_passed,
        "ann_validity_gate": ann_valid,
        "snn_quality_gate": snn_quality,
        "matched_quality_gate": matched_quality,
        "training_speed_gate": training,
        "response_speed_gate": response,
        "storage_gate": storage,
    }
    overall = "PASS" if all(gates.values()) else "FAIL"
    if overall == "PASS":
        next_route = "sg22_scale_sparse_primal_pcg_and_multimodal_state"
    elif not ann_valid:
        next_route = "sg21r_matched_ann_same_budget_optimization_diagnostic"
    elif not snn_quality or not matched_quality:
        next_route = "sg21r_unknown_residual_quality_diagnostic"
    else:
        next_route = "sg22_sparse_primal_pcg_woodbury_acceleration"
    return {
        **{name: "PASS" if value else "FAIL" for name, value in gates.items()},
        "overall": overall,
        "training_comparisons": training_comparisons,
        "response_comparisons": response_comparisons,
        "storage_comparisons": storage_comparisons,
        "next_route": next_route,
    }


def run_experiment(args: argparse.Namespace) -> Dict[str, Any]:
    device = torch.device("cpu")
    torch.set_num_threads(args.threads)
    corpus_root = args.corpus_dir.expanduser().resolve()
    sg21_reference, sg21_digest = sg16._load_reference(
        args.sg21_reference.expanduser().resolve(),
        SG21_REFERENCE_SHA256,
        SG21_EXPERIMENT,
    )
    if sg21_reference["decision"]["overall"] != "PASS":
        raise ValueError("SG21R requires passing SG21 mechanism reference")
    manifest = tw0._manifest_provenance(
        corpus_root, expected_seeds_by_split=EXPECTED_SEEDS
    )
    artifact_hashes = _artifact_hashes(corpus_root)
    corpus = load_event_corpus(corpus_root)
    examples, vocabulary = sg10.build_multichannel_examples(corpus_root, corpus)
    data_audit = sg10.audit_multichannel_examples(
        examples,
        vocabulary,
        expected_counts=sg16.EXPECTED_COUNTS,
        expected_groups=sg16.EXPECTED_GROUPS,
    )
    alphabet = build_action_alphabet(examples)
    alphabet_index = {token: index for index, token in enumerate(alphabet)}
    action_order = sg18._action_order(corpus_root)
    plans, plan_audit = sg19.load_objective_plans(corpus_root)
    if not plan_audit["all_plans_equal_walkthrough_for_audit"]:
        raise AssertionError("SG21R objective plan compiler audit failed")
    cache, cache_digest, cache_reused, collection_wall = (
        collect_or_load_fresh_cache(
            args, corpus_root, corpus, action_order, artifact_hashes
        )
    )
    exhaustive = cache["exhaustive"]
    tree = cache["branch_tree"]
    repaired_tree, repair_audit = sg17.repair_persistent_room_semantics(tree)
    if repair_audit["changed_pair_count"] != 0:
        raise AssertionError("fresh SG21R tree unexpectedly needs legacy repair")
    tensors = {
        split: sg19.tensorize_extended(
            exhaustive[split]["records"],
            plans[split],
            alphabet_index=alphabet_index,
            device=device,
        )
        for split in SPLITS
    }
    unique = sg19.compress_extended(tensors["train"])
    coefficients, base_fit = sg19.fit_weighted_extended(
        tensors["train"], unique, device=device
    )
    snapshots, graph_audit = sg21.build_graph_snapshots(
        corpus_root, corpus, exhaustive, action_order
    )
    graph_split_metrics = {
        split: sg21.evaluate_split_with_graph(
            split,
            tensors[split],
            exhaustive[split]["records"],
            snapshots[split],
            unique,
            coefficients,
            action_order,
        )
        for split in SPLITS
    }
    snn_scorer = SnnEndToEndScorer(
        alphabet_index, unique, coefficients, device
    )
    snn_rollout = sg21.evaluate_two_step_with_graph(
        repaired_tree,
        exhaustive["test"]["records"],
        snapshots["test"],
        plans["test"],
        alphabet_index=alphabet_index,
        unique=unique,
        coefficients=coefficients,
        action_order=action_order,
        device=device,
        score_fn=snn_scorer,
    )
    max_graph_bytes = max(
        graph_audit["splits"][split]["maximum_logical_graph_bytes"]
        for split in SPLITS
    )
    base_model_bytes = int(
        unique["keys"].shape[0]
        * (
            unique["keys"].shape[1]
            + 1
            + len(action_order)
            + 3
            + coefficients.shape[1] * 4
        )
    )
    snn_training_wall = (
        base_fit["deployment_training_wall_seconds"]
        + graph_audit["splits"]["train"]["build_seconds"]
    )
    snn_result = {
        "base_weighted_fit": base_fit,
        "graph_split_metrics": graph_split_metrics,
        "rollout": snn_rollout,
        "training_wall_seconds": snn_training_wall,
        "base_model_logical_bytes": base_model_bytes,
        "combined_logical_bytes": base_model_bytes + max_graph_bytes,
    }

    feature_vocabulary = build_feature_vocabulary(action_order, action_order)
    matched_tokens = {}
    encoding_seconds = {}
    for split in SPLITS:
        matched_tokens[split], encoding_seconds[split] = tensorize_matched_features(
            exhaustive[split]["records"],
            plans[split],
            feature_vocabulary,
            action_order,
            device=device,
        )
    ann_results = []
    for seed in TRAINING_SEEDS:
        for name in MODEL_NAMES:
            model, training = train_matched_model(
                name,
                seed,
                matched_tokens["train"],
                tensors["train"]["target_code"],
                tensors["train"],
                action_order,
                len(feature_vocabulary.tokens),
                encoding_seconds["train"],
                device=device,
            )
            training["graph_plus_model_training_wall_seconds"] = (
                training["deployment_training_wall_seconds"]
                + graph_audit["splits"]["train"]["build_seconds"]
            )
            raw_metrics = {}
            graph_metrics = {}
            with torch.no_grad():
                for split in SPLITS:
                    logits = model(matched_tokens[split])
                    raw_metrics[split] = _raw_metrics(
                        logits, tensors[split], action_order
                    )
                    graph_metrics[split] = evaluate_logits_with_graph(
                        logits,
                        tensors[split],
                        exhaustive[split]["records"],
                        snapshots[split],
                        action_order,
                        split,
                    )
            scorer = MatchedAnnScorer(
                model, feature_vocabulary, action_order, device
            )
            rollout = sg21.evaluate_two_step_with_graph(
                repaired_tree,
                exhaustive["test"]["records"],
                snapshots["test"],
                plans["test"],
                alphabet_index=alphabet_index,
                unique=unique,
                coefficients=coefficients,
                action_order=action_order,
                device=device,
                score_fn=scorer,
            )
            ann_results.append(
                {
                    "seed": seed,
                    "model": name,
                    "training": training,
                    "raw_split_metrics": raw_metrics,
                    "graph_split_metrics": graph_metrics,
                    "rollout": rollout,
                }
            )

    old_corpus_root = Path(
        str(sg21_reference["configuration"]["corpus_dir"])
    )
    old_game_hashes = {
        str(episode["game_sha256"])
        for split in ("valid", "test")
        for episode in (
            json.loads(line)
            for line in (old_corpus_root / split / "episodes.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
        )
    }
    fresh_game_hash_overlap_count = sum(
        str(episode["game_sha256"]) in old_game_hashes
        for split in ("valid", "test")
        for episode in (
            json.loads(line)
            for line in (corpus_root / split / "episodes.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
        )
    )
    data_passed, data_gate_audit = _data_gate(
        manifest,
        data_audit,
        artifact_hashes,
        cache,
        graph_audit,
        fresh_game_hash_overlap_count,
    )
    decision = _decision(
        data_passed,
        snn_result,
        ann_results,
        base_model_bytes + max_graph_bytes,
        max_graph_bytes,
    )
    return {
        "schema_version": 1,
        "experiment": "E3-SG21R sixth-fresh matched ANN graph confirmation",
        "formal": not args.quick,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "environment": _environment(device),
        "research_hypothesis": {
            "epistemic_status": "independent sixth-fresh confirmation",
            "statement": (
                "With identical structured state and episodic graph, the "
                "closed-form SNN residual should retain quality while training "
                "and responding faster than LSTM and Transformer."
            ),
            "what_if": (
                "What if shared deterministic memory reveals the residual "
                "learner, rather than architecture-specific input advantages?"
            ),
        },
        "references": {
            "sg21_mechanism": {
                "path": str(args.sg21_reference.expanduser().resolve()),
                "sha256": sg21_digest,
            },
            "fresh_cache": {
                "path": str(args.cache.expanduser().resolve()),
                "sha256": cache_digest,
                "reused": cache_reused,
                "collection_wall_seconds": collection_wall,
            },
        },
        "configuration": {
            "corpus_dir": str(corpus_root),
            "threads": args.threads,
            "training_seeds": TRAINING_SEEDS,
            "matched_sequence_slots": 16,
            "feature_vocabulary_size": len(feature_vocabulary.tokens),
            "feature_vocabulary": feature_vocabulary.tokens,
            "d_model": D_MODEL,
            "transformer_heads": TRANSFORMER_HEADS,
            "transformer_ffn": TRANSFORMER_FFN,
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "timing_boundary": "feature encoding plus model forward plus graph projection",
        },
        "dataset": {
            "manifest": manifest,
            "data_audit": data_audit,
            "data_gate_audit": data_gate_audit,
            "artifact_hashes": artifact_hashes,
            "vocabulary_fingerprint": vocabulary.fingerprint,
            "action_alphabet": alphabet,
            "action_order": action_order,
            "objective_plan_audit": plan_audit,
            "tree_repair_audit": repair_audit,
            "graph_audit": graph_audit,
            "fresh_tree": {
                name: value
                for name, value in tree.items()
                if name not in ("first_records", "games")
            },
        },
        "snn": snn_result,
        "matched_ann": ann_results,
        "decision": decision,
    }


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-dir", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument(
        "--sg21-reference", type=Path, default=DEFAULT_SG21_REFERENCE
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--refresh-cache", action="store_true")
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args(argv)
    if args.threads <= 0:
        parser.error("--threads must be positive")
    return args


def main() -> None:
    args = _parse_args()
    result = run_experiment(args)
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result["decision"], indent=2, sort_keys=True))
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
