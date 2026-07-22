"""CRWM-1 no-oracle candidate generation over the frozen SG22R substrate.

The projected first-step affordance mask must propose the complete second-step
candidate set before the evaluator materializes official candidates or targets.
Only self-routed context is allowed for the second transition.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.e2_f0_fusion_benchmark import _environment  # noqa: E402
from experiments import e3_sg10_multichannel_delta as sg10  # noqa: E402
from experiments import e3_sg16_closed_loop_planner as sg16  # noqa: E402
from experiments import e3_sg17_two_step_rollout as sg17  # noqa: E402
from experiments import e3_sg18_affordance_weighted_krr as sg18  # noqa: E402
from experiments import e3_sg19_plan_edge_spikes as sg19  # noqa: E402
from experiments import e3_sg21_episodic_edge_graph as sg21  # noqa: E402
from experiments import e3_sg21r_sixth_fresh_matched_ann as sg21r  # noqa: E402
from experiments import e3_sg22_plan_path_constraints as sg22  # noqa: E402
from experiments import e3_sg22r_seventh_fresh_confirmation as sg22r  # noqa: E402
from experiments.e3_sg12_spike_delay_rls import (  # noqa: E402
    build_action_alphabet,
)
from vpsc.world_model.event_corpus import load_event_corpus  # noqa: E402
from vpsc.world_model.wikitext import SPLITS, file_sha256  # noqa: E402


DEFAULT_CORPUS = Path("results/e3_scan/textworld_sg22r_l5")
DEFAULT_CACHE = Path("results/e3_scan/e3_sg22r_fresh_exhaustive_tree_cache.json")
DEFAULT_REFERENCE = Path(
    "results/e3_scan/e3_sg22r_seventh_fresh_confirmation.json"
)
DEFAULT_OUTPUT = Path("results/e3_scan/e3_crwm1_no_oracle_candidates.json")
REFERENCE_SHA256 = (
    "1A75839740A7913E555FBEBD5EB462AA4C50D5324709B11F507A9FB607B7DB92"
)
CACHE_SHA256 = (
    "2016BF42DF694FBE6F4EDCD81E21C03E09F4A92348BDDB8909DD0118A2565A5E"
)
EXPERIMENT = "E3-CRWM1 no-oracle candidate generation"


def candidate_actions_from_mask(
    mask: Sequence[int], action_order: Sequence[str]
) -> Tuple[str, ...]:
    if len(mask) != len(action_order):
        raise ValueError("candidate mask and action order must have equal length")
    if any(int(bit) not in (0, 1) for bit in mask):
        raise ValueError("candidate mask must be binary")
    return tuple(
        action for action, bit in zip(action_order, mask) if int(bit) == 1
    )


def candidate_set_counts(
    proposed: Sequence[str], official: Sequence[str]
) -> Dict[str, Any]:
    proposed_set = set(proposed)
    official_set = set(official)
    if len(proposed_set) != len(tuple(proposed)):
        raise ValueError("proposed candidates must be unique")
    if len(official_set) != len(tuple(official)):
        raise ValueError("official candidates must be unique")
    true_positive = len(proposed_set & official_set)
    false_positive = len(proposed_set - official_set)
    false_negative = len(official_set - proposed_set)
    return {
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "exact": proposed_set == official_set,
    }


def _runtime_scorer(
    alphabet_index: Mapping[str, int],
    unique: Mapping[str, Any],
    coefficients: torch.Tensor,
    device: torch.device,
):
    runtime_unique = {
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
    runtime_coefficients = coefficients.to(device=device, dtype=torch.float32)

    def score(
        context_actions: Sequence[str],
        current_mask: Sequence[int],
        candidate_action: str,
        plan: Sequence[str],
        last_move: Optional[str],
    ) -> Tuple[torch.Tensor, Tuple[int, ...], float]:
        return sg19._runtime_score(
            context_actions,
            current_mask,
            candidate_action,
            plan,
            last_move,
            alphabet_index=alphabet_index,
            unique=runtime_unique,
            coefficients=runtime_coefficients,
            device=device,
        )

    return score


def evaluate_no_oracle_two_step(
    tree: Mapping[str, Any],
    exhaustive_test: Sequence[Mapping[str, Any]],
    snapshots: Mapping[Tuple[int, int], sg21.GraphState],
    plans: Mapping[int, Sequence[str]],
    *,
    action_order: Sequence[str],
    score_fn: Any,
) -> Dict[str, Any]:
    lookup = {
        (
            int(record["game_seed"]),
            int(record["root_step"]),
            str(record["candidate_action"]),
        ): record
        for record in exhaustive_test
    }
    group_records = []
    total_true_positive = 0
    total_false_positive = 0
    total_false_negative = 0
    exact_candidate_sets = 0
    proposed_count = 0
    official_count = 0
    second_correct = 0
    second_evaluated = 0
    second_union_count = 0
    self_context_failures = 0
    first_latencies = []
    second_latencies = []
    errors = []

    for first in tree["first_records"]:
        seed = int(first["game_seed"])
        root_step = int(first["root_step"])
        first_action = str(first["action"])
        context = tuple(first["context_actions"])
        root_record = lookup[(seed, root_step, first_action)]
        started = time.perf_counter_ns()
        first_scores, predicted_mask, _model_elapsed = score_fn(
            context,
            root_record["current_mask"],
            first_action,
            plans[seed],
            sg19._last_move(context),
        )
        base_summary = sg16.decode_prediction(
            first_scores[:, : sg10.TOTAL_LOGITS]
        )
        first_prediction, first_mask, branch_graph, first_projection = (
            sg21.project_graph_transition(
                sg21._indices_from_scores(first_scores),
                predicted_mask,
                snapshots[(seed, root_step)],
                first_action,
                action_order,
                branch_tag=f"crwm1:{seed}:{root_step}:{first_action}",
                plan=plans[seed],
                phase=len(context),
                enforce_plan_path_constraint=True,
            )
        )

        # This is the frozen information barrier: proposal is materialized from
        # the model projection before official second candidates are read.
        proposed = candidate_actions_from_mask(first_mask, action_order)
        proposal_fingerprint = hashlib.sha256(
            json.dumps(proposed, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        first_latencies.append((time.perf_counter_ns() - started) / 1e6)

        official_targets = {
            str(second["action"]): tuple(int(v) for v in second["target_indices"])
            for second in first["seconds"]
        }
        official = tuple(sorted(official_targets))
        counts = candidate_set_counts(proposed, official)
        exact_candidate_sets += int(counts["exact"])
        total_true_positive += int(counts["true_positive"])
        total_false_positive += int(counts["false_positive"])
        total_false_negative += int(counts["false_negative"])
        proposed_count += len(proposed)
        official_count += len(official)
        second_union_count += len(set(proposed) | set(official))

        projected_summary = sg21._summary_from_indices(
            first_prediction, base_summary.confidence_margin
        )
        self_context = sg17.imagined_context_after(
            context, first_action, projected_summary
        )
        if self_context is None and proposed:
            self_context_failures += 1
        next_last_move = (
            first_action
            if first_action in sg19.INVERSE_MOVE
            else sg19._last_move(context)
        )
        for second_action in proposed:
            target = official_targets.get(second_action)
            if target is None or self_context is None:
                if len(errors) < 100:
                    errors.append(
                        {
                            "first_id": first["first_id"],
                            "second_action": second_action,
                            "error": (
                                "invalid_candidate"
                                if target is None
                                else "premature_self_stop"
                            ),
                        }
                    )
                continue
            second_started = time.perf_counter_ns()
            second_scores, second_mask, _second_model_elapsed = score_fn(
                self_context,
                first_mask,
                second_action,
                plans[seed],
                next_last_move,
            )
            second_prediction, _mask, _graph, projection = (
                sg21.project_graph_transition(
                    sg21._indices_from_scores(second_scores),
                    second_mask,
                    branch_graph,
                    second_action,
                    action_order,
                    branch_tag=(
                        f"crwm1:self:{seed}:{root_step}:"
                        f"{first_action}:{second_action}"
                    ),
                    plan=plans[seed],
                    phase=len(self_context),
                    enforce_plan_path_constraint=True,
                )
            )
            second_latencies.append(
                (time.perf_counter_ns() - second_started) / 1e6
            )
            second_evaluated += 1
            is_correct = second_prediction == target
            second_correct += int(is_correct)
            if not is_correct and len(errors) < 100:
                errors.append(
                    {
                        "first_id": first["first_id"],
                        "second_action": second_action,
                        "error": "transition_mismatch",
                        "prediction": second_prediction,
                        "target": target,
                        "projection": projection,
                    }
                )

        group_records.append(
            {
                "first_id": first["first_id"],
                "proposal_fingerprint": proposal_fingerprint,
                "proposed_count": len(proposed),
                "official_count": len(official),
                "true_positive": counts["true_positive"],
                "false_positive": counts["false_positive"],
                "false_negative": counts["false_negative"],
                "exact": counts["exact"],
                "first_projection_kind": first_projection["kind"],
            }
        )

    precision = (
        total_true_positive / proposed_count if proposed_count else 1.0
    )
    recall = total_true_positive / official_count if official_count else 1.0
    group_count = len(group_records)
    return {
        "information_flow_audit": {
            "candidate_source": "projected_first_next_affordance_mask",
            "proposal_materialized_before_evaluator_targets": True,
            "teacher_context_call_count": 0,
            "future_oracle_candidate_proposal_count": 0,
            "official_values_used_for_scoring_only": True,
        },
        "counts": {
            "first_groups": group_count,
            "proposed_candidates": proposed_count,
            "official_candidates": official_count,
            "true_positive": total_true_positive,
            "false_positive": total_false_positive,
            "false_negative": total_false_negative,
            "self_context_failures": self_context_failures,
            "second_evaluated": second_evaluated,
            "second_union": second_union_count,
        },
        "candidate_precision": precision,
        "candidate_recall": recall,
        "candidate_set_exact_accuracy": exact_candidate_sets / group_count,
        "invalid_transition_rate": (
            total_false_positive / proposed_count if proposed_count else 0.0
        ),
        "no_oracle_second_exact_accuracy": (
            second_correct / second_union_count
            if second_union_count
            else 1.0
        ),
        "second_correct": second_correct,
        "first_latency": sg16._timing_summary(first_latencies),
        "second_latency": sg16._timing_summary(second_latencies),
        "groups": group_records,
        "errors_first_100": errors,
    }


def decide(data_passed: bool, rollout: Mapping[str, Any]) -> Dict[str, Any]:
    flow = rollout["information_flow_audit"]
    information_flow = bool(
        flow["proposal_materialized_before_evaluator_targets"]
        and flow["teacher_context_call_count"] == 0
        and flow["future_oracle_candidate_proposal_count"] == 0
    )
    candidate_gate = bool(
        rollout["candidate_precision"] >= 0.95
        and rollout["candidate_recall"] >= 0.95
        and rollout["candidate_set_exact_accuracy"] >= 0.95
    )
    transition_gate = bool(
        rollout["no_oracle_second_exact_accuracy"] >= 0.95
        and rollout["invalid_transition_rate"] <= 0.05
    )
    passed = data_passed and information_flow and candidate_gate and transition_gate
    return {
        "data_provenance_gate": data_passed,
        "information_flow_gate": information_flow,
        "candidate_generation_gate": candidate_gate,
        "self_transition_gate": transition_gate,
        "verdict": (
            "PHASE1_GO_LONG_HORIZON_REQUIRED"
            if passed
            else "STOP_ORACLE_DEPENDENCE"
        ),
        "overall": "PASS" if passed else "FAIL",
        "next_route": (
            "crwm2_live_horizon_2_8_32_matched_baselines"
            if passed
            else "audit_candidate_generation_failure"
        ),
    }


def run_experiment(args: argparse.Namespace) -> Dict[str, Any]:
    device = torch.device("cpu")
    torch.set_num_threads(args.threads)
    reference, reference_digest = sg16._load_reference(
        args.reference.expanduser().resolve(),
        REFERENCE_SHA256,
        "E3-SG22R seventh-fresh constrained matched confirmation",
    )
    if reference["decision"]["overall"] != "PASS":
        raise ValueError("CRWM-1 requires passing SG22R reference")
    cache_path = args.cache.expanduser().resolve()
    cache_digest = file_sha256(cache_path).upper()
    if cache_digest != CACHE_SHA256:
        raise ValueError(
            f"cache SHA mismatch: expected {CACHE_SHA256}, got {cache_digest}"
        )
    cache = json.loads(cache_path.read_text(encoding="utf-8"))
    corpus_root = args.corpus_dir.expanduser().resolve()
    artifact_hashes = sg21r._artifact_hashes(corpus_root)
    artifacts_equal = artifact_hashes == reference["dataset"]["artifact_hashes"]
    manifest = {
        "mode": "offline_frozen_artifact_identity",
        "reference_manifest_verified": bool(
            reference["dataset"]["manifest"]["verified"]
        ),
        "cross_split_seed_disjoint": bool(
            reference["dataset"]["manifest"]["cross_split_seed_disjoint"]
        ),
        "reference_fingerprint_sha256": reference["dataset"]["manifest"][
            "fingerprint_sha256"
        ],
        "artifact_hashes_equal_reference": artifacts_equal,
        "game_binary_revalidation": "not_required_for_offline_frozen_cache",
    }
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
    exhaustive = cache["exhaustive"]
    tree, repair_audit = sg17.repair_persistent_room_semantics(
        cache["branch_tree"]
    )
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
    constraint_audit = sg22.audit_plan_path_constraint(
        exhaustive, plans, action_order
    )
    score_fn = _runtime_scorer(
        alphabet_index, unique, coefficients, device
    )
    rollout = evaluate_no_oracle_two_step(
        tree,
        exhaustive["test"]["records"],
        snapshots["test"],
        plans["test"],
        action_order=action_order,
        score_fn=score_fn,
    )
    tree_ok = bool(
        tree["game_count"] == 8
        and tree["root_count"] == 40
        and tree["first_branch_count"] == 160
        and tree["second_pair_count"] == 616
        and tree["all_live_factual_won"]
        and tree["all_counterfactuals_non_mutating"]
    )
    data_passed = bool(
        manifest["reference_manifest_verified"]
        and manifest["cross_split_seed_disjoint"]
        and artifacts_equal
        and data_audit["passed"]
        and tree_ok
        and repair_audit["changed_pair_count"] == 0
        and graph_audit["all_masks_match_exhaustive_cache"]
        and graph_audit["all_binding_steps_precede_snapshot_root"]
        and graph_audit["all_rooms_unique_and_present"]
        and graph_audit["no_edge_conflicts"]
        and constraint_audit["all_counts_match"]
        and constraint_audit["all_targets_match"]
        and tuple(action_order) == tuple(reference["dataset"]["action_order"])
        and vocabulary.fingerprint
        == reference["dataset"]["vocabulary_fingerprint"]
    )
    decision = decide(data_passed, rollout)
    return {
        "schema_version": 1,
        "experiment": EXPERIMENT,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "environment": _environment(device),
        "references": {
            "sg22r": {
                "path": str(args.reference.expanduser().resolve()),
                "sha256": reference_digest,
            },
            "cache": {"path": str(cache_path), "sha256": cache_digest},
        },
        "configuration": {
            "corpus_dir": str(corpus_root),
            "threads": args.threads,
            "candidate_precision_min": 0.95,
            "candidate_recall_min": 0.95,
            "candidate_set_exact_min": 0.95,
            "second_exact_min": 0.95,
            "invalid_transition_rate_max": 0.05,
        },
        "dataset": {
            "manifest": manifest,
            "artifact_hashes": artifact_hashes,
            "data_audit": data_audit,
            "tree_audit_passed": tree_ok,
            "tree_repair_audit": repair_audit,
            "graph_audit": graph_audit,
            "constraint_audit": constraint_audit,
            "objective_plan_audit": plan_audit,
            "action_order": action_order,
            "vocabulary_fingerprint": vocabulary.fingerprint,
        },
        "fit": base_fit,
        "rollout": rollout,
        "reference_metrics": {
            "oracle_candidate_role": reference["dataset"]["fresh_tree"][
                "future_candidate_set_role"
            ],
            "sg22r_self_second_exact": reference["snn"]["rollout"][
                "self_rollout_second"
            ]["exact_vector_accuracy"],
        },
        "decision": decision,
    }


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-dir", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--threads", type=int, default=4)
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
