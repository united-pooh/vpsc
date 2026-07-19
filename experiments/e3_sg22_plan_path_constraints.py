"""SG22 shared topological plan-path mask constraints on sixth-fresh data."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
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
from experiments import e3_tw0_sparse_event_lm as tw0  # noqa: E402
from experiments.e3_sg12_spike_delay_rls import (  # noqa: E402
    build_action_alphabet,
)
from vpsc.world_model.event_corpus import load_event_corpus  # noqa: E402
from vpsc.world_model.wikitext import SPLITS  # noqa: E402


DEFAULT_OUTPUT = Path("results/e3_scan/e3_sg22_plan_path_constraints.json")
DEFAULT_SG21R_REFERENCE = Path(
    "results/e3_scan/e3_sg21r_sixth_fresh_matched_ann.json"
)
SG21R_REFERENCE_SHA256 = (
    "00D20C2F64AE51E4C4A9DDCCDF687C263D2559B8AF9822CE00CBEAD3BAC976F0"
)
SG21R_EXPERIMENT = "E3-SG21R sixth-fresh matched ANN graph confirmation"


def audit_plan_path_constraint(
    exhaustive: Mapping[str, Any],
    plans: Mapping[str, Mapping[int, Sequence[str]]],
    action_order: Sequence[str],
) -> Dict[str, Any]:
    splits = {}
    for split in SPLITS:
        checked = 0
        errors = []
        for record in exhaustive[split]["records"]:
            phase = len(record["context_actions"])
            candidate = str(record["candidate_action"])
            mask = sg21.plan_path_constraint_mask(
                candidate,
                plans[split][int(record["game_seed"])],
                phase,
                action_order,
            )
            if mask is None:
                continue
            checked += 1
            target = tuple(int(value) for value in record["next_mask"])
            if mask != target and len(errors) < 100:
                errors.append(
                    {
                        "record_id": record["record_id"],
                        "predicted_mask": mask,
                        "target_mask": target,
                    }
                )
        splits[split] = {
            "plan_move_count": checked,
            "constraint_error_count": len(errors),
            "errors_first_100": tuple(errors),
        }
    return {
        "source": "public objective plan plus compass inverse algebra",
        "expected_plan_move_counts": {
            "train": 128,
            "valid": 32,
            "test": 32,
        },
        "all_counts_match": all(
            splits[split]["plan_move_count"]
            == {"train": 128, "valid": 32, "test": 32}[split]
            for split in SPLITS
        ),
        "all_targets_match": all(
            splits[split]["constraint_error_count"] == 0 for split in SPLITS
        ),
        "splits": splits,
    }


def run_experiment(args: argparse.Namespace) -> Dict[str, Any]:
    device = torch.device("cpu")
    torch.set_num_threads(args.threads)
    corpus_root = args.corpus_dir.expanduser().resolve()
    reference, reference_digest = sg16._load_reference(
        args.sg21r_reference.expanduser().resolve(),
        SG21R_REFERENCE_SHA256,
        SG21R_EXPERIMENT,
    )
    if (
        reference["decision"]["snn_quality_gate"] != "PASS"
        or reference["decision"]["matched_quality_gate"] != "FAIL"
    ):
        raise ValueError("SG22 expects SG21R SNN PASS / matched-quality FAIL")
    manifest = tw0._manifest_provenance(
        corpus_root, expected_seeds_by_split=sg21r.EXPECTED_SEEDS
    )
    artifact_hashes = sg21r._artifact_hashes(corpus_root)
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
    cache, cache_digest, cache_reused, collection_wall = (
        sg21r.collect_or_load_fresh_cache(
            args, corpus_root, corpus, action_order, artifact_hashes
        )
    )
    exhaustive = cache["exhaustive"]
    tree = cache["branch_tree"]
    repaired_tree, repair_audit = sg17.repair_persistent_room_semantics(tree)
    if repair_audit["changed_pair_count"] != 0:
        raise AssertionError("SG22 fresh tree unexpectedly needs repair")
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
    constraint_audit = audit_plan_path_constraint(
        exhaustive, plans, action_order
    )
    if not (
        constraint_audit["all_counts_match"]
        and constraint_audit["all_targets_match"]
    ):
        raise AssertionError("SG22 plan-path constraint audit failed")

    snn_graph_metrics = {
        split: sg21.evaluate_split_with_graph(
            split,
            tensors[split],
            exhaustive[split]["records"],
            snapshots[split],
            unique,
            coefficients,
            action_order,
            plans=plans[split],
            enforce_plan_path_constraint=True,
        )
        for split in SPLITS
    }
    snn_scorer = sg21r.SnnEndToEndScorer(
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
        enforce_plan_path_constraint=True,
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
        "graph_split_metrics": snn_graph_metrics,
        "rollout": snn_rollout,
        "training_wall_seconds": snn_training_wall,
        "base_model_logical_bytes": base_model_bytes,
        "combined_logical_bytes": base_model_bytes + max_graph_bytes,
    }

    feature_vocabulary = sg21r.build_feature_vocabulary(
        action_order, action_order
    )
    matched_tokens = {}
    encoding_seconds = {}
    for split in SPLITS:
        matched_tokens[split], encoding_seconds[split] = (
            sg21r.tensorize_matched_features(
                exhaustive[split]["records"],
                plans[split],
                feature_vocabulary,
                action_order,
                device=device,
            )
        )
    ann_results = []
    for seed in sg21r.TRAINING_SEEDS:
        for name in sg21r.MODEL_NAMES:
            model, training = sg21r.train_matched_model(
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
                    raw_metrics[split] = sg21r._raw_metrics(
                        logits, tensors[split], action_order
                    )
                    graph_metrics[split] = sg21r.evaluate_logits_with_graph(
                        logits,
                        tensors[split],
                        exhaustive[split]["records"],
                        snapshots[split],
                        action_order,
                        split,
                        plans=plans[split],
                        enforce_plan_path_constraint=True,
                    )
            scorer = sg21r.MatchedAnnScorer(
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
                enforce_plan_path_constraint=True,
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

    data_passed = bool(
        reference["decision"]["data_no_leak_gate"] == "PASS"
        and data_audit["passed"]
        and plan_audit["all_plans_equal_walkthrough_for_audit"]
        and cache_reused
        and graph_audit["all_masks_match_exhaustive_cache"]
        and graph_audit["all_binding_steps_precede_snapshot_root"]
        and graph_audit["all_rooms_unique_and_present"]
        and graph_audit["no_edge_conflicts"]
        and constraint_audit["all_targets_match"]
    )
    decision = sg21r._decision(
        data_passed,
        snn_result,
        ann_results,
        base_model_bytes + max_graph_bytes,
        max_graph_bytes,
    )
    decision["independent_confirmation_required"] = True
    if decision["overall"] == "PASS":
        decision["next_route"] = "sg22r_seventh_fresh_confirmation"
    return {
        "schema_version": 1,
        "experiment": "E3-SG22 shared plan-path topological mask constraints",
        "formal": not args.quick,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "environment": _environment(device),
        "research_hypothesis": {
            "epistemic_status": "mechanism on observed sixth games",
            "statement": (
                "Plan-path topology exactly determines the unknown forward "
                "room affordance mask and should be projected, not learned."
            ),
            "what_if": (
                "What if the fastest exact training step is removing "
                "deducible bits from learning altogether?"
            ),
        },
        "references": {
            "sg21r_matched_quality_negative": {
                "path": str(args.sg21r_reference.expanduser().resolve()),
                "sha256": reference_digest,
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
            "shared_constraint": (
                "unknown plan-current move -> inverse plus plan-next mask"
            ),
            "constraint_parameter_count": 0,
            "ann_protocol_equal_sg21r": True,
        },
        "dataset": {
            "manifest": manifest,
            "data_audit": data_audit,
            "artifact_hashes": artifact_hashes,
            "vocabulary_fingerprint": vocabulary.fingerprint,
            "action_alphabet": alphabet,
            "action_order": action_order,
            "objective_plan_audit": plan_audit,
            "tree_repair_audit": repair_audit,
            "graph_audit": graph_audit,
            "constraint_audit": constraint_audit,
        },
        "snn": snn_result,
        "matched_ann": ann_results,
        "decision": decision,
    }


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-dir", type=Path, default=sg21r.DEFAULT_CORPUS)
    parser.add_argument("--cache", type=Path, default=sg21r.DEFAULT_CACHE)
    parser.add_argument(
        "--sg21r-reference", type=Path, default=DEFAULT_SG21R_REFERENCE
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
