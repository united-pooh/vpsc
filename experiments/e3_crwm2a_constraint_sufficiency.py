"""CRWM-2A live audit of whether public objectives already solve TextWorld.

The policy parses only the live public objective and executes that plan without
consulting admissible actions for selection.  Admissible actions and stored
walkthroughs are materialized only after each proposal for evaluation.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import statistics
import sys
import time
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments import e3_sg19_plan_edge_spikes as sg19  # noqa: E402
from vpsc.world_model.textworld import open_textworld  # noqa: E402
from vpsc.world_model.wikitext import file_sha256  # noqa: E402


DEFAULT_CORPUS = Path("results/e3_scan/textworld_sg22r_l5")
DEFAULT_GAMES = Path("data/e3_scan/textworld/games_sg22r_l5")
DEFAULT_REFERENCE = Path(
    "results/e3_scan/e3_sg22r_seventh_fresh_confirmation.json"
)
DEFAULT_CRWM1 = Path("results/e3_scan/e3_crwm1_no_oracle_candidates.json")
DEFAULT_OUTPUT = Path(
    "results/e3_scan/e3_crwm2a_constraint_sufficiency.json"
)
REFERENCE_SHA256 = (
    "1A75839740A7913E555FBEBD5EB462AA4C50D5324709B11F507A9FB607B7DB92"
)
CRWM1_SHA256 = (
    "64CA1BDDAAC177E9AEA51831EA586B14A73B2FB34807726D37BDA132415A56DE"
)
EXPERIMENT = "E3-CRWM2A public-objective constraint sufficiency"


def _load_frozen(path: Path, expected_sha256: str) -> Dict[str, Any]:
    digest = file_sha256(path).upper()
    if digest != expected_sha256:
        raise ValueError(
            f"frozen artifact SHA mismatch for {path}: "
            f"expected {expected_sha256}, got {digest}"
        )
    return json.loads(path.read_text(encoding="utf-8"))


def objective_only_action(plan: Sequence[str], step: int) -> Optional[str]:
    """Return an action without receiving any environment candidate set."""

    if step < 0:
        raise ValueError("step must be non-negative")
    return str(plan[step]) if step < len(plan) else None


def _percentile(values: Sequence[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(value) for value in values)
    position = q * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def decide(
    data_identity: bool, horizon_metrics: Mapping[str, Mapping[str, Any]]
) -> Dict[str, Any]:
    if not data_identity:
        return {
            "data_identity_gate": False,
            "constraint_sufficiency_gate": None,
            "residual_value_task_identifiable": None,
            "crwm0_success_improvement_feasible": None,
            "overall": "FAIL",
            "verdict": "STOP_DATA_IDENTITY_FAILURE",
        }
    horizon_8 = horizon_metrics["8"]
    horizon_32 = horizon_metrics["32"]
    constraint_sufficient = bool(
        horizon_8["win_rate"] == 1.0
        and horizon_32["win_rate"] == 1.0
        and horizon_8["invalid_action_rate"] == 0.0
        and horizon_32["invalid_action_rate"] == 0.0
        and horizon_8["plan_walkthrough_exact_rate"] == 1.0
        and horizon_32["plan_walkthrough_exact_rate"] == 1.0
    )
    task_identifiable = not constraint_sufficient
    return {
        "data_identity_gate": data_identity,
        "constraint_sufficiency_gate": constraint_sufficient,
        "residual_value_task_identifiable": task_identifiable,
        "crwm0_success_improvement_feasible": task_identifiable,
        "overall": "PASS",
        "verdict": (
            "STOP_TEXTWORLD_TASK_CEILING_PIVOT_ENVIRONMENT"
            if constraint_sufficient
            else "PROCEED_CRWM2_MATCHED_MATRIX"
        ),
    }


def _episode_records(corpus_root: Path) -> Tuple[Dict[str, Any], ...]:
    return tuple(
        json.loads(line)
        for line in (corpus_root / "test" / "episodes.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    )


def _expected_games(reference: Mapping[str, Any]) -> Dict[int, Dict[str, Any]]:
    games = reference["dataset"]["manifest"]["splits"]["test"]["games"]
    return {int(game["seed"]): dict(game) for game in games}


def _game_path(games_root: Path, expected: Mapping[str, Any]) -> Path:
    return games_root / Path(str(expected["path"])).name


def run_game(
    game_path: Path,
    episode: Mapping[str, Any],
    *,
    horizon: int,
) -> Dict[str, Any]:
    adapter = open_textworld(game_path, extras=())
    trajectory = []
    selection_latencies = []
    invalid_count = 0
    won = False
    done = False
    try:
        adapter.reset()
        public_objective = str(adapter.objective)
        plan = sg19.parse_objective_plan(public_objective)
        plan_fingerprint = hashlib.sha256(
            json.dumps(plan, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        stored_walkthrough = tuple(str(a) for a in episode["walkthrough"])
        for step in range(horizon):
            started = time.perf_counter_ns()
            proposed_action = objective_only_action(plan, step)
            selection_latencies.append(
                (time.perf_counter_ns() - started) / 1e6
            )
            if proposed_action is None:
                break

            # Frozen information barrier: only after proposal do evaluator
            # values become available for validity and outcome scoring.
            evaluator_actions = tuple(sorted(set(adapter.admissible_actions)))
            valid = proposed_action in evaluator_actions
            invalid_count += int(not valid)
            transition = adapter.step(proposed_action)
            won = bool(transition.info.get("won", False))
            done = bool(transition.done)
            trajectory.append(
                {
                    "step": step,
                    "proposed_action": proposed_action,
                    "proposal_valid": valid,
                    "reward": float(transition.reward),
                    "done": done,
                    "won": won,
                }
            )
            if done:
                break
    finally:
        adapter.close()
    return {
        "seed": int(episode["seed"]),
        "horizon": horizon,
        "public_objective_sha256": hashlib.sha256(
            public_objective.encode("utf-8")
        ).hexdigest(),
        "parsed_plan": plan,
        "parsed_plan_sha256": plan_fingerprint,
        "stored_walkthrough_materialized_after_run": True,
        "plan_walkthrough_exact": plan == stored_walkthrough,
        "candidate_oracle_selection_calls": 0,
        "action_count": len(trajectory),
        "invalid_action_count": invalid_count,
        "won": won,
        "done": done,
        "selection_latencies_ms": selection_latencies,
        "trajectory": trajectory,
    }


def _aggregate(records: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    actions = sum(int(record["action_count"]) for record in records)
    invalid = sum(int(record["invalid_action_count"]) for record in records)
    latencies = [
        float(value)
        for record in records
        for value in record["selection_latencies_ms"]
    ]
    return {
        "game_count": len(records),
        "win_count": sum(bool(record["won"]) for record in records),
        "win_rate": sum(bool(record["won"]) for record in records)
        / len(records),
        "done_rate": sum(bool(record["done"]) for record in records)
        / len(records),
        "mean_action_count": statistics.mean(
            int(record["action_count"]) for record in records
        ),
        "invalid_action_count": invalid,
        "invalid_action_rate": invalid / actions if actions else 0.0,
        "plan_walkthrough_exact_rate": sum(
            bool(record["plan_walkthrough_exact"]) for record in records
        )
        / len(records),
        "candidate_oracle_selection_calls": sum(
            int(record["candidate_oracle_selection_calls"])
            for record in records
        ),
        "selection_latency_p50_ms": _percentile(latencies, 0.50),
        "selection_latency_p95_ms": _percentile(latencies, 0.95),
    }


def run_experiment(args: argparse.Namespace) -> Dict[str, Any]:
    reference_path = args.reference.expanduser().resolve()
    crwm1_path = args.crwm1.expanduser().resolve()
    reference = _load_frozen(reference_path, REFERENCE_SHA256)
    crwm1 = _load_frozen(crwm1_path, CRWM1_SHA256)
    if reference["decision"]["overall"] != "PASS":
        raise ValueError("CRWM-2A requires passing SG22R reference")
    if crwm1["decision"]["verdict"] != "PHASE1_GO_LONG_HORIZON_REQUIRED":
        raise ValueError("CRWM-2A requires passing CRWM-1 reference")

    corpus_root = args.corpus_dir.expanduser().resolve()
    games_root = args.games_dir.expanduser().resolve()
    episodes = _episode_records(corpus_root)
    expected_games = _expected_games(reference)
    identity_records = []
    game_paths = {}
    for episode in episodes:
        seed = int(episode["seed"])
        expected = expected_games[seed]
        game_path = _game_path(games_root, expected)
        exists = game_path.is_file()
        digest = file_sha256(game_path).upper() if exists else None
        size = game_path.stat().st_size if exists else None
        passed = bool(
            exists
            and digest == str(expected["sha256"]).upper()
            and size == int(expected["size_bytes"])
        )
        identity_records.append(
            {
                "seed": seed,
                "path": str(game_path),
                "exists": exists,
                "sha256": digest,
                "expected_sha256": str(expected["sha256"]).upper(),
                "size_bytes": size,
                "expected_size_bytes": int(expected["size_bytes"]),
                "passed": passed,
            }
        )
        game_paths[seed] = game_path
    data_identity = all(record["passed"] for record in identity_records)
    by_horizon = {}
    raw_records = {}
    if data_identity:
        for horizon in args.horizons:
            records = tuple(
                run_game(
                    game_paths[int(episode["seed"])],
                    episode,
                    horizon=horizon,
                )
                for episode in episodes
            )
            raw_records[str(horizon)] = records
            by_horizon[str(horizon)] = _aggregate(records)
    decision = decide(data_identity, by_horizon)
    return {
        "schema_version": 1,
        "experiment": EXPERIMENT,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "references": {
            "sg22r": {"path": str(reference_path), "sha256": REFERENCE_SHA256},
            "crwm1": {"path": str(crwm1_path), "sha256": CRWM1_SHA256},
        },
        "configuration": {
            "corpus_dir": str(corpus_root),
            "games_dir": str(games_root),
            "horizons": args.horizons,
            "policy_input": "live public objective text only",
            "candidate_oracle_allowed_for_selection": False,
            "constraint_sufficient_threshold": {
                "horizon_8_win_rate": 1.0,
                "horizon_32_win_rate": 1.0,
                "invalid_action_rate": 0.0,
                "plan_walkthrough_exact_rate": 1.0,
            },
        },
        "data_identity": {
            "all_passed": data_identity,
            "games": identity_records,
        },
        "horizon_metrics": by_horizon,
        "runs": raw_records,
        "decision": decision,
    }


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-dir", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--games-dir", type=Path, default=DEFAULT_GAMES)
    parser.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--crwm1", type=Path, default=DEFAULT_CRWM1)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--horizons", nargs="+", type=int, default=[2, 8, 32])
    args = parser.parse_args(argv)
    if sorted(set(args.horizons)) != [2, 8, 32]:
        parser.error("--horizons must contain exactly 2 8 32")
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
