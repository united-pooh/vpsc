"""Audit whether TextWorld counterfactual targets are identifiable from history."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any, Dict, Optional, Sequence, Tuple

import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.e2_f0_fusion_benchmark import _environment  # noqa: E402
from experiments import e3_tw0_sparse_event_lm as tw0  # noqa: E402
from experiments.e3_sg0_counterfactual_generation import (  # noqa: E402
    _action_type,
    normalize_textworld_observation,
)
from vpsc.world_model.wikitext import SPLITS, file_sha256  # noqa: E402


@dataclass(frozen=True)
class IdentifiabilityRecord:
    split: str
    episode_index: int
    game_seed: int
    step_index: int
    counterfactual_index: int
    action: str
    action_type: str
    current_room: Optional[str]
    target_room: Optional[str]
    target_room_visible_in_current: bool
    target_room_in_prior_history: bool
    target_surface_equals_current: bool
    target_surface_in_prior_history: bool
    target_surface_history_lag: Optional[int]

    @property
    def example_id(self) -> str:
        return (
            f"{self.split}:{self.game_seed}:{self.step_index}:"
            f"{self.counterfactual_index}"
        )


def extract_room_name(normalized_observation: str) -> Optional[str]:
    """Return the canonical room header name, if the response contains one."""

    for line in normalized_observation.splitlines():
        if line.startswith("-=") and line.endswith("=-"):
            room = line[2:-2].strip()
            return room or None
    return None


def _last_surface_lag(
    target: str, prior_observations: Sequence[str], step_index: int
) -> Optional[int]:
    for prior_index in range(len(prior_observations) - 1, -1, -1):
        if prior_observations[prior_index] == target:
            return step_index - prior_index
    return None


def build_identifiability_records(
    corpus_root: Path,
) -> Dict[str, Tuple[IdentifiabilityRecord, ...]]:
    records_by_split: Dict[str, Tuple[IdentifiabilityRecord, ...]] = {}
    for split in SPLITS:
        path = corpus_root / split / "episodes.jsonl"
        records = []
        for episode_index, line in enumerate(
            path.read_text(encoding="utf-8").splitlines()
        ):
            episode = json.loads(line)
            if episode.get("split") != split:
                raise ValueError(f"episode split mismatch: {path}")
            steps = episode["steps"]
            normalized_observations = tuple(
                normalize_textworld_observation(step["observation"])
                for step in steps
            )
            for step_index, step in enumerate(steps):
                current = normalized_observations[step_index]
                current_room = extract_room_name(current)
                prior = normalized_observations[:step_index]
                prior_rooms = {
                    room
                    for room in (extract_room_name(value) for value in prior)
                    if room is not None
                }
                for counterfactual_index, counterfactual in enumerate(
                    step["counterfactuals"]
                ):
                    target = normalize_textworld_observation(
                        counterfactual["next_obs"]
                    )
                    target_room = extract_room_name(target)
                    lag = _last_surface_lag(target, prior, step_index)
                    action = str(counterfactual["action"])
                    records.append(
                        IdentifiabilityRecord(
                            split=split,
                            episode_index=episode_index,
                            game_seed=int(episode["seed"]),
                            step_index=step_index,
                            counterfactual_index=counterfactual_index,
                            action=action,
                            action_type=_action_type(action),
                            current_room=current_room,
                            target_room=target_room,
                            target_room_visible_in_current=(
                                target_room is not None
                                and target_room.casefold() in current.casefold()
                            ),
                            target_room_in_prior_history=(
                                target_room is not None and target_room in prior_rooms
                            ),
                            target_surface_equals_current=target == current,
                            target_surface_in_prior_history=lag is not None,
                            target_surface_history_lag=lag,
                        )
                    )
        records_by_split[split] = tuple(records)
    return records_by_split


def _ratio(numerator: int, denominator: int) -> Optional[float]:
    return numerator / denominator if denominator else None


def _summarize(records: Sequence[IdentifiabilityRecord]) -> Dict[str, Any]:
    move = tuple(record for record in records if record.action_type == "move")
    look = tuple(record for record in records if record.action_type == "look")
    move_room_current = sum(record.target_room_visible_in_current for record in move)
    move_room_history = sum(record.target_room_in_prior_history for record in move)
    move_surface_history = sum(record.target_surface_in_prior_history for record in move)
    look_surface_current = sum(record.target_surface_equals_current for record in look)
    lags = Counter(
        record.target_surface_history_lag
        for record in move
        if record.target_surface_history_lag is not None
    )
    return {
        "example_count": len(records),
        "action_types": dict(sorted(Counter(record.action_type for record in records).items())),
        "move_count": len(move),
        "move_target_room_visible_in_current_count": move_room_current,
        "move_target_room_visible_in_current_ratio": _ratio(
            move_room_current, len(move)
        ),
        "move_target_room_in_prior_history_count": move_room_history,
        "move_target_room_in_prior_history_ratio": _ratio(
            move_room_history, len(move)
        ),
        "move_target_surface_in_prior_history_count": move_surface_history,
        "move_target_surface_in_prior_history_ratio": _ratio(
            move_surface_history, len(move)
        ),
        "move_target_surface_history_lag": {
            str(key): value for key, value in sorted(lags.items())
        },
        "look_count": len(look),
        "look_target_surface_equals_current_count": look_surface_current,
        "look_target_surface_equals_current_ratio": _ratio(
            look_surface_current, len(look)
        ),
    }


def run_audit(corpus_root: Path) -> Dict[str, Any]:
    root = corpus_root.expanduser().resolve()
    manifest = tw0._manifest_provenance(root)
    records = build_identifiability_records(root)
    summaries = {
        split: _summarize(records[split]) for split in SPLITS
    }
    held_out_move = tuple(
        record
        for split in ("valid", "test")
        for record in records[split]
        if record.action_type == "move"
    )
    visible_current = sum(
        record.target_room_visible_in_current for record in held_out_move
    )
    surface_in_history = sum(
        record.target_surface_in_prior_history for record in held_out_move
    )
    single_observation_identifiable = (
        bool(held_out_move) and visible_current == len(held_out_move)
    )
    history_route_feasible = (
        len(held_out_move) >= 8
        and surface_in_history / len(held_out_move) >= 0.80
    )
    source_files = {
        split: {
            "path": str((root / split / "episodes.jsonl").resolve()),
            "sha256": file_sha256(root / split / "episodes.jsonl"),
            "size_bytes": (root / split / "episodes.jsonl").stat().st_size,
        }
        for split in SPLITS
    }
    return {
        "schema_version": 1,
        "experiment": "E3-SG1 TextWorld target identifiability audit",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "environment": _environment(torch.device("cpu")),
        "dataset": {
            "synthetic": False,
            "corpus_root": str(root),
            "manifest_provenance": manifest,
            "episodes": source_files,
        },
        "splits": summaries,
        "held_out_move": {
            "count": len(held_out_move),
            "target_room_visible_in_current_count": visible_current,
            "target_room_visible_in_current_ratio": _ratio(
                visible_current, len(held_out_move)
            ),
            "target_surface_in_prior_history_count": surface_in_history,
            "target_surface_in_prior_history_ratio": _ratio(
                surface_in_history, len(held_out_move)
            ),
        },
        "decision": {
            "sg0_single_observation_identifiability": (
                "PASS" if single_observation_identifiable else "FAIL"
            ),
            "sg1_history_conditioned_route": (
                "PASS" if history_route_feasible else "FAIL"
            ),
            "next_experiment": (
                "history_conditioned_generation"
                if history_route_feasible
                else "collect_exploration_or_use_delta_ranking"
            ),
        },
        "records": [
            {"example_id": record.example_id, **asdict(record)}
            for split in SPLITS
            for record in records[split]
        ],
    }


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-dir", type=Path, default=tw0.DEFAULT_CORPUS_DIR)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/e3_scan/e3_sg1_history_identifiability.json"),
    )
    return parser.parse_args(argv)


def main() -> None:
    args = _parse_args()
    result = run_audit(args.corpus_dir)
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result["decision"], indent=2, sort_keys=True))
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
