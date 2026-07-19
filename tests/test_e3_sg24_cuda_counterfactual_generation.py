from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "experiments/e3_sg24_cuda_counterfactual_generation.py"
SPEC = importlib.util.spec_from_file_location("e3_sg24_cuda_counterfactual_generation", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
sg24 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(sg24)


def test_frozen_input_audit_matches_canonical_sg0_inputs() -> None:
    audit = sg24.frozen_input_audit(ROOT / "results/e2_world_model/textworld_l5")

    assert audit["passed"]
    assert audit["base_runner"]["passed"]
    assert all(record["passed"] for record in audit["corpus"].values())


def test_formal_decision_requires_backend_inputs_and_parent_pass() -> None:
    base = {"overall": "PASS", "data_gate": "PASS"}

    passed = sg24._augment_decision(
        base, frozen_inputs_passed=True, backend_passed=True, quick=False
    )
    failed = sg24._augment_decision(
        base, frozen_inputs_passed=True, backend_passed=False, quick=False
    )

    assert passed["overall"] == "PASS"
    assert passed["sg0_architecture_gates_overall"] == "PASS"
    assert failed["overall"] == "FAIL"
    assert failed["backend_gate"] == "FAIL"


def test_quick_decision_preserves_smoke_only_when_hardware_is_valid() -> None:
    decision = sg24._augment_decision(
        {"overall": "SMOKE"},
        frozen_inputs_passed=True,
        backend_passed=True,
        quick=True,
    )

    assert decision["overall"] == "SMOKE"
