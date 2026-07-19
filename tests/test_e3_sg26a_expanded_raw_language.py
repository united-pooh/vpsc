from copy import deepcopy
from types import SimpleNamespace

from experiments import e3_sg26a_expanded_raw_language as sg26a


def test_expanded_bucket_audit_freezes_all_320_examples():
    examples = []
    for count, input_length, target_length in (
        (129, 64, 27),
        (116, 65, 28),
        (72, 97, 56),
        (3, 129, 70),
    ):
        examples.extend(
            SimpleNamespace(
                input_length=input_length,
                target_ids=tuple(range(target_length)),
            )
            for _ in range(count)
        )
    audit = sg26a.expanded_bucket_audit(examples)
    assert audit["passed"]
    assert audit["counts"] == sg26a.EXPECTED_BUCKET_COUNTS
    assert audit["parallel_example_count"] == 317
    assert audit["serial_fallback_example_count"] == 3


def test_expanded_backend_restores_imported_modules():
    original_buckets = sg26a.sg25e.BUCKET_CAPACITIES
    original_logits = sg26a.sg25f._mode_logits
    with sg26a._expanded_backend():
        assert sg26a.sg25e.BUCKET_CAPACITIES == sg26a.BUCKET_CAPACITIES
        assert sg26a.sg25f._mode_logits is not original_logits
    assert sg26a.sg25e.BUCKET_CAPACITIES is original_buckets
    assert sg26a.sg25f._mode_logits is original_logits


def test_equivalence_prefix_covers_every_bucket_before_schedule():
    keys = ((16, 128, 71), (16, 64, 27), (16, 160, 71), (16, 96, 55))
    batches = tuple(SimpleNamespace(key=key) for key in keys)
    prefix = sg26a._coverage_prefix(batches, updates=8)
    assert tuple(batch.key for batch in prefix[:4]) == tuple(sorted(keys))


def _primary_fixture():
    def record(eps, target_tps, p50, nll, edit):
        return {
            "capture": {
                "shape_count": 4,
                "allocated_delta_bytes": 20,
                "peak_additional_allocated_bytes": 25,
            },
            "equivalence": {"passed": True},
            "benchmark": {
                "effective_examples_per_second": eps,
                "effective_target_tokens_per_second": target_tps,
                "per_real_example_timing": {"p50_ms": p50},
            },
            "profiler": {"host_launch_and_copy_api_count": 9},
            "bucket_profiler": {
                "bucket": {"host_launch_and_copy_api_count": 9}
            },
            "quality": {
                "all_losses_finite": True,
                "update_count_passed": True,
                "pre_teacher": {"test": {"nll": 4.0}},
                "post_teacher": {"test": {"nll": nll}},
                "generation": {
                    "edit_similarity": edit,
                    "paired_action_sensitivity": 1.0,
                },
            },
        }

    return {
        "snn_parallel": record(20_500.0, 900_000.0, 0.045, 2.05, 0.75),
        "lstm": record(19_000.0, 850_000.0, 0.050, 2.00, 0.74),
        "transformer": record(12_000.0, 600_000.0, 0.070, 2.20, 0.72),
    }


def test_formal_decision_requires_task_and_cross_architecture_quality():
    primary = _primary_fixture()
    canonical = {
        "capture": {
            "allocated_delta_bytes": 20,
            "peak_additional_allocated_bytes": 25,
        }
    }
    decision = sg26a._decision(
        {"passed": True, "task_edit_threshold": 0.70},
        {"passed": True},
        {"passed": True},
        primary,
        canonical,
        quick=False,
    )
    assert decision["overall"] == "PASS"
    assert decision["next_route"] == "multimodal_rollout_closed_loop"

    snn_failure = deepcopy(primary)
    snn_failure["snn_parallel"]["quality"]["generation"][
        "edit_similarity"
    ] = 0.68
    decision = sg26a._decision(
        {"passed": True, "task_edit_threshold": 0.70},
        {"passed": True},
        {"passed": True},
        snn_failure,
        canonical,
        quick=False,
    )
    assert decision["task_validity_gate"] == "FAIL"
    assert decision["next_route"] == "snn_representation_learning"
