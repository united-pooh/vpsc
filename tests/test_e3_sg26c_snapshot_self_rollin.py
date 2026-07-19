from types import SimpleNamespace

import torch

from experiments import e3_sg26c_snapshot_self_rollin as sg26c


def _batch():
    return sg26c.sg25e.DeviceBatch(
        input_ids=torch.tensor([[1, 2, 3, 4, 0]], dtype=torch.long),
        query_indices=torch.tensor([[1, 2, 3]], dtype=torch.long),
        gather_indices=torch.tensor([[1, 2, 3]], dtype=torch.long),
        targets=torch.tensor([[3, 4, 9]], dtype=torch.long),
        token_mask=torch.ones(1, 3),
        example_mask=torch.ones(1),
        real_example_count=1,
        input_token_count=4,
        target_token_count=3,
        example_indices=(0,),
    )


def test_full_rollin_replaces_only_target_history_inputs():
    batch = _batch()
    examples = [SimpleNamespace(prompt_ids=(1, 2), target_ids=(3, 4, 9))]
    audit = sg26c.apply_rollin_corruption(
        (batch,),
        examples,
        {0: (7, 8, 9)},
        rate=1.0,
        stage_start_epoch=50,
        eos_id=9,
        device=torch.device("cpu"),
    )
    assert audit["passed"]
    assert audit["selected_history_token_count"] == 2
    assert batch.input_ids.tolist() == [[1, 2, 7, 8, 0]]


def test_zero_rollin_needs_no_predictions_and_changes_nothing():
    batch = _batch()
    original = batch.input_ids.clone()
    examples = [SimpleNamespace(prompt_ids=(1, 2), target_ids=(3, 4, 9))]
    audit = sg26c.apply_rollin_corruption(
        (batch,),
        examples,
        {},
        rate=0.0,
        stage_start_epoch=50,
        eos_id=9,
        device=torch.device("cpu"),
    )
    assert audit["passed"]
    assert audit["eligible_history_token_count"] == 2
    assert torch.equal(batch.input_ids, original)


def test_position_hash_is_deterministic_and_rate_bounded():
    first = [
        sg26c._selected(
            0.25, epoch=50 + index // 100, example_index=index % 100, history_index=index
        )
        for index in range(10_000)
    ]
    second = [
        sg26c._selected(
            0.25, epoch=50 + index // 100, example_index=index % 100, history_index=index
        )
        for index in range(10_000)
    ]
    assert first == second
    assert abs(sum(first) / len(first) - 0.25) < 0.02


def _candidate(rate, *, nll, edit, room=0.0):
    return {
        "rate": rate,
        "quality": {
            "all_losses_finite": True,
            "update_count_passed": True,
            "corruption": {"passed": True},
            "post_teacher": {"valid": {"nll": nll}},
            "generation": {
                "edit_similarity": edit,
                "room_accuracy": room,
                "paired_action_sensitivity": 1.0,
            },
        },
    }


def test_rate_selection_excludes_nll_regression_then_uses_valid_edit():
    sweep = {
        "rate_0_0": _candidate(0.0, nll=0.65, edit=0.62),
        "rate_0_25": _candidate(0.25, nll=0.70, edit=0.70, room=0.1),
        "rate_0_5": _candidate(0.5, nll=0.76, edit=0.90, room=0.2),
    }
    selection = sg26c.select_rate(
        sweep, {"task_edit_threshold": 0.68}
    )
    assert not selection["eligible"]["rate_0_5"]
    assert selection["selected_label"] == "rate_0_25"
    assert selection["passed"]
