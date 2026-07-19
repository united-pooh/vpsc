import torch
import torch.nn.functional as F

from experiments import e3_sg26b_length_mass_objective as sg26b


def test_length_mass_alpha_extremes_match_frozen_objectives():
    torch.manual_seed(7)
    logits = torch.randn(3, 5, 11)
    targets = torch.randint(0, 11, (3, 5))
    token_mask = torch.tensor(
        [[1, 1, 0, 0, 0], [1, 1, 1, 1, 1], [0, 0, 0, 0, 0]],
        dtype=torch.float32,
    )
    example_mask = torch.tensor([1, 1, 0], dtype=torch.float32)
    legacy = sg26b.sg25e._masked_example_mean_loss(
        logits, targets, token_mask, example_mask
    )
    alpha_one = sg26b.length_mass_loss(
        logits, targets, token_mask, example_mask, alpha=1.0
    )
    token_losses = F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        targets.reshape(-1),
        reduction="none",
    ).reshape_as(targets)
    token_mean = (token_losses * token_mask).sum() / token_mask.sum()
    alpha_zero = sg26b.length_mass_loss(
        logits, targets, token_mask, example_mask, alpha=0.0
    )
    assert torch.equal(alpha_one, legacy)
    assert torch.equal(alpha_zero, token_mean)


def test_patched_loss_restores_original():
    original = sg26b.sg25e._masked_example_mean_loss
    with sg26b._patched_loss(0.5):
        assert sg26b.sg25e._masked_example_mean_loss is not original
    assert sg26b.sg25e._masked_example_mean_loss is original


def _candidate(alpha, *, nll, edit, room=0.0, sensitivity=1.0):
    return {
        "alpha": alpha,
        "quality": {
            "all_losses_finite": True,
            "update_count_passed": True,
            "post_teacher": {"valid": {"nll": nll}},
            "generation": {
                "edit_similarity": edit,
                "room_accuracy": room,
                "paired_action_sensitivity": sensitivity,
            },
        },
    }


def test_selection_uses_valid_only_constraints_before_edit_score():
    sweep = {
        "alpha_1_0": _candidate(1.0, nll=0.7, edit=0.66),
        "alpha_0_5": _candidate(0.5, nll=0.75, edit=0.72, room=0.1),
        "alpha_0_0": _candidate(0.0, nll=0.81, edit=0.90, room=0.2),
    }
    selection = sg26b.select_alpha(
        sweep, {"task_edit_threshold": 0.70}
    )
    assert not selection["eligible"]["alpha_0_0"]
    assert selection["selected_label"] == "alpha_0_5"
    assert selection["passed"]
