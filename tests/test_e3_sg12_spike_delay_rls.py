from pathlib import Path

import torch

from experiments.e3_sg12_spike_delay_rls import (
    PRIMARY_ORDER,
    audit_delay_data,
    block_rls,
    build_action_alphabet,
    delay_feature_tensor,
    delay_initial,
    delay_state_after,
    delay_step,
    expected_feature_dimension,
    extract_delay_features,
)
from experiments.e3_sg10_multichannel_delta import build_multichannel_examples
from vpsc.world_model.event_corpus import load_event_corpus


CORPUS_ROOT = Path("results/e3_scan/textworld_sg2_l5")


def _examples():
    corpus = load_event_corpus(CORPUS_ROOT)
    return build_multichannel_examples(CORPUS_ROOT, corpus)[0]


def test_sg12_delay_line_is_an_exact_binary_shift_register() -> None:
    state = delay_initial(1, 3, 4, device=torch.device("cpu"))
    eye = torch.eye(4, dtype=torch.bool)
    for index in (2, 0, 3, 1):
        state = delay_step(state, eye[index : index + 1])
    expected = torch.stack((eye[1], eye[3], eye[0]), dim=0)[None]
    assert state.dtype == torch.bool
    torch.testing.assert_close(state, expected)
    assert int(state.sum().item()) == 3


def test_sg12_sparse_outer_feature_has_frozen_dimension_and_activity() -> None:
    state = delay_initial(2, 3, 8, device=torch.device("cpu"))
    eye = torch.eye(8, dtype=torch.bool)
    state = delay_step(state, eye[[1, 2]])
    state = delay_step(state, eye[[3, 4]])
    candidate = eye[[5, 6]]
    features = delay_feature_tensor(state, candidate, dtype=torch.float64)
    assert features.shape == (2, expected_feature_dimension(3, 8))
    assert features.shape[1] == 225
    # bias + two context spikes + candidate + two context/candidate bindings
    assert torch.equal((features != 0).sum(dim=1), torch.tensor((6, 6)))


def test_sg12_real_data_audit_proves_order_three_is_first_unambiguous() -> None:
    examples = _examples()
    alphabet = build_action_alphabet(examples)
    assert len(alphabet) == 8
    audit = audit_delay_data(examples, alphabet)
    assert audit["passed"]
    assert audit["conditional_history"]["2"]["train_ambiguous_key_count"] == 12
    assert audit["conditional_history"]["3"]["train_ambiguous_key_count"] == 0
    assert audit["conditional_history"]["3"]["splits"]["test"] == {
        "covered_examples": 55,
        "total_examples": 60,
        "covered_majority_accuracy": 1.0,
    }
    alphabet_index = {token: index for index, token in enumerate(alphabet)}
    extracted = extract_delay_features(
        examples["test"],
        order=PRIMARY_ORDER,
        alphabet_index=alphabet_index,
        device=torch.device("cpu"),
    )
    assert extracted["features"].shape == (60, 225)
    assert extracted["active_feature_count"]["max"] == 8


def test_sg12_block_rls_matches_batch_ridge_on_fixed_blocks() -> None:
    torch.manual_seed(12)
    x = torch.randn(24, 9, dtype=torch.float64)
    y = torch.randn(24, 4, dtype=torch.float64)
    ridge_lambda = 0.5
    identity = torch.eye(9, dtype=torch.float64)
    batch = torch.linalg.solve(
        x.T @ x + ridge_lambda * identity, x.T @ y
    )
    schedule = tuple(tuple(range(start, start + 6)) for start in range(0, 24, 6))
    online, covariance, timing = block_rls(
        x,
        y,
        schedule,
        ridge_lambda=ridge_lambda,
        device=torch.device("cpu"),
    )
    torch.testing.assert_close(online, batch, atol=1e-10, rtol=1e-10)
    torch.testing.assert_close(
        covariance,
        torch.linalg.inv(x.T @ x + ridge_lambda * identity),
        atol=1e-10,
        rtol=1e-10,
    )
    assert timing["block_updates"] == 4
    assert timing["examples_seen"] == 24
