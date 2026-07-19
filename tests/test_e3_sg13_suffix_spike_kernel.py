from pathlib import Path

import torch

from experiments.e3_sg13_suffix_spike_kernel import (
    KERNEL_SPECS,
    PRIMARY_SPEC,
    _padded_history_key,
    block_schur_kernel_fit,
    build_game_folds,
    extract_kernel_records,
    suffix_spike_kernel,
)
from experiments.e3_sg12_spike_delay_rls import build_action_alphabet
from experiments.e3_sg10_multichannel_delta import build_multichannel_examples
from vpsc.world_model.event_corpus import load_event_corpus


CORPUS_ROOT = Path("results/e3_scan/textworld_sg2_l5")


def _examples():
    corpus = load_event_corpus(CORPUS_ROOT)
    return build_multichannel_examples(CORPUS_ROOT, corpus)[0]


def test_sg13_nested_suffix_kernel_has_expected_backoff_and_is_psd() -> None:
    keys = torch.tensor(
        (
            (8, 8, 0, 1),
            (8, 8, 0, 1),
            (8, 8, 2, 1),
            (3, 4, 2, 1),
            (3, 4, 2, 5),
        ),
        dtype=torch.long,
    )
    phases = torch.tensor((1, 1, 1, 4, 4), dtype=torch.long)
    kernel = suffix_spike_kernel(
        keys, keys, phases, phases, PRIMARY_SPEC
    )
    assert kernel[0, 1] == 5.0  # candidate + 3 suffix depths + phase
    assert kernel[0, 2] == 2.0  # candidate plus same phase; last action differs
    assert kernel[2, 3] == 2.0  # candidate plus last1; phase differs
    assert kernel[3, 4] == 1.0  # phase only; candidate differs
    for spec in KERNEL_SPECS:
        value = suffix_spike_kernel(keys, keys, phases, phases, spec)
        eigenvalues = torch.linalg.eigvalsh(value)
        assert float(eigenvalues.min().item()) >= -1e-10


def test_sg13_padding_and_game_folds_preserve_causal_order_and_isolation() -> None:
    alphabet_index = {"<event_a>": 0, "<event_b>": 1, "<event_c>": 2}
    key = _padded_history_key(
        ("a", "b"),
        "c",
        alphabet_index=alphabet_index,
        pad_index=3,
    )
    assert key == (3, 0, 1, 2)

    game_seeds = tuple(seed for seed in range(32) for _ in range(15))
    folds = build_game_folds(game_seeds, fold_count=4)
    assert tuple(len(fold) for fold in folds) == (120, 120, 120, 120)
    fold_games = [
        {game_seeds[index] for index in fold} for fold in folds
    ]
    assert all(len(games) == 8 for games in fold_games)
    assert not any(
        fold_games[left] & fold_games[right]
        for left in range(4)
        for right in range(left + 1, 4)
    )


def test_sg13_real_records_use_train_only_keys_and_four_balanced_folds() -> None:
    examples = _examples()
    alphabet = build_action_alphabet(examples)
    alphabet_index = {token: index for index, token in enumerate(alphabet)}
    records = extract_kernel_records(
        examples["train"],
        alphabet_index=alphabet_index,
        device=torch.device("cpu"),
    )
    assert records["keys"].shape == (480, 4)
    assert records["phases"].tolist()[:3] == [0, 0, 0]
    folds = build_game_folds(records["game_seeds"], fold_count=4)
    assert tuple(len(fold) for fold in folds) == (120, 120, 120, 120)


def test_sg13_block_schur_online_fit_matches_batch_kernel_solution() -> None:
    keys = torch.tensor(
        (
            (8, 8, 0, 1),
            (8, 8, 2, 1),
            (8, 0, 2, 1),
            (0, 2, 3, 1),
            (2, 3, 4, 1),
            (3, 4, 5, 2),
        ),
        dtype=torch.long,
    )
    phases = torch.tensor((1, 1, 2, 3, 4, 4), dtype=torch.long)
    torch.manual_seed(13)
    targets = torch.randn(6, 3, dtype=torch.float64)
    records = {
        "train": {
            "keys": keys,
            "phases": phases,
            "target_code": targets,
        }
    }
    ridge_lambda = 0.25
    kernel = suffix_spike_kernel(
        keys, keys, phases, phases, PRIMARY_SPEC
    )
    batch = torch.linalg.solve(
        kernel + ridge_lambda * torch.eye(6, dtype=torch.float64), targets
    )
    online, timing = block_schur_kernel_fit(
        records,
        PRIMARY_SPEC,
        ((4, 1), (5, 0), (3, 2)),
        ridge_lambda=ridge_lambda,
        device=torch.device("cpu"),
    )
    torch.testing.assert_close(online, batch, atol=1e-10, rtol=1e-10)
    assert timing["block_updates"] == 3
    assert timing["examples_seen"] == 6
