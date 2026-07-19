from __future__ import annotations

from dataclasses import dataclass

import torch

from experiments.e3_sg25e_bucketed_batch_graph import (
    _masked_example_mean_loss,
    build_epoch_batches,
)


@dataclass(frozen=True)
class _Example:
    prompt_ids: tuple[int, ...]
    target_ids: tuple[int, ...]


@dataclass(frozen=True)
class _Vocabulary:
    pad_id: int = 0


def _example(input_length: int, target_length: int) -> _Example:
    prompt_length = input_length - target_length + 1
    return _Example(
        prompt_ids=tuple(range(1, prompt_length + 1)),
        target_ids=tuple(range(100, 100 + target_length)),
    )


def test_bucket_batches_keep_all_examples_and_mask_dummy_rows() -> None:
    examples = (
        _example(60, 6),
        _example(62, 6),
        _example(90, 39),
        _example(94, 41),
        _example(120, 60),
    )

    batches = build_epoch_batches(
        examples,
        _Vocabulary(),
        batch_size=2,
        epoch=0,
        seed=25_800_000,
        device=torch.device("cpu"),
    )

    seen = sorted(index for batch in batches for index in batch.example_indices)
    assert seen == list(range(len(examples)))
    assert sum(batch.real_example_count for batch in batches) == len(examples)
    for batch in batches:
        assert int(batch.example_mask.sum().item()) == batch.real_example_count
        assert int(batch.token_mask.sum().item()) == batch.target_token_count
        for row in range(batch.real_example_count):
            valid_count = int(batch.token_mask[row].sum().item())
            assert torch.all(batch.query_indices[row, :valid_count] >= 0)
            assert torch.all(batch.query_indices[row, valid_count:] == -1)


def test_loss_means_tokens_per_example_before_examples() -> None:
    logits = torch.tensor(
        (
            ((4.0, 0.0), (0.0, 4.0), (4.0, 0.0)),
            ((0.0, 4.0), (4.0, 0.0), (0.0, 4.0)),
        )
    )
    targets = torch.tensor(((0, 1, 0), (0, 0, 0)))
    token_mask = torch.tensor(((1.0, 1.0, 1.0), (1.0, 0.0, 0.0)))
    example_mask = torch.tensor((1.0, 1.0))

    loss = _masked_example_mean_loss(
        logits, targets, token_mask, example_mask
    )
    first = torch.nn.functional.cross_entropy(logits[0], targets[0])
    second = torch.nn.functional.cross_entropy(logits[1, :1], targets[1, :1])

    torch.testing.assert_close(loss, (first + second) / 2.0)
