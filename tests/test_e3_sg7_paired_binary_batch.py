from collections import Counter
from pathlib import Path

import torch

from experiments.e3_sg1_history_generation import build_history_models
from experiments.e3_sg6_move_delta import LABELS, build_move_delta_examples
from experiments.e3_sg7_paired_binary_batch import (
    _paired_batch_tensors,
    build_paired_batch_schedule,
    evaluate_binary_teacher,
    train_paired_binary,
)
from vpsc.world_model.event_corpus import load_event_corpus


CORPUS_ROOT = Path("results/e3_scan/textworld_sg2_l5")


def test_sg7_schedule_keeps_pairs_balanced_and_covers_each_epoch() -> None:
    corpus = load_event_corpus(CORPUS_ROOT)
    examples, vocabulary = build_move_delta_examples(CORPUS_ROOT, corpus)
    schedule = build_paired_batch_schedule(
        examples["train"], epochs=2, batch_groups=16, seed=9_901_000
    )
    repeated = build_paired_batch_schedule(
        examples["train"], epochs=2, batch_groups=16, seed=9_901_000
    )
    assert schedule == repeated
    assert len(schedule) == 12
    label_ids = {vocabulary.token_id(label) for label in LABELS}
    for batch in schedule:
        assert len(batch) == 32
        groups = Counter(examples["train"][index].step_group_id for index in batch)
        assert set(groups.values()) == {2}
        targets = Counter(examples["train"][index].target_ids[0] for index in batch)
        assert set(targets) == label_ids
        assert set(targets.values()) == {16}
    for epoch in range(2):
        epoch_indices = [
            index
            for batch in schedule[epoch * 6 : (epoch + 1) * 6]
            for index in batch
        ]
        assert Counter(epoch_indices) == Counter(range(192))


def test_sg7_paired_batch_tensors_have_one_shared_query() -> None:
    corpus = load_event_corpus(CORPUS_ROOT)
    examples, vocabulary = build_move_delta_examples(CORPUS_ROOT, corpus)
    schedule = build_paired_batch_schedule(
        examples["train"], epochs=1, batch_groups=16, seed=9_901_000
    )
    input_ids, query_indices, targets = _paired_batch_tensors(
        examples["train"], schedule[0], vocabulary, device=torch.device("cpu")
    )
    assert input_ids.shape == (32, 17)
    assert query_indices.tolist() == [16]
    assert targets.shape == (32,)
    assert Counter(targets.tolist()) == Counter({0: 16, 1: 16})


def test_sg7_batched_reverse_adjoint_update_is_finite() -> None:
    corpus = load_event_corpus(CORPUS_ROOT)
    examples, vocabulary = build_move_delta_examples(CORPUS_ROOT, corpus)
    models = build_history_models(
        9_900_000,
        vocabulary,
        d_model=32,
        state_dim=31,
        num_heads=4,
        device=torch.device("cpu"),
    )
    schedule = build_paired_batch_schedule(
        examples["train"], epochs=1, batch_groups=16, seed=9_901_000
    )
    before = evaluate_binary_teacher(
        models["snn_ra0"], examples["test"], vocabulary, device=torch.device("cpu")
    )
    training = train_paired_binary(
        "snn_ra0",
        models["snn_ra0"],
        examples["train"],
        vocabulary,
        schedule[:1],
        epochs=1,
        batches_per_epoch=1,
        device=torch.device("cpu"),
    )
    after = evaluate_binary_teacher(
        models["snn_ra0"], examples["test"], vocabulary, device=torch.device("cpu")
    )
    assert before["example_count"] == after["example_count"] == 24
    assert training["updates"] == 1
    assert training["example_exposures"] == 32
    assert training["batch_examples"] == 32
    assert training["loss_first"] >= 0.0
    assert training["examples_per_second_total"] > 0.0
