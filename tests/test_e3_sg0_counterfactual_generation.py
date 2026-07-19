from pathlib import Path

import pytest
import torch

from experiments.e3_sg0_counterfactual_generation import (
    _example_tensors,
    audit_examples,
    build_counterfactual_examples,
    normalize_textworld_observation,
)
from vpsc.world_model.event_corpus import load_event_corpus


CORPUS_ROOT = Path("results/e2_world_model/textworld_l5")
EXPECTED_VOCABULARY_FINGERPRINT = (
    "dd3e51c6deb5b1aede57b71b9d9745f390a301ba4b7ccd3a66a237f066717364"
)


def test_sg0_normalization_removes_ui_prefix_and_preserves_world_state() -> None:
    source = """TextWorld banner
Goal line

-= Pantry =-
  A   quiet room.
> score: 0
Exits: north
"""
    assert normalize_textworld_observation(source) == (
        "-= Pantry =-\nA quiet room.\nExits: north"
    )


def test_sg0_task_vocabulary_is_deterministic_train_only_and_valid() -> None:
    corpus = load_event_corpus(CORPUS_ROOT)
    first, first_vocabulary = build_counterfactual_examples(CORPUS_ROOT, corpus)
    second, second_vocabulary = build_counterfactual_examples(CORPUS_ROOT, corpus)

    assert first == second
    assert first_vocabulary.tokens == second_vocabulary.tokens
    assert len(first_vocabulary) == 183
    assert first_vocabulary.fingerprint == EXPECTED_VOCABULARY_FINGERPRINT
    assert first_vocabulary.fingerprint != corpus.vocabulary.fingerprint
    assert sum(value.prompt_unknowns for value in first["train"]) == 0
    assert sum(value.target_unknowns for value in first["train"]) == 0

    audit = audit_examples(first, first_vocabulary)
    assert audit["passed"]
    assert {
        split: audit["splits"][split]["example_count"]
        for split in ("train", "valid", "test")
    } == {"train": 40, "valid": 10, "test": 10}
    assert audit["splits"]["valid"]["target_unknown_ratio"] == pytest.approx(
        21 / 273
    )
    assert audit["splits"]["test"]["target_unknown_ratio"] == pytest.approx(
        17 / 258
    )


def test_sg0_queries_align_every_target_token_causally() -> None:
    corpus = load_event_corpus(CORPUS_ROOT)
    examples, _ = build_counterfactual_examples(CORPUS_ROOT, corpus)
    example = examples["train"][0]
    input_ids, query_indices, targets = _example_tensors(
        example, device=torch.device("cpu")
    )

    expected_sequence = example.prompt_ids + example.target_ids
    assert input_ids.shape == (1, len(expected_sequence) - 1)
    assert input_ids[0].tolist() == list(expected_sequence[:-1])
    assert targets[0].tolist() == list(example.target_ids)
    assert query_indices[0].item() == len(example.prompt_ids) - 1
    assert query_indices[-1].item() == input_ids.shape[1] - 1
    assert len(query_indices) == len(example.target_ids)
    assert [expected_sequence[index + 1] for index in query_indices.tolist()] == list(
        example.target_ids
    )
