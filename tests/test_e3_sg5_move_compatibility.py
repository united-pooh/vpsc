from pathlib import Path

import torch

from experiments import e3_sg0_counterfactual_generation as sg0
from experiments.e3_sg4_move_pair_ranking import EXPECTED_COUNTS, EXPECTED_GROUPS
from experiments.e3_sg5_move_compatibility import (
    COMPATIBLE,
    INCOMPATIBLE,
    audit_compatibility_examples,
    build_compatibility_examples,
)
from vpsc.world_model.event_corpus import load_event_corpus


CORPUS_ROOT = Path("results/e3_scan/textworld_sg2_l5")
EXPECTED_VOCABULARY_FINGERPRINT = (
    "43dbe84bfb295e0168bf166102e9fd1f035d280d2c6b6fe202887732a311025c"
)


def test_sg5_compatibility_data_is_deterministic_balanced_and_valid() -> None:
    corpus = load_event_corpus(CORPUS_ROOT)
    first, first_vocabulary = build_compatibility_examples(CORPUS_ROOT, corpus)
    second, second_vocabulary = build_compatibility_examples(CORPUS_ROOT, corpus)
    assert first == second
    assert first_vocabulary.tokens == second_vocabulary.tokens
    assert len(first_vocabulary) == 300
    assert first_vocabulary.fingerprint == EXPECTED_VOCABULARY_FINGERPRINT

    audit = audit_compatibility_examples(
        first,
        first_vocabulary,
        expected_counts=EXPECTED_COUNTS,
        expected_groups=EXPECTED_GROUPS,
        max_prompt_tokens=448,
    )
    assert audit["passed"]
    assert audit["splits"]["test"]["label_counts"] == {
        COMPATIBLE: 24,
        INCOMPATIBLE: 24,
    }
    assert audit["splits"]["test"]["candidate_group_count"] == 24
    assert audit["splits"]["test"]["step_group_count"] == 12
    assert audit["splits"]["test"]["prompt_length"]["max"] == 303


def test_sg5_each_candidate_has_one_positive_and_one_negative_query() -> None:
    corpus = load_event_corpus(CORPUS_ROOT)
    examples, vocabulary = build_compatibility_examples(CORPUS_ROOT, corpus)
    groups: dict[str, list] = {}
    for example in examples["test"]:
        groups.setdefault(example.group_id, []).append(example)
        input_ids, query_indices, targets = sg0._example_tensors(
            example, device=torch.device("cpu")
        )
        assert input_ids.shape == (1, len(example.prompt_ids))
        assert query_indices.tolist() == [len(example.prompt_ids) - 1]
        assert targets.shape == (1, 1)
    for group in groups.values():
        assert {example.outcome_index for example in group} == {0, 1}
        assert {vocabulary.decode(example.target_ids)[0] for example in group} == {
            COMPATIBLE,
            INCOMPATIBLE,
        }
        assert len({example.outcome_text for example in group}) == 2
