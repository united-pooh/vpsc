from pathlib import Path

import torch

from experiments import e3_sg0_counterfactual_generation as sg0
from experiments.e3_sg1_history_generation import build_history_models
from experiments.e3_sg4_move_pair_ranking import (
    EXPECTED_COUNTS,
    EXPECTED_GROUPS,
    LABEL_A,
    LABEL_B,
    audit_ranking_examples,
    build_ranking_examples,
)
from vpsc.world_model.cores import count_parameters
from vpsc.world_model.event_corpus import load_event_corpus


CORPUS_ROOT = Path("results/e3_scan/textworld_sg2_l5")
EXPECTED_VOCABULARY_FINGERPRINT = (
    "dcf0d7070a97b93e33d39d6c792a97117f73b60e4fce9c790ee422def041e846"
)


def test_sg4_hard_move_pairs_are_deterministic_balanced_and_valid() -> None:
    corpus = load_event_corpus(CORPUS_ROOT)
    first, first_vocabulary = build_ranking_examples(CORPUS_ROOT, corpus)
    second, second_vocabulary = build_ranking_examples(CORPUS_ROOT, corpus)
    assert first == second
    assert first_vocabulary.tokens == second_vocabulary.tokens
    assert len(first_vocabulary) == 301
    assert first_vocabulary.fingerprint == EXPECTED_VOCABULARY_FINGERPRINT

    audit = audit_ranking_examples(
        first,
        first_vocabulary,
        expected_counts=EXPECTED_COUNTS,
        expected_groups=EXPECTED_GROUPS,
        max_prompt_tokens=512,
    )
    assert audit["passed"]
    assert audit["splits"]["test"]["label_counts"] == {
        LABEL_A: 24,
        LABEL_B: 24,
    }
    assert audit["splits"]["test"]["semantic_group_count"] == 24
    assert audit["splits"]["test"]["prompt_length"]["max"] == 358


def test_sg4_swaps_labels_and_keeps_single_causal_query() -> None:
    corpus = load_event_corpus(CORPUS_ROOT)
    examples, vocabulary = build_ranking_examples(CORPUS_ROOT, corpus)
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
        assert {example.option_order for example in group} == {(0, 1), (1, 0)}
        assert {vocabulary.decode(example.target_ids)[0] for example in group} == {
            LABEL_A,
            LABEL_B,
        }


def test_dynamic_d64_models_remain_parameter_matched() -> None:
    corpus = load_event_corpus(CORPUS_ROOT)
    _, vocabulary = build_ranking_examples(CORPUS_ROOT, corpus)
    models = build_history_models(
        9_600_000,
        vocabulary,
        d_model=64,
        state_dim=63,
        num_heads=4,
        device=torch.device("cpu"),
    )
    counts = {name: count_parameters(model) for name, model in models.items()}
    spread = (max(counts.values()) - min(counts.values())) / sg0._mean(
        counts.values()
    )
    assert spread < 0.02
    bptt_state = models["snn_bptt"].state_dict()
    for name in ("snn_at1", "snn_ra0"):
        for key, value in models[name].state_dict().items():
            torch.testing.assert_close(value, bptt_state[key], atol=0.0, rtol=0.0)
