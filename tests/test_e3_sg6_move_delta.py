from pathlib import Path

import torch

from experiments import e3_sg0_counterfactual_generation as sg0
from experiments.e3_sg1_history_generation import build_history_models
from experiments.e3_sg6_move_delta import (
    EXPECTED_COUNTS,
    EXPECTED_STEP_GROUPS,
    NOVEL_ROOM,
    PREVIOUS_ROOM,
    audit_move_delta_examples,
    build_move_delta_examples,
)
from vpsc.world_model.cores import count_parameters
from vpsc.world_model.event_corpus import load_event_corpus


CORPUS_ROOT = Path("results/e3_scan/textworld_sg2_l5")
EXPECTED_VOCABULARY_FINGERPRINT = (
    "4c660544808a1553f1d54383c3be69a28c2377d3602f09bc9698a091fd5b5b0e"
)


def test_sg6_move_delta_data_is_deterministic_balanced_and_valid() -> None:
    corpus = load_event_corpus(CORPUS_ROOT)
    first, first_vocabulary = build_move_delta_examples(CORPUS_ROOT, corpus)
    second, second_vocabulary = build_move_delta_examples(CORPUS_ROOT, corpus)
    assert first == second
    assert first_vocabulary.tokens == second_vocabulary.tokens
    assert len(first_vocabulary) == 18
    assert first_vocabulary.fingerprint == EXPECTED_VOCABULARY_FINGERPRINT

    audit = audit_move_delta_examples(
        first,
        first_vocabulary,
        expected_counts=EXPECTED_COUNTS,
        expected_step_groups=EXPECTED_STEP_GROUPS,
        max_prompt_tokens=32,
    )
    assert audit["passed"]
    assert audit["splits"]["test"]["label_counts"] == {
        NOVEL_ROOM: 12,
        PREVIOUS_ROOM: 12,
    }
    assert audit["splits"]["test"]["step_group_count"] == 12
    assert audit["splits"]["test"]["prompt_length"]["max"] == 17
    assert audit["splits"]["test"]["relationship_violation_count"] == 0


def test_sg6_labels_come_from_real_outcome_history_membership() -> None:
    corpus = load_event_corpus(CORPUS_ROOT)
    examples, vocabulary = build_move_delta_examples(CORPUS_ROOT, corpus)
    groups: dict[str, list] = {}
    for example in examples["test"]:
        groups.setdefault(example.step_group_id, []).append(example)
        input_ids, query_indices, targets = sg0._example_tensors(
            example, device=torch.device("cpu")
        )
        assert input_ids.shape == (1, len(example.prompt_ids))
        assert query_indices.tolist() == [len(example.prompt_ids) - 1]
        assert targets.shape == (1, 1)

    for group in groups.values():
        assert len(group) == 2
        by_source = {example.source: example for example in group}
        factual = by_source["factual"]
        counterfactual = by_source["counterfactual"]
        assert factual.prior_match_lags == ()
        assert counterfactual.prior_match_lags == (1,)
        assert vocabulary.decode(factual.target_ids) == (NOVEL_ROOM,)
        assert vocabulary.decode(counterfactual.target_ids) == (PREVIOUS_ROOM,)
        assert factual.previous_action == counterfactual.previous_action
        assert factual.candidate_action != counterfactual.candidate_action
        assert factual.outcome_text != counterfactual.outcome_text


def test_sg6_compact_models_remain_parameter_matched() -> None:
    corpus = load_event_corpus(CORPUS_ROOT)
    _, vocabulary = build_move_delta_examples(CORPUS_ROOT, corpus)
    models = build_history_models(
        9_800_000,
        vocabulary,
        d_model=32,
        state_dim=31,
        num_heads=4,
        device=torch.device("cpu"),
    )
    counts = {name: count_parameters(model) for name, model in models.items()}
    spread = (max(counts.values()) - min(counts.values())) / sg0._mean(
        counts.values()
    )
    assert spread < 0.03
    bptt_state = models["snn_bptt"].state_dict()
    for name in ("snn_at1", "snn_ra0"):
        for key, value in models[name].state_dict().items():
            torch.testing.assert_close(value, bptt_state[key], atol=0.0, rtol=0.0)
