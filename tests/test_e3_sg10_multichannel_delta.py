from collections import Counter
from pathlib import Path

import torch

from experiments import e3_sg0_counterfactual_generation as sg0
from experiments.e3_sg10_multichannel_delta import (
    DONE_LABELS,
    EXPECTED_COUNTS,
    EXPECTED_GROUPS,
    REWARD_LABELS,
    audit_multichannel_examples,
    build_class_weights,
    build_length_stratified_schedule,
    build_multichannel_examples,
    build_multichannel_models,
    evaluate_cached_multichannel,
    fit_multichannel_ridge,
)
from experiments.e3_sg8_bilinear_closed_form import RIDGE_LAMBDAS
from vpsc.world_model.cores import count_parameters
from vpsc.world_model.event_corpus import load_event_corpus


CORPUS_ROOT = Path("results/e3_scan/textworld_sg2_l5")
EXPECTED_VOCABULARY_FINGERPRINT = (
    "9d31551deae8300719634f1fc584b6ec508348c74b94efd3c456eb2096f1f749"
)


def _examples():
    corpus = load_event_corpus(CORPUS_ROOT)
    return build_multichannel_examples(CORPUS_ROOT, corpus)


def test_sg10_multichannel_data_is_real_deterministic_and_unambiguous() -> None:
    first, first_vocabulary = _examples()
    second, second_vocabulary = _examples()
    assert first == second
    assert first_vocabulary.tokens == second_vocabulary.tokens
    assert len(first_vocabulary) == 13
    assert first_vocabulary.fingerprint == EXPECTED_VOCABULARY_FINGERPRINT
    audit = audit_multichannel_examples(
        first,
        first_vocabulary,
        expected_counts=EXPECTED_COUNTS,
        expected_groups=EXPECTED_GROUPS,
    )
    assert audit["passed"]
    assert audit["splits"]["test"]["exact_vector_majority_accuracy"] == 0.2
    assert audit["splits"]["test"]["ambiguous_input_count"] == 0
    assert audit["heldout_full_input_coverage"]["test"] == {
        "covered": 52,
        "total": 60,
    }
    assert audit["splits"]["test"]["channel_label_counts"]["reward"] == {
        REWARD_LABELS[1]: 4,
        REWARD_LABELS[0]: 56,
    }
    assert audit["splits"]["test"]["channel_label_counts"]["done"] == {
        DONE_LABELS[0]: 56,
        DONE_LABELS[1]: 4,
    }


def test_sg10_schedule_is_length_homogeneous_balanced_and_complete() -> None:
    examples, _vocabulary = _examples()
    schedule = build_length_stratified_schedule(
        examples["train"], epochs=2, batch_groups=16, seed=10_201_000
    )
    assert len(schedule) == 20
    for batch in schedule:
        assert len(batch) == 48
        assert len({len(examples["train"][index].prompt_ids) for index in batch}) == 1
        groups = Counter(examples["train"][index].step_group_id for index in batch)
        assert set(groups.values()) == {3}
    for epoch in range(2):
        indices = [
            index
            for batch in schedule[epoch * 10 : (epoch + 1) * 10]
            for index in batch
        ]
        assert Counter(indices) == Counter(range(480))
    weights = build_class_weights(
        examples["train"], device=torch.device("cpu")
    )
    torch.testing.assert_close(
        weights["reward"], torch.tensor([480 / 896, 7.5])
    )
    torch.testing.assert_close(weights["done"], weights["reward"])


def test_sg10_models_ridge_and_cached_stream_keep_fair_contracts() -> None:
    examples, vocabulary = _examples()
    models = build_multichannel_models(
        10_200_000,
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
    reference = models["snn_bptt"].relation_head.state_dict()
    for model in models.values():
        for key, value in model.relation_head.state_dict().items():
            torch.testing.assert_close(value, reference[key], atol=0.0, rtol=0.0)

    ridge = fit_multichannel_ridge(
        models["snn_ra0"].language_model,
        examples,
        device=torch.device("cpu"),
        lambdas=RIDGE_LAMBDAS,
    )
    assert ridge["feature_dimension"] == 1089
    assert ridge["output_dimension"] == 11
    assert ridge["selected_lambda"] in RIDGE_LAMBDAS
    assert 0.0 <= ridge["test"]["exact_vector_accuracy"] <= 1.0

    cached = evaluate_cached_multichannel(
        models["snn_ra0"],
        examples["test"],
        device=torch.device("cpu"),
        use_cached_decay=True,
        timing_repeats=1,
        timing_warmup_repeats=0,
    )
    assert cached["candidate_timing_sample_count"] == 60
    assert cached["max_full_logit_abs_difference"] <= 1e-5
