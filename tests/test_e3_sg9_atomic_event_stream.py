from pathlib import Path

import torch

from experiments.e3_sg6_move_delta import build_move_delta_examples
from experiments.e3_sg9_atomic_event_stream import (
    audit_atomic_event_examples,
    build_atomic_event_examples,
    build_bilinear_models,
    evaluate_cached_stream,
    fit_event_closed_form_ridge,
)
from experiments.e3_sg8_bilinear_closed_form import RIDGE_LAMBDAS
from vpsc.world_model.event_corpus import load_event_corpus


CORPUS_ROOT = Path("results/e3_scan/textworld_sg2_l5")
EXPECTED_VOCABULARY_FINGERPRINT = (
    "57231174773d2471e6cd666c69e651aad7d384757b387e4f0e449161d182562a"
)


def _examples():
    corpus = load_event_corpus(CORPUS_ROOT)
    source, source_vocabulary = build_move_delta_examples(CORPUS_ROOT, corpus)
    return build_atomic_event_examples(source, source_vocabulary)


def test_sg9_atomic_events_are_deterministic_real_and_balanced() -> None:
    first, first_vocabulary = _examples()
    second, second_vocabulary = _examples()
    assert first == second
    assert first_vocabulary.tokens == second_vocabulary.tokens
    assert len(first_vocabulary) == 10
    assert first_vocabulary.fingerprint == EXPECTED_VOCABULARY_FINGERPRINT
    audit = audit_atomic_event_examples(
        first,
        first_vocabulary,
        expected_counts={"train": 192, "valid": 24, "test": 24},
        expected_step_groups={"train": 96, "valid": 12, "test": 12},
    )
    assert audit["passed"]
    assert audit["splits"]["test"]["prompt_length"]["max"] == 2
    assert audit["splits"]["test"]["unique_event_token_count"] == 4
    assert audit["splits"]["test"]["event_token_collision_count"] == 0


def test_sg9_cached_candidate_matches_full_causal_event_pair() -> None:
    examples, vocabulary = _examples()
    models = build_bilinear_models(
        10_100_000,
        vocabulary,
        d_model=32,
        state_dim=31,
        num_heads=4,
        device=torch.device("cpu"),
    )
    for name, model in models.items():
        generic = evaluate_cached_stream(
            model,
            examples["test"],
            vocabulary,
            device=torch.device("cpu"),
            use_cached_decay=False,
        )
        assert generic["max_full_logit_abs_difference"] <= 1e-5
        if name.startswith("snn_"):
            cached = evaluate_cached_stream(
                model,
                examples["test"],
                vocabulary,
                device=torch.device("cpu"),
                use_cached_decay=True,
            )
            assert cached["max_full_logit_abs_difference"] <= 1e-5


def test_sg9_atomic_event_frozen_reservoir_has_closed_form_readout() -> None:
    examples, vocabulary = _examples()
    models = build_bilinear_models(
        10_100_000,
        vocabulary,
        d_model=32,
        state_dim=31,
        num_heads=4,
        device=torch.device("cpu"),
    )
    result = fit_event_closed_form_ridge(
        models["snn_ra0"].language_model,
        examples,
        vocabulary,
        device=torch.device("cpu"),
        lambdas=RIDGE_LAMBDAS,
    )
    assert result["feature_dimension"] == 1089
    assert result["selected_lambda"] in RIDGE_LAMBDAS
    assert result["valid"]["accuracy"] == 1.0
    assert result["test"]["accuracy"] == 1.0
