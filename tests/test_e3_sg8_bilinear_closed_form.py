from pathlib import Path

import torch

from experiments import e3_sg0_counterfactual_generation as sg0
from experiments.e3_sg6_move_delta import build_move_delta_examples
from experiments.e3_sg7_paired_binary_batch import build_paired_batch_schedule
from experiments.e3_sg8_bilinear_closed_form import (
    RIDGE_LAMBDAS,
    _bilinear_batch_tensors,
    action_query_indices,
    build_bilinear_models,
    evaluate_bilinear,
    fit_closed_form_ridge,
    train_bilinear,
)
from vpsc.world_model.cores import count_parameters
from vpsc.world_model.event_corpus import load_event_corpus


CORPUS_ROOT = Path("results/e3_scan/textworld_sg2_l5")


def test_sg8_queries_action_identity_positions_not_final_label_position() -> None:
    corpus = load_event_corpus(CORPUS_ROOT)
    examples, vocabulary = build_move_delta_examples(CORPUS_ROOT, corpus)
    first = examples["train"][0]
    assert action_query_indices(first, vocabulary) == (5, 11)
    input_ids, query_indices, targets = _bilinear_batch_tensors(
        examples["train"], tuple(range(32)), vocabulary, device=torch.device("cpu")
    )
    assert input_ids.shape == (32, 17)
    assert query_indices.tolist() == [5, 11]
    assert targets.shape == (32,)


def test_sg8_all_cores_share_the_same_bilinear_head_and_parameter_budget() -> None:
    corpus = load_event_corpus(CORPUS_ROOT)
    _, vocabulary = build_move_delta_examples(CORPUS_ROOT, corpus)
    models = build_bilinear_models(
        10_000_000,
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


def test_sg8_reverse_adjoint_update_and_closed_form_fit_are_finite() -> None:
    corpus = load_event_corpus(CORPUS_ROOT)
    examples, vocabulary = build_move_delta_examples(CORPUS_ROOT, corpus)
    models = build_bilinear_models(
        10_000_000,
        vocabulary,
        d_model=32,
        state_dim=31,
        num_heads=4,
        device=torch.device("cpu"),
    )
    reservoir = models["snn_ra0"].language_model
    ridge = fit_closed_form_ridge(
        reservoir,
        examples,
        vocabulary,
        device=torch.device("cpu"),
        lambdas=RIDGE_LAMBDAS,
    )
    assert ridge["feature_dimension"] == 1089
    assert ridge["readout_parameter_count"] == 1089
    assert ridge["selected_lambda"] in RIDGE_LAMBDAS
    assert 0.0 <= ridge["test"]["accuracy"] <= 1.0
    assert ridge["training_wall_seconds"] > 0.0

    schedule = build_paired_batch_schedule(
        examples["train"], epochs=1, batch_groups=16, seed=10_001_000
    )
    training = train_bilinear(
        "snn_ra0",
        models["snn_ra0"],
        examples["train"],
        vocabulary,
        schedule[:1],
        epochs=1,
        batches_per_epoch=1,
        device=torch.device("cpu"),
    )
    evaluation = evaluate_bilinear(
        models["snn_ra0"],
        examples["test"],
        vocabulary,
        device=torch.device("cpu"),
        include_records=False,
    )
    assert training["updates"] == 1
    assert training["example_exposures"] == 32
    assert training["loss_first"] >= 0.0
    assert 0.0 <= evaluation["accuracy"] <= 1.0
