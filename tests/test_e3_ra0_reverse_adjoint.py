from pathlib import Path

import torch

from experiments.e3_ra0_reverse_adjoint import (
    build_textworld_models,
    run_equivalence,
)
from vpsc.world_model.cores import E3GatedTraceScanCore
from vpsc.world_model.event_corpus import load_event_corpus


CORPUS_ROOT = Path("results/e2_world_model/textworld_l5")


def test_ra0_formal_equivalence_matrix_passes() -> None:
    result = run_equivalence(torch.device("cpu"))
    assert result["passed"]
    assert len(result["cases"]) == 4
    formal = result["cases"][-1]
    assert formal["time"] == 512
    assert formal["query_count"] == 16
    assert formal["input_gradient"]
    assert formal["eligibility_backward_mode"] == "reverse_adjoint"


def test_ra0_textworld_model_uses_shared_initialisation() -> None:
    corpus = load_event_corpus(CORPUS_ROOT)
    models = build_textworld_models(
        9_398_000, corpus.vocabulary, device=torch.device("cpu")
    )
    assert tuple(models) == (
        "snn_bptt",
        "snn_at1",
        "snn_ra0",
        "lstm",
        "transformer",
    )
    assert isinstance(models["snn_ra0"].core, E3GatedTraceScanCore)
    assert models["snn_ra0"].core.eligibility_backward_mode == "reverse_adjoint"
    bptt_state = models["snn_bptt"].state_dict()
    for name in ("snn_at1", "snn_ra0"):
        candidate_state = models[name].state_dict()
        assert candidate_state.keys() == bptt_state.keys()
        for key, value in candidate_state.items():
            torch.testing.assert_close(value, bptt_state[key], atol=0.0, rtol=0.0)
