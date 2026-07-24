"""TDD tests for CCPA Phase 0 diagnostics."""
import torch
from experiments.ccpa import diag_common


def test_build_layer_fixed_point_shape():
    layer = diag_common.build_layer(n=8, rho=0.7, seed=0)
    x = torch.randn(4, 8)
    m = layer(x)
    assert m.shape == (4, 8)
    assert bool((m.abs() <= 1.0 + 1e-5).all())


def test_beta_sweep_decomposes_F():
    layer = diag_common.build_layer(n=8, rho=0.7, seed=0)
    x = torch.randn(4, 8)
    rows = diag_common.beta_sweep(layer, [0.2, 0.5, 1.0], x)
    assert len(rows) == 3
    for r in rows:
        assert abs(r["F"] - (r["quad"] + r["interaction"] + r["entropy"])) < 1e-4
