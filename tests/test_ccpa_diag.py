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


def test_orthogonal_floor_grows_with_beta():
    """RC2: continuous (orthogonal) prior floor grows as m saturates toward ±1."""
    from experiments.ccpa import d_rc2_errorfloor
    layer = diag_common.build_layer(n=16, rho=0.7, seed=0)
    g = torch.Generator().manual_seed(0)
    prior = torch.randn(4, 16, generator=g)
    q, _ = torch.linalg.qr(prior.t()); ortho = q.t()
    x = torch.randn(64, 16)
    fo = d_rc2_errorfloor.error_floor(layer, ortho, x, betas=[0.1, 1.0, 1.1])
    assert fo[-1] > 1.5 * fo[0]  # saturation pushes m away from continuous prior


def test_rho_grows_without_cap():
    from experiments.ccpa import d_rc3_rho_degeneracy
    from vpsc.recurrent import RecurrentVPSCNet
    torch.manual_seed(0)
    net = RecurrentVPSCNet([8, 8], n_classes=4, beta=0.5, rec_rho0=0.5, lam_spec=0.0)
    rhos = d_rc3_rho_degeneracy.train_no_cap(net, epochs=20, lr=0.05, T=8, n_in=8, seed=0)
    assert rhos[-1] > rhos[0]
    assert rhos[-1] > 0.9
