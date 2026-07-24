"""TDD tests for CCPA Phase 1-2 fixes."""
import torch
from experiments.ccpa import diag_common
from vpsc.recurrent import _sym, _binary_entropy


def test_phi_structure():
    """Fix1: Phi = beta*(quad+interaction+wd) - sum H_bin (no barrier/tikhonov by default)."""
    layer = diag_common.build_layer(n=6, rho=0.5, seed=0, beta=0.8)
    x = torch.randn(2, 6); m = layer(x); mu = torch.zeros_like(m)
    Phi = layer.free_energy_phi(m, mu)
    Ws = _sym(layer.W_rec)
    quad = 0.5 * (1.0 / layer.sigma ** 2) * ((m - mu) ** 2).sum(dim=-1)
    inter = -0.5 * (m * (m @ Ws)).sum(dim=-1)
    wd = 0.5 * layer.wd * (layer.W_rec ** 2).sum()
    expected = (layer.beta * (quad + inter + wd) - _binary_entropy(m).sum(dim=-1)).mean()
    assert abs(float(Phi.mean().item()) - float(expected.item())) < 1e-4


def test_phi_monotone_at_fixed_beta():
    """Fix1: at fixed beta, Phi non-increasing under gradient steps (Theorem 2 on Phi)."""
    from vpsc.recurrent import RecurrentVPSCNet
    torch.manual_seed(0)
    net = RecurrentVPSCNet([6, 6], n_classes=4, beta=0.5, rec_rho0=0.5)
    x = torch.randn(8, 6, 6); y = torch.randint(0, 4, (6,))
    opt = torch.optim.SGD(net.parameters(), lr=0.01)
    phis = []
    for _ in range(10):
        opt.zero_grad(); out = net(x)
        Phi = net.total_free_energy_phi(out["traj"], labels=y)
        Phi.backward(); opt.step(); net.project_spectral()
        phis.append(float(Phi.item()))
    assert phis[-1] <= phis[0] + 1e-3
