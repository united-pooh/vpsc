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


def test_rho_bounded_without_cap_via_barrier():
    """Fix2: with log-det barrier, rho stays bounded WITHOUT project_spectral."""
    from vpsc.recurrent import RecurrentVPSCNet, _sym, spectral_radius_square
    torch.manual_seed(0)
    net = RecurrentVPSCNet([8, 8], n_classes=4, beta=0.5, rec_rho0=0.6, lam_spec=0.0)
    for l in net.layers:
        l.use_log_det_barrier = True; l.gamma = 1.0
    x = torch.randn(8, 32, 8); y = torch.randint(0, 4, (32,))
    opt = torch.optim.Adam(net.parameters(), lr=0.03)
    rhos = []
    for _ in range(40):
        opt.zero_grad(); out = net(x)
        loss = net.total_free_energy_phi(out["traj"], labels=y)
        loss.backward(); opt.step()  # NO project_spectral
        rhos.append(max(spectral_radius_square(_sym(l.W_rec.data)) for l in net.layers))
    assert max(rhos) <= 0.95  # bounded without hard cap


def test_pc_inference_differs_from_hard_forward():
    """Fix3: pc_inference produces finite states that differ from the hard forward."""
    from vpsc.recurrent import RecurrentVPSCNet
    torch.manual_seed(0)
    net = RecurrentVPSCNet([6, 6], n_classes=4, beta=0.7, rec_rho0=0.5)
    for l in net.layers:
        l.use_log_det_barrier = True; l.gamma = 1.0
    x = torch.randn(8, 6, 6)
    out_hard = net(x)
    out_pc = net.pc_inference(x, K=8, tol=1e-4)
    m_hard = out_hard["traj"][-1][-1]
    m_pc = out_pc["traj"][-1][-1]
    assert torch.isfinite(m_pc).all()
    assert not torch.allclose(m_hard, m_pc, atol=1e-3)


def test_continuation_annealer_stays_below_beta_c():
    """Fix4: anneal toward beta_c - delta, never exceeding beta_c."""
    from vpsc.recurrent import RecurrentVPSCNet
    from vpsc.free_energy import ContinuationAnnealer
    torch.manual_seed(0)
    net = RecurrentVPSCNet([6, 6], n_classes=4, rec_rho0=0.5, beta=0.2)
    beta_c = net.critical_beta()
    ann = ContinuationAnnealer(net, start=0.2, steps=20)
    betas = [ann.step() for _ in range(20)]
    assert betas[-1] <= beta_c + 1e-6
    assert betas[-1] >= beta_c - 0.1 * beta_c
    assert all(b <= beta_c + 1e-6 for b in betas)
