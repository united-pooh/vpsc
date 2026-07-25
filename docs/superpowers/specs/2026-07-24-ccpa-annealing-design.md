# CCPA 退火修复实验 — 设计规格

- 日期：2026-07-24
- 执行分支：`codex/research-ccpa-annealing`（执行时从最新 `main` 新建）
- 状态：已通过 brainstorming 设计审阅，待写实现计划（writing-plans）
- 范围：最小验证档（SHD，CCPA vs 纯 F，不含 LSTM/Transformer）
- 结构：门控顺序（Phase 0 → gate0 → Phase 1 → gate1 → Phase 2 → Phase 3 → 写 LOG）

---

## 1. 背景与根因（来自 2026-07-23 推导）

VPSC 每层自由能（`vpsc/recurrent.py:167–179`）：

$$F_l = \tfrac{1}{2\sigma^2}\|\mathbf{m}-\boldsymbol{\mu}\|^2 - \tfrac12 \mathbf{m}^\top W_s \mathbf{m} + \tfrac{1}{\beta}\sum_i H_{\text{bin}}(m_i) + \tfrac{wd}{2}\|\mathbf{W}\|^2 + \lambda_{\text{spec}}\max(0,\rho(W_s)-\rho_{\max})^2$$

不动点 `m = tanh(β(Ws m + I − θ))`，`β_c = 1/ρ(Ws)`，线性退火 `β: start → β_c`，顶层 `μ = class_prior[label]`（正交、连续）。

四个根因：

- **RC1 非相干同伦**：熵按 1/β 标度，其它项不标度 → 跨 β 无 Lyapunov 保证（定理 2 仅固定 β 成立）。
- **RC2 饱和抬高预测误差**：顶层 `class_prior` 连续正交，`m` 饱和向 ±1 → 误差地板随 β 上升 → 非判别（chance）。
- **RC3 饱和处熵消失** → `ρ(W)→∞` 退化；硬盖 `project_spectral` 是创可贴。
- **RC4 β_c 处 Hessian/Jacobian 奇异**：`ρ(DG)→1`、`λ_min(H_F)→0`，训练在最优点条件最差。

## 2. CCPA 修复（每根因一个 Fix）

- **Fix1（RC1）**：无量纲自由能 `Φ = β·E − S`，把现有项重标度进 `E`：`Φ = β·[quad+interaction+wd] − ΣH_bin`（此时不含屏障；log-det 屏障由 Fix2 加入，合并后 `Φ = β·[quad+interaction+wd] − ΣH_bin + B(W)`）。`β_c` 保持。
- **Fix2（RC3）**：log-det 谱屏障 `B = −(γ/2) logdet(I − β² Ws²)`，替换 `project_spectral` + `lam_spec`。
- **Fix3（RC2）**：PC 推理回路——每 β 步梯度前 `K` 轮自上而下/自下而上松弛至 `‖Δm‖ < tol`，顶层 prior 被推断非硬设。
- **Fix4（RC4）**：Hessian 监控延拓——`ContinuationAnnealer` 跟 `λ_min(H_Φ)`、退火到 `β_c − δ`、warm-start、小 β 增量、Tikhonov `(ε/2)‖m‖²` 保 H 正定。

---

## 3. 分支 / 计算 / 工件约定

- **分支**：`codex/research-ccpa-annealing`（从最新 `main` 新建）。诊断脚本、Fix1–4 实现全部在此分支；仅 PASS 最小代码 cherry-pick 进 `main`（遵循 `docs/research_workflow.md`）。
- **计算**：CPU 本地小规模（`toy_verify`/`deep_critical` 量级；SHD 真 corpus ~70MB HDF5，小网 CPU 可跑）。不上 FE-2H GPU——本计划是"退火本身修好没"的理论验证，非主线扩规模。
- **工件**：`results/ccpa/<exp>.{json,png}`，JSON 含命令/环境/seed/原始数值/SHA-256（沿用仓库 provenance 约定）。
- **最终结果摘要**写 `dev/LOG.md`。

## 4. 阶段门控骨架

Phase 0 诊断 → **gate0** → Phase 1 Fix1+2 → **gate1** → Phase 2 Fix3+4 → Phase 3 验证 → 写 LOG。
每 gate 预注册、不过即停手记负结果，不为晋级事后调参。

## 5. Phase 0 — 诊断（5 个，CPU 一晚）

| 实验 | 做什么 | 产物 | 确认 |
|---|---|---|---|
| D-RC1 | 小递归层（复用 `RecurrentMeanFieldLayer`），扫 β∈[0.1, 1.2·β_c]，固定 W(ρ<1)+固定输入；解不动点；分解 F_l 三分量随 β | `d_rc1_components` | RC1 |
| D-RC2 | 顶层 `class_prior`（正交）vs 双极 `sign(prior)`；扫 β，测 `½‖m_top − prior‖²` 地板 | `d_rc2_errorfloor` | RC2 |
| D-RC3 | `RecurrentVPSCNet` 纯 F 训练，关 `project_spectral` 且 `lam_spec=0`；画 `ρ(W_s)` 随步数 | `d_rc3_rho_degeneracy` | RC3 |
| D-RC4a | 不动点处 `H_F = ∂²F/∂m²`（`torch.func.hessian`），特征分解，画 `λ_min(β)` | `d_rc4a_hessian` | RC4 |
| D-RC4b | `DG = β·diag(1−m²)·W_s` 的 `ρ(DG)(β)`（`svdvals`） | `d_rc4b_jacobian` | RC4 |

**gate0**：RC1/RC2/RC3/RC4 至少 3 条被实测支持（RC4a+4b 计一条）。<3 → STOP 记负，回看推导。

## 6. Phase 1 — 核心修复（Fix1+Fix2）

- **Fix1（无量纲 Φ）**：`RecurrentMeanFieldLayer` 加 `free_energy_phi` 分支。
  验证：(a) `∂Φ/∂m=0` 数值梯度检查仍给 `m=tanh(β(Ws m+I−θ))`；(b) `β_c=1/ρ(W)` 不变（重跑 `toy_verify` P2）；(c) 固定 β 下 Φ 单调非增（P1 逻辑）。
  产物：`fix1_{gradcheck,p2,p1}`
- **Fix2（log-det 屏障 `B=−(γ/2)logdet(I−β²Ws²)`，`slogdet`+ε 稳定）**：替换 `project_spectral`+`lam_spec`。
  产物：`fix2_rho_bounded`
- **gate1**：Fix1 梯度检查过 + P1/P2 在 Φ 上仍过 + Fix2 不靠硬盖 `ρ` 有界 + `β_c` 保持。否则 STOP 记负。

## 7. Phase 2 — 推理 + 延拓（Fix3+Fix4）

- **Fix3（PC 推理回路）**：每 β 步梯度前 `K` 轮自上而下/自下而上松弛至 `‖Δm‖ < tol`。
  门：顶层误差地板（D-RC2）在 PC 下低于硬 prior。产物：`fix3_pc_inference`
- **Fix4（`ContinuationAnnealer`）**：跟 `λ_min(H_Φ)`、退火到 `β_c − δ`、warm-start、小 β 增量、Tikhonov `(ε/2)‖m‖²`。
  门：训练全程 `λ_min > ε`、`β* ∈ [β_c − δ, β_c]`。产物：`fix4_continuation`
- 注：Fix3 是本计划最高工程风险项（跨层 inference 是新代码）。

## 8. Phase 3 — 验证（最小档）

SHD（真 corpus 或 synthetic fallback），≥3 seeds：CCPA（Fix1–4）vs 纯 F（`free_energy_loss` + `BetaAnnealer` + `project_spectral`）。
报告：test acc、F 轨迹、`λ_min` 轨迹、`ρ` 轨迹（无硬盖）、`β*`。
产物：`val_shd_ccpa_vs_puref`

## 9. 预注册成功门 + 失败处理

验证成功门（预注册，未达即记 NEGATIVE 不调参）：

- **Higher**：CCPA acc > 2×chance（>10%；SHD 20 类 chance=5%）且显著高于纯 F（≥3 seeds, p<0.05）。
- **Stronger**：训练全程 `λ_min(H_Φ) > ε`。
- **Cheaper/Stronger**：`ρ(W_s) ≤ ρ_max(0.9)` 全程不靠 `project_spectral`。
- **机制完整**：`β_c` 保持（与 `1/ρ(W)` 偏差 ≤5%）。

CCPA acc ≤ chance 或不显著高于纯 F → NEGATIVE，记 LOG 与分支，不为晋 `main` 改判据。

## 10. 冻结超参（Phase 1 运行前冻结，记入 LOG，不为过门调参）

| 超参 | 默认值 | 说明 |
|---|---|---|
| γ（log-det 屏障强度） | 1.0 | 量级对齐旧 `lam_spec` |
| δ（β 回退） | 自适应到 `λ_min > ε`，上限 `0.1·β_c` | Fix4 |
| K（PC 松弛轮数） | 8 | 对齐 `n_relax` |
| tol（PC 收敛阈） | 1e-4 | Fix3 |
| ε（Hessian/Tikhonov 正定阈） | 1e-3 | Fix4 + 成功门 |
| ρ_max | 0.9 | 沿用 `RecurrentVPSCNet` 默认 |
| seeds | ≥3 | 验证 |

## 11. LOG 写入

`dev/LOG.md` 顶部加 `2026-07-24：CCPA 退火修复实验` 条目（NoA 格式：背景/假设/冻结配置/预注册门/原始结果+SHA/各 gate 判定/claim 边界）。LOG 条目按审计先例提交 `main`；研究代码留 `codex/research-ccpa-annealing`，仅 PASS 最小代码 cherry-pick 进 `main`。

## 12. 不在本计划范围（YAGNI）

- 等参 LSTM/Transformer 主线性价比对（属"打败 Transformer"另一轮）。
- 35M / 0.8B 扩规模（write-budget 实验通过前不扩）。
- MNIST 双任务、`ce_loss`-P2 同参对比（中等档 B，本轮不做）。
