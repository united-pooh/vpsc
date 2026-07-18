# VPSC 研究与实验日志

本日志按 NoA 规范维护：证据层（命令/运行/产物）+ 决策层（动机/假设/证据/决定）。倒序排列。

---

## 2026-07-18：E3-IC0 结果 — 二值 input event 将 A0 推到 96%，但未过 99%/streaming 硬门（混合结果）

### 证据

- 正式命令：`.venv-wsl/bin/python experiments/e3_ic0_input_code.py --output results/e3_scan/e3_ic0_input_code.json`；产物 SHA-256 `799A35142D980C0D019663F5D9D5988F61EA0F2FBDF02BFD75724A544968E29B`。
- input event/output spike 二值性、scan/serial output/state/gradient 等价由新增单测通过。参数/state 与 L1 相同（8,516 / 336 bytes）。

| core | A0 accuracy | NLL | last-100 loss | A0 train p50 ms |
|---|---:|---:|---:|---:|
| IC0 | **0.9619** | 0.29 | 0.49 | 1.26 |
| LSTM | 1.0000 | 0.06 | 0.11 | **0.88** |
| Transformer | 1.0000 | 0.16 | 0.54 | 1.66 |

IC0 相比 sigmoid-charge L1 的 32.23% 大幅提高，证明显式 binary input-event population code 是正确方向；但冻结门为 99%，实际少约 2.8 个百分点，故 **H-IC0-A0 FAIL**，不向上取整、不增加 update、不跑 short delay。

T=512 core forward+backward p50：threads 1/4/16 为 `1.319/1.110/1.673 ms`，LSTM 为 `1.194/1.431/2.593 ms`；IC0 在 4/16 线程快 `22.5%/35.5%`，scan node 45。streaming p95 IC0 `0.186/0.168/0.192 ms`，仍慢于 LSTM `0.114/0.125/0.163 ms`，所以联合 ANN 门 FAIL。

### 判定

- 正面：strict binary input events + exact hard reset + time-parallel scan 已同时接近 ANN 训练速度与即时 token 质量，这是当前最强 strict-SNN 工程候选。
- 负面：仍未满足预注册质量与实时 step；additive exact-modulo 的结构调参主线在此停止。后续只允许把 IC0 作为独立 eligibility/online-local 或 exact-event kernel 的 substrate，不能继续事后微调 IC0 同一实验。

---

## 2026-07-18：E3-IC0 预注册 — learnable binary input-event code + exact modulo scan

### 唯一结构变化

S0/L1 的连续 sigmoid charge 让小 embedding 差异主要表现为 phase 微扰，A0/LG0 都未形成可靠类别 code。IC0 仍是一层、仍用同一个 cumulative floor/difference/modulo hard-reset scan，但把每个 token 的 sensory projection 先经 hard surrogate threshold 得到明确的 E/I input events `z_E,z_I∈{0,1}`，再令 `q=0.125+0.75z∈{0.125,0.875}`。两值均为精确 `2^-3` 网格且 `<1`，所以 output spike 仍严格二值、scan/streaming exact。

输入 E/I linear bias 冻结初始化为 0（之后可训练），避免小 embedding 被随机 bias 全部压成同一事件 pattern；其余参数、readout、surrogate、state 与 L1 相同。forward 内没有 sigmoid/ANN recurrent；continuous membrane只作为标准 SNN state/readout存在。

### 门

- scan/serial/streaming 的 output/state/spike/input与全参数 gradient 继续使用 S0 `2e-6/1e-5` 等价门；input events 与 output spikes 都必须只含0/1。
- `D=32,state_dim=42` 参数/state 与 L1 相同。**H-IC0-A0** 完全复用 D2/L1 A0 300-update 数据与共同 wrapper，global decoder accuracy≥99%，LSTM/Transformer≥99%。不添加 local loss。
- 同时复测 T=512 threads 1/4/16 train+streaming；只有同线程均≤LSTM 才过 ANN 门。A0 PASS 才跑 delay4/16；FAIL 则 additive exact-modulo quality路线终止。
- 产物冻结为 `results/e3_scan/e3_ic0_input_code.json`。

---

## 2026-07-18：E3-LG0 结果 — 固定 local code 仅 21%，且损害 global decoder（负面结果）

### 证据

- 正式命令：`.venv-wsl/bin/python experiments/e3_lg0_local_code.py --output results/e3_scan/e3_lg0_local_code.json`；产物 SHA-256 `300E94B26C163FAE81EC53ED5195FA445D347BC64D433049CD6EEA36D401DFEB`。
- LSTM/Transformer global test accuracy 均为 100%，任务与共享 runner 继续有效。global-only L1 本次复现为 33%；LG0-L1 的 global accuracy 反而降到 `29%`、NLL `2.52`。
- LG0 固定 codebook 的 local test accuracy 只有 `21%`；last-100 global/local loss 为 `2.69/2.67`。训练 p50 从 global-only `1.28 ms` 增至 `1.46 ms`。

### 判定

- **H-LG0-A0 FAIL**，短 delay不运行。失败不是“local code 学会但 global readout 接不上”：local 本身也远低于 99%，且辅助目标与 global objective 发生负迁移。
- 这条固定 supervised code 不是 eligibility learning；结果只否定 `weight=1,temp=0.2,seed=19001` 的预注册训练目标，不外推到 BrainTrace/pp-prop。但结合 L1 32% 与 LG0 21%，继续只换 loss 的优先级下降。
- 下一结构诊断应直接改变 input event representation（显式 learnable binary population injection），保持 exact modulo scan；若仍失败，再终止 additive S0 quality 主线并把资源转到 exact event/eligibility 独立实现。

---

## 2026-07-18：E3-LG0 预注册 — 固定 population code 的训练期 local objective

### 假设 / 构造

L1 已把 A0 从 S0-L2 的 8.69% 提高到 32.23%，证明直接 residual readout 保留了部分 token 信息，但全局 surrogate objective 未形成可分的 16 类 spike code。LG0 保持 L1 forward/inference 完全不变，只在训练期给 WRITE 位置的原始 `[s_E,s_I,u_E,u_I]` 加一个无参数 local code loss。

- 用 seed `19001` 冻结 16×168 的 Rademacher `{-1,+1}` codebook；representation 映射为 `2r-1` 后与 L2-normalised codebook 做 cosine logits，temperature `0.2`；local CE target 就是当前已知 payload。总 loss=`global query CE + 1.0×local CE`。
- codebook 不训练、不计参数，推理时完全删除 local loss；因此若成功，它证明训练目标/信用分配可修复 population code，不是用额外 ANN inference head 偷渡质量。

### 门 / 边界

- 完全复用 L1-A0 的数据 hash 生成规则、300 update、共同初始化与参数公平；global-only L1、LG0-L1、LSTM、Transformer 同跑。**H-LG0-A0 PASS** 要求 LG0 global decoder test accuracy≥99%，而不是只看 local code accuracy。
- 另报 global/local loss、global-only 对照和训练 p50；不因看到结果调整 weight/temperature/codebook。A0 PASS 才运行短 delay 4/16，并在 delay 任务只对 WRITE 位置用 current-token local target，不泄漏未来 query target。
- local target 使用离散 token identity，是工程化 self-supervised encoder loss，但不等于生物 eligibility rule；真正 online eligibility/pp-prop 仍是后续独立分支。
- 产物冻结为 `results/e3_scan/e3_lg0_local_code.json`。

---

## 2026-07-18：E3-S0-L1 结果 — 即时编码升至 32%，训练部分超过 LSTM，但质量/streaming 双门失败

### 证据

- 正式命令：`.venv-wsl/bin/python experiments/e3_s0_l1.py --output results/e3_scan/e3_s0_l1.json`；产物 SHA-256 `DC4BBF4141243A54190248785A82A5869D8DC22CC052019F0798CB125E257FBF`。
- A0：LSTM/Transformer 再次为 100%，L1 test accuracy `0.3223`、NLL `2.300`、last-100 train loss `2.57`。相比 L2 的 8.69% 是明确改善，但远低于预注册 99%，故 **H-L1-A0 FAIL**，短 delay 未运行。
- 参数匹配：L1 core 8,516（LSTM 8,448），state 336 bytes（LSTM 256）。T=512 forward+backward：

| threads | S0-L1 ms | S0-L2 | LSTM | Transformer |
|---:|---:|---:|---:|---:|
| 1 | 1.246 | 1.92 | **1.138** | 7.00 |
| 4 | **1.381** | 1.97 | 1.513 | 3.28 |
| 16 | **1.859** | 3.33 | 2.635 | 3.45 |

L1 scan node 55；在 4/16 线程训练 p50 首次比 LSTM 快约 `8.7%/29.4%`，但 streaming p95 `0.160/0.296 ms` 仍慢于 LSTM `0.143/0.165 ms`，所以联合 ANN 门 FAIL。

### 判定

- 去掉 spike-only 第二层同时改善了质量与训练速度，支持“第二层是信息/工程负担”；但 32% 说明 additive phase/surrogate 仍没有可靠 token code。
- L1 是当前 strict hard-reset 候选中最接近 ANN train speed 的实现，却仍不能进入真实任务。下一步用 LG0 训练期 local code objective 检验表示学习，而不是继续改 forward 或增加 update。

---

## 2026-07-18：E3-S0-L1 预注册 — 去除 spike-only 第二层，直接读出 exact-reset population code

### 动机 / 唯一改动

D2 已把 S0 质量失败定位到最小同位 token 编码；两层 S0 的第二层只接收第一层瞬时 spike，丢弃其 residual membrane。L1 只做一个预注册消融：`num_layers=1`，readout 直接使用该层 `[s_E,-s_I,u_E,-u_I]`；cumulative charge、2^-10/2^-12 量化、0/1 spike、hard modulo reset、surrogate 与 prefix scan 全部不变。

`D=32,state_dim=42` 时 core 8,516 参数，较 LSTM 8,448 多 `0.805%`；state 336 bytes，较 LSTM 256 多 31.25%，必须报告。L1 没有 signed inter-layer pathway，只保留 sensory→E/I 与 signed E/I readout；若成功，它是 exact-reset population encoder 的可行性证据，不是完整 recurrent E/I world model。

### 冻结门

- **H-L1-A0**：完全复用 D2-A0 的 `T=32` 随机 WRITE 同位 16-token decode、300 update、共同 wrapper/初始化；L1/LSTM/Transformer 均须≥99%。
- **H-L1-SPEED**：`B=1,T=512,D=32` scan forward+backward 与 continuous step p95，在 threads 1/4/16 与 LSTM/Transformer/S0-L2 同范围比较；只有同一线程 train+step 都≤LSTM 才过 ANN 门。
- A0 PASS 后才运行 D2-B 的随机 delay 4/16；B PASS 后才回到 marked 64/256。A0 FAIL 则 exact additive population code 路线终止，转 local objective/eligibility 或 event segmentation。
- 产物冻结为 `results/e3_scan/e3_s0_l1.json`；先写 A0+speed，后续质量若执行可追加独立文件，不覆盖。

---

## 2026-07-18：E3-P0 结果 — complex scan 等价，但 T=512 并行/ANN 速度门失败（负面结果）

### 证据

- 正式命令：`.venv-wsl/bin/python experiments/e3_p0_oscillator_benchmark.py --output results/e3_scan/e3_p0_oscillator_benchmark.json`；产物 SHA-256 `0B37CC85AE45B2A126F2E9D7AA918CBF9903F554ABF98762C47063ABA3234879`。
- `B/T=(1/1),(4/32),(1/512)` 的 serial/scan sequence、complex state、逐元素 spike、streaming、input/state/全参数 gradient 全部通过冻结 `3e-5/1e-4`；P0 8,371 参数、248-byte complex state，与 LSTM 8,448/256 相近。

T=512 forward+backward p50 ms：

| threads | P0 serial | P0 scan | speedup | S0 scan | LSTM | Transformer |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 14.08 | 3.73 | 3.78× | 1.82 | **1.23** | 7.37 |
| 4 | 14.66 | 2.51 | **5.84×** | 2.02 | **1.44** | 2.77 |
| 16 | 19.79 | 5.12 | 3.87× | 3.52 | **2.78** | 3.79 |

scan/serial autograd node ratio为 `8.06%`，图深显著下降，但最好加速仅 `5.84×<10×`；复数 multiply/cat 的多轮 Hillis–Steele kernel 在 CPU 上吞掉了算法收益。T=2,048 时相对 serial 达 `13.29–14.68×`，绝对 scan 为 `6.32–9.65 ms`，仍未稳定赢 LSTM `3.61–9.16 ms`，且慢于 S0 `4.28–6.51 ms`。

streaming P0 p95 在 1/4/16 线程为 `0.266/0.149/0.256 ms`，对应 LSTM `0.135/0.096/0.173 ms`；联合 ANN 门三档均失败。

### 判定

- **H-P0-EQ PASS；H-P0-PAR FAIL；H-P0-ANN FAIL；A0 按预注册 NOT_RUN。** reset-free PRF 分支没有获得继续投入质量训练的速度依据。
- 负面结果限于当前 PyTorch complex prefix 实现/CPU；不否定 FFT/专用 complex kernel 在 GPU 的可能性。本机无 CUDA，不能把文献中的 GPU 加速移植为本项目证据。
- 下一步优先 exact event segmentation 或 sparse event kernel，因为 S0 已证明 additive exact reset 的并行速度可行；同时把 eligibility/local objective 作为解决 S0 surrogate 表示学习失败的独立方向。

---

## 2026-07-18：E3-P0 预注册 — PRF-style selective complex oscillator scan（reset-free 分支）

### 构造 / 诚实边界

对每个复数 oscillator：`h_t=a_t h_{t-1}+b_t`；`|a_t|∈[0.5,0.995]`，phase 由可学习 base frequency 加输入选择性 phase modulation，`b_t` 由输入的 real/imag drive 产生。复 affine pair 使用与 S1 已验证相同的 associative composition，serial/scan 应在浮点容差内等价。

输出 E/I spikes 分别为 real/imag membrane 越过冻结阈值的 0/1 surrogate step，readout 使用 `[s_real,-s_imag,Re(h),Im(h)]`。内部 oscillator **不执行 reset**；因此即使速度/质量成功，也只能成为后续世界模型的 oscillatory spiking substrate，不能被记为最终 strict hard-reset SNN 替代 ANN。该分支测试的是 PRF/稳定复振荡数学，不覆盖 S1 的 hard-reset fixed-point 失败。

### 冻结门

- `D=32,state_dim=31`，使 core 参数与 LSTM 差≤2%，complex state bytes 单独报告。scan/serial 覆盖 `B∈{1,4},T∈{1,32,512}`；spike 必须逐元素相同，sequence/state/input/全参数 gradient 通过 `atol=3e-5,rtol=1e-4`（complex reduction 的容差在结果前冻结）。
- **H-P0-PAR**：`B=1,T=512,D=32` scan/serial forward+backward p50≥10×，scan node≤serial 25%；同时与 E3-S0/LSTM/Transformer 报绝对速度。
- **H-P0-A0**：复用 D2 A0 数据/共同 wrapper，300 update 后 16-token same-position test accuracy≥99%，LSTM/Transformer 仍≥99%；未通过则不跑延迟记忆。
- **H-P0-ANN**：T=512 train p50 与 continuous streaming p95 在同一线程档达到 LSTM；reset-free 边界不因速度 PASS 消失。

### 决策 / 产物

- EQ/PAR 通过且 A0 通过后，才按短 4/16 → 长 64/256 的顺序测 token memory；若 oscillator 质量通过，再研究 event-triggered exact reset/phase wrap，而不是把连续 complex state隐藏起来。
- 本机 CPU threads 1/4/16；CUDA-aware 但当前 unavailable。microbenchmark 产物 `results/e3_scan/e3_p0_oscillator_benchmark.json`；A0 结果可同文件或独立 `e3_p0_a0.json`，必须明确 scope。

---

## 2026-07-18：E3-S1-FP 结果 — K≤8 无法逼近 serial hard reset，收敛门失败（明确负面结果）

### 证据

- 正式命令：`.venv-wsl/bin/python experiments/e3_s1_fixed_point.py --output results/e3_scan/e3_s1_fixed_point.json`；产物 SHA-256 `BF453962912F9F0A15CEFC2AB01A808F05137B457B323CC9BD5E62A36836227C`。
- affine prefix scan 本身已由单测验证：无 reset 时与 serial affine recurrence 的 forward/gradient 通过 `2e-6/1e-5`；T=1 fixed-point 与 exact hard reset 完全一致。因此失败定位在跨时间 reset-event fixed point，而不是 pair composition 实现错误。

四个正式 case 的 serial spike rate 都约 0.50。K=8：

| B | T | input scale | spike mismatch | output max abs | state max abs |
|---:|---:|---:|---:|---:|---:|
| 1 | 32 | 0.25 | 0.3740 | 1.056 | 0.847 |
| 4 | 32 | 1.00 | 0.1492 | 1.560 | 0.982 |
| 1 | 512 | 0.25 | **0.4918** | 1.398 | 0.869 |
| 1 | 512 | 1.00 | 0.1681 | 1.408 | 0.561 |

跨全部 case 的 worst mismatch 从 K=1/2/4/8 的 `0.4998/0.4978/0.4958/0.4918` 几乎没有改善；worst output/state error 在 K=8 仍为 `1.560/0.982`，远超预注册 `0.1%` 与 `1e-3` 门。

### 判定 / 数学解释

- **H-S1-CONV FAIL；H-S1-PAR、H-S1-A0 均按预注册 NOT_RUN。** 不能对错误 spike 序列报告速度或质量。
- 证据支持的解释：Jacobi round 只根据上一轮 `s_{t-1}` 切断 affine coefficient；在约每两步一次 reset 的链上，正确 segment boundary 必须沿时间传播，常数 K 并没有把 hard-reset 因果依赖变成可靠的 O(log T) 解。动态 decay 的 contraction 不足以跨越 threshold discontinuity。
- 这不否定 reset-free affine/oscillatory scan；它否定的是“用 K≤8 fixed-point correction 保留本轮 exact serial hard reset”的方案。下一步分开测试 PRF/reset-free oscillatory spike code 与 exact event segmentation，不把两者混成一个成功故事。

---

## 2026-07-18：E3-S1-FP 预注册 — dynamic-decay affine scan 与 hard-reset fixed-point correction

### 数学路线

S0 的 prefix sum 已解决时间并行，却在 A0 即时 token 编码失败。S1 不继续调 S0 readout，而测试另一条已预先列出的数学路线：对每个 E/I neuron 由输入生成 `a_t∈[0.5,0.99]` 与正电荷 `b_t`，serial hard-reset dynamics 为：

`p_t = a_t·u_{t-1}+b_t`；`s_t=H(p_t-1)`；`u_t=p_t·(1-s_t)`。

给定上一轮 spike 估计 `ŝ_{t-1}`，它变为 affine recurrence `p_t=A_t p_{t-1}+b_t`，其中 `A_t=a_t(1-ŝ_{t-1})`；pair composition `(A₂,b₂)∘(A₁,b₁)=(A₂A₁,b₂+A₂b₁)` 可用 Hillis–Steele prefix scan 在 `O(log T)` graph depth 并行。每轮由新 `p` 更新 hard spike，做 `K∈{1,2,4,8}` 次 Jacobi/fixed-point correction；forward 始终是 0/1 spike 与乘法 hard reset，backward 对 threshold 用冻结 surrogate、对 reset gate stop-gradient。

这是近似并行求解，不先验宣称 K 与 T 无关；serial reference 是真值。如果 hard reset 的因果边界必须逐步传播而 K=8 仍不收敛，本路线应记录失败，而不是把软 activation 当成功。

### 预注册门

- **H-S1-CONV**：随机/边界压力输入，`B∈{1,4},T∈{32,512}`；报告 K=1/2/4/8 对 serial 的 spike mismatch、state/output max error。选择最小满足 spike mismatch≤0.1%、state/output `atol=1e-3,rtol=1e-3` 的 K；若 K=8 仍失败则 CONV FAIL。
- **H-S1-PAR**：只有 CONV PASS 才评速度；所选 K 在 `B=1,T=512,D=32` forward+backward p50 至少比 serial `10×`，autograd node≤serial 25%。
- **H-S1-A0**：参数匹配的同位 16-token decode，冻结 D2-A0 的 300 update；S1 test accuracy≥99%，且 LSTM/Transformer 校验仍为≥99%。未通过则不进入长延迟。
- **H-S1-ANN**：T=512 train p50 与 continuous step p95 必须在同一线程档达到 LSTM；质量门和速度门分开报告。

### 实现边界 / 后续

- 先实现 diagonal dynamic decay，避免把 dense ANN recurrent GEMM 偷渡进 scan；E/I input drive 与 readout 都用 discrete spike + post-reset membrane。后续 signed inter-layer只在基本收敛/A0 后增加。
- CUDA-aware runner保留；本机 CUDA unavailable 只能报告 CPU。产物冻结为 `results/e3_scan/e3_s1_fixed_point.json`。
- 若 CONV FAIL，转 PRF/reset-free oscillatory spike code 或 exact event segmentation；若 CONV PASS 但 A0 FAIL，转 eligibility/local objective，而不是增加 K 掩盖表示失败。

---

## 2026-07-18：E3-MEM-D2 结果 — 共享 runner 有效，E3-S0 在同位 token 编码即失败

### 证据

- 正式命令：`.venv-wsl/bin/python experiments/e3_memory_diagnostic.py --output results/e3_scan/e3_memory_diagnostic.json`；产物 SHA-256 `F6A174C2E0EAB45FF35FBEE6EB8267120BAE164FEE3C7542101FD97F9C7CBB18`。
- A0 使用 `T=32` 随机 batch，在 WRITE token 同一位置直接预测 payload，训练 300 update；参数 E3/LSTM/Transformer 为 `10,264/10,256/10,416`，公平门通过。

| core | test accuracy | NLL | last-100 train loss | p50 ms/update |
|---|---:|---:|---:|---:|
| E3-S0 scan | **0.0869** | 2.7492 | 2.7649 | 2.107 |
| LSTM | **1.0000** | 0.0658 | 0.1255 | **0.882** |
| Transformer | **1.0000** | 0.1473 | 0.5116 | 1.431 |

### 判定

- **A0 shared-runner validation PASS，E3-S0 A0 FAIL。** LSTM/Transformer 在相同 embedding/decoder/data/optimizer 下达到 100%，因此不能再把此前 chance 归因于 gather、target、optimizer 或共同训练代码。
- E3 的 8.69% 只略高于 6.25% chance；梯度 finite 且 norm 非零。证据支持结构性信息瓶颈：S0 第二层只接收第一层瞬时 0/1 spike，在小幅 embedding drive 与随机 phase 下没有形成可解码的 16 类 population code。A1/B 按预注册没有运行，避免在已失败的最小门上浪费实验。
- S0 的时间并行速度结果仍成立，但质量路线在最小 token 编码处终止；不能进入 TextWorld/HomeGrid。下一步执行 S1 fixed-point selective reset scan。

---

## 2026-07-18：E3-MEM-D2 预注册 — 同位/短延迟诊断阶梯，定位 runner 与长期信用分配

### 冻结阶梯

1. **A0 same-position decode**：`T=32`，在标记 WRITE token 所在位置直接预测 payload；随机 batch，300 update。三核心 test accuracy 均须≥99%，否则 embedding/query gather/decoder/optimizer runner 有 bug。
2. **A1 fixed-batch overfit**：`T=32`，显式 WRITE 后在 delay 1/4 的 READ 位置预测；同一个 `B=8` batch 重复 1,000 update。三核心 train accuracy 均须≥99%，否则该核心/梯度路径无法完成最小 delayed credit assignment；固定 batch 结果不作为泛化证据。
3. **B random generalisation**：只有 A0/A1 全部通过才执行；`T=64`、delay 4/16、随机且隔离的 train/test batch，3 seed、500 update。LSTM overall/两 bucket≥90% 才有效；E3 quality PASS 仍要求≥90% 且对 LSTM 非劣 2 个百分点。

所有阶段 `D=32,B=8`、threads 4、AdamW `1e-3`、参数差≤2%，共同 embedding/decoder 初始化。D2 只诊断学习性，不重测或覆盖 S0 已失败的 streaming/`B=8,T=512` 速度门。产物冻结为 `results/e3_scan/e3_memory_diagnostic.json`。

### 决策

- A0 失败：修 runner，不解释神经动力学；A0 PASS/A1 某核心失败：记录该核心的短信用分配失败；A1 PASS/B FAIL：任务泛化或优化失败；B PASS 而 64/256 INVALID：长期梯度/可寻址记忆是瓶颈，进入 S1 selective decay/gating。
- 禁止在看到阶段结果后增加 update 或降低 99%/90% 阈值；任何新预算另写预注册。

---

## 2026-07-18：E3-S0-MEM-D1 结果 — 显式 WRITE/READ 后三核心仍为 chance，继续判 INVALID

### 证据

- 正式命令：`.venv-wsl/bin/python experiments/e3_s0_marked_delay.py --output results/e3_scan/e3_s0_marked_delay.json`；墙钟约 117 秒。产物 SHA-256 `D54F6C0655308CB4D404D3196929D8B547B1986C624BDD5DF39CAE0496915327`。
- 3 seed、每模型 1,000 update 全部完成；loss finite，参数公平通过。即使 source 以 delay-specific `WRITE(payload)` 明确标记，三者 loss_last_100 仍约 `2.777≈ln(16)`。

| core | overall acc | delay 64 | delay 256 | NLL | train p50 ms |
|---|---:|---:|---:|---:|---:|
| E3-S0 scan | 0.0628 | 0.0605 | 0.0651 | 2.7752 | 6.724 |
| LSTM | 0.0625 | 0.0671 | 0.0579 | 2.7749 | **3.467** |
| Transformer | 0.0667 | 0.0703 | 0.0632 | 2.7758 | 16.155 |

### 判定

- LSTM 三项远低于预注册 90%，故 **D1 INVALID**；E3 quality 不判 PASS/FAIL，速度仍比 LSTM 慢 `1.94×`。
- D1 否定了“只因 source 未标记”这一充分解释，但仍不能区分 runner 错误、稀疏监督优化失败、长 BPTT 信号消失或模型容量不足。按预注册转入 D2 阶梯，不继续盲目增加长任务 update。

---

## 2026-07-18：E3-S0-MEM-D1 预注册 — 标记 WRITE/READ 的双寄存器长延迟诊断

### 为什么需要 D1

紧邻下方的 S0-MEM 正式结果满足预注册 INVALID 条件：LSTM、Transformer、E3 全部停在 chance。事后审计发现原任务没有标记 source，query 到来之前模型不知道 512 个随机 payload 中哪一个会被读取，实质要求 d=32 核心学习完整随机 shift register，而不是验证稀疏事件记忆。D1 不覆盖或“修好”原结果，单独回答更小的问题：核心能否在明确 WRITE(payload) 后跨 64/256 step 响应 READ(delay)。

### 冻结任务 / 判据

- `T=512`；背景只有 4 类 distractor。每条样本放一个 `WRITE64(payload)` source 和一个 `WRITE256(payload)` source（每个 delay 各有 16 个带 payload 的 source token，共 32 个 WRITE token），在精确延迟后放对应 `READ64/READ256` query；source/query 四个位置互不冲突且随 batch seed 随机。每条序列仅 2 个监督点，chance 仍为 6.25%。
- 仍用 3 seed、`B=8,D=32`、CPU threads 4、1,000 update、AdamW `1e-3`、clip 1.0；train/test seed 隔离、同批次顺序、共同 embedding/decoder 初始化、E3/LSTM/Transformer total parameters 对 LSTM 差≤2%。
- LSTM overall 与两个 delay mean 均须≥90%，否则 D1 仍 INVALID。D1 的 **S0 quality PASS** 要求 E3 三项均≥90% 且不低于 LSTM 2 个百分点；速度继续原样报告，但 D1 quality PASS 不能覆盖 S0-MEM 已失败的 `B=8,T=512` 速度门。
- 若 LSTM PASS 而 E3 FAIL，支持“additive modulo state 缺少可寻址/选择性记忆”；进入 S1 dynamic decay/gated charge。若 E3 也 PASS，才逐级增加并发 WRITE 数，而不是直接宣称 TextWorld 就绪。
- 产物冻结为 `results/e3_scan/e3_s0_marked_delay.json`。

---

## 2026-07-18：E3-S0-MEM 结果 — 三核心均为 chance，任务校验失败，按预注册判 INVALID

### 证据

- 正式命令：`.venv-wsl/bin/python experiments/e3_s0_delayed_copy.py --output results/e3_scan/e3_s0_delayed_copy.json`；总墙钟约 121 秒。产物 SHA-256 `5E528DE75C294A6C59D9BC47011AACF5468472488CCC3616BBCE633BCA7F30EF`。
- 每个模型/seed 完整消费 1,000 个相同 `B=8,T=512` train batch；参数为 E3 9,624、LSTM 9,616、Transformer 9,776，全部在 2% 内。三模型 loss 从约 2.78 收敛到约 `2.77`，即 `ln(16)=2.7726` 附近，没有隐藏发散。

| core | test accuracy mean±std | delay 64 | delay 256 | NLL | train p50 ms/update |
|---|---:|---:|---:|---:|---:|
| E3-S0 scan | 0.0640±0.0033 | 0.0646 | 0.0635 | 2.7729 | 7.103 |
| LSTM | 0.0633±0.0029 | 0.0643 | 0.0623 | 2.7728 | **3.741** |
| Transformer | 0.0673±0.0008 | 0.0658 | 0.0688 | 2.7728 | 16.132 |

chance 为 0.0625；所有结果都与 chance 相容。E3 两层 E/I spike rate 约 `0.47–0.48`，因此失败不是“完全不发放”，而是 spike/residual 没有形成可解码的延迟地址。E3 本任务训练还比 LSTM 慢 `1.90×`，speed check 同样失败。

### 判定

- **`status=INVALID`，不是 H-S0-MEM FAIL/PASS。** 预注册要求 LSTM≥80%，实际只有 6.33%；不能用 E3 与失败基线相近来宣称非劣。
- 任务审计后的解释（推断，不冒充直接证据）：source 没有 salience/write marker，模型只有在未来 query 到来时才知道应检索哪个历史 token，d=32 的三种核心在 1,000 update 内都没有学到随机 shift register。下一步用 D1 显式标记 WRITE/READ，区分“训练任务不可学”与“S0 记忆数学不足”。
- 该 INVALID 不改变已成立的速度事实：S0 prefix scan 在 T=512 相对 serial 通过并行门、T=2,048 部分线程超过 LSTM；也不改变 ANN 联合门仍失败。

---

## 2026-07-18：E3-S0-MEM 预注册 — 512-token 双延迟事件检索质量门

### 任务 / 数据隔离

- 每条序列长度 512，payload vocabulary 16；额外两个 query token 分别要求回忆 64 或 256 step 前的 payload。每条样本各放 4 个不冲突 query，target 只在 8 个 query position 上计 loss/accuracy，chance 为 6.25%。query/source position 随 batch seed 改变，不能靠固定绝对位置背答案。
- train/test 由不重叠的冻结 seed 生成；所有模型/seed 消费完全相同的预生成 train batch 顺序，数据生成不计入训练吞吐。test batch 不参与调参。
- 三个 seed；`B=8,T=512,D=32`，CPU threads 固定为 4，1,000 update，AdamW、学习率 `1e-3`、clip norm 1.0。正式运行前只允许 runner smoke 检查 shape/finite，不用 smoke 指标改预算。

### 模型公平性 / 指标

- 共同 token embedding 与 decoder 逐 seed 使用完全相同初值；仅替换 E3-S0 scan / 单层 LSTM / 单层 causal Transformer core。E3 `state_dim=27`、两层；三者 total parameter 与 LSTM 差必须≤2%。
- 主质量指标为 held-out query token accuracy，另报 delay 64/256、cross-entropy、逐 seed、mean/std。LSTM test mean 若低于 80%，判为训练预算/任务校验失败，不能用“三者都差”给 S0 过关。
- **H-S0-MEM PASS** 需要：E3 overall 与两个 delay bucket 均≥80%，且各自不低于 LSTM 2 个百分点；同时 E3 的 train p50 milliseconds/update 或等价总 wall-clock 必须低于 LSTM。只快不准、只赢一个 delay、或 Transformer 独赢都不算 S0 memory 成功。
- 训练吞吐覆盖 embedding + core + query CE + backward + clip + AdamW；报告每 update ms、sequence token/s、query/s。因完整 1,000-step wall-clock 单次样本不足以估 p50，runner 同时记录后 900 step 的逐 update timing 分布。

### 决策

- 若 E3 质量门 PASS，下一步接冻结 TextWorld next-event；若速度 PASS、质量 FAIL，直接支持 S1 dynamic decay/selective memory，不对 S0 事后加泄漏再冒充同一实验。
- 若 LSTM 自身低于 80%，本轮记为 INVALID 而非 S0 FAIL，并另行预注册更可学习的诊断预算；Transformer 结果仍保留用于确认任务是否可解。
- 产物冻结为 `results/e3_scan/e3_s0_delayed_copy.json`。

---

## 2026-07-18：E3-S0 速度结果 — exact-reset SNN 时间并行门 PASS；长序列训练首次超过 ANN，但 T=512 训推联合门仍失败（混合结果）

### 证据 / 等价性

- 正式命令：`.venv-wsl/bin/python experiments/e3_s0_scan_benchmark.py --output results/e3_scan/e3_s0_scan_benchmark.json`；墙钟约 74 秒。
- 产物 SHA-256：`25C73ADC54208BED2F68D84723E0A72D56DFA76F0712BE2817BF2D066D74F8B4`。环境仍为 PyTorch `2.13.0+cpu`、Ryzen 9 7950X、CUDA unavailable。
- 4/4 equivalence case PASS，包含 `T=512` 两层：serial/scan spike 逐元素完全相同，hard-reset residual 有界，sequence/state/input/initial-state/全参数 surrogate gradient、逐 token streaming 全部通过冻结 allclose。全 case 最大绝对差 `3.3379e-6`（发生在允许 relative tolerance 的梯度），streaming sequence 最大差 `3.5763e-7`。
- `D=32` 时搜索并冻结 E3 `state_dim=27`：E3 8,456 参数，LSTM 8,448，差 `0.095%`；E2 8,416，Transformer 8,608。E3 两层 residual state 为 432 bytes，较 LSTM/E2 的 256 bytes 多 68.75%，不能隐藏这个代价。

### T=512 时间并行硬门

`B=1,D=32` forward+backward p50 ms：

| CPU threads | E3 serial | E3 scan | scan/serial | E2 fused | LSTM | Transformer |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 104.98 | 2.419 | **43.40×** | 56.82 | **1.541** | 8.44 |
| 4 | 105.01 | 2.845 | **36.91×** | 54.64 | **2.015** | 3.43 |
| 16 | 105.42 | 4.264 | **24.72×** | 55.89 | **3.318** | 4.33 |

serial autograd node 为 9,313，scan 恒为 117，node ratio `1.256%`，远低于预注册 25%；三档线程都超过 `10×`。**H-S0-PAR PASS**，这与 F0 的 2–4× kernel 收益性质不同：时间图从 O(T) Python/autograd 链变为常数深度 prefix primitive。

绝对速度上，T=512 scan 仍比 LSTM 慢 `1.57×/1.41×/1.29×`，但已快于 Transformer（1/4/16 线程分别约 `3.49×/1.21×/1.02×`），并比 E2 fused 快 `13–24×`。

### 长序列 crossover 与 streaming 负债

T=2,048：

| CPU threads | E3 scan p50 ms | LSTM | Transformer | 判读 |
|---:|---:|---:|---:|---|
| 1 | 4.77 | **4.13** | 96.26 | LSTM 仍快 13.4% |
| 4 | **4.86** | 5.12 | 26.80 | E3 首次快 5.1% |
| 16 | **6.97** | 9.63 | 29.08 | E3 快 27.6% |

这是本项目第一次在参数匹配条件下观察到严格 spike/reset SNN 的训练 p50 超过 LSTM；但它只发生在长序列，且尚未经过任务质量门，不能外推为“替代 ANN”。

连续 step streaming p95 ms：

| CPU threads | E3 scan | E2 fused | LSTM | Transformer |
|---:|---:|---:|---:|---:|
| 1 | 0.291 | 0.13 | **0.098** | 0.22 |
| 4 | 0.296 | 0.14 | **0.118** | 0.22 |
| 16 | 0.578 | 0.21 | **0.184** | 0.36 |

S0 的单 token 路径包含两层 Python dispatch、两次 quantise/sigmoid/floor/reset 与 readout，p95 比 LSTM 慢约 `2.5–3.1×`；训练 scan 与 streaming step 没有共用最优 kernel。**H-S0-ANN FAIL**，因为预注册要求同一线程档同时赢 T=512 train 与 streaming。

### 判定 / 下一步

- **H-S0-EQ PASS；H-S0-PAR PASS；H-S0-ANN FAIL。** 数学并行路线得到强支持，工程实时推理仍未解决。
- 按预注册进入 delayed-event/copy 质量实验：若 additive hard-reset state 无法保留延迟信息，则速度成功仍是任务失败，并直接支持 S1 dynamic-decay/selective memory；若质量非劣，再接真实 TextWorld next-event。
- 同时保留一个明确优化靶：把 T=1 的量化、两层 E/I update 与 readout 融成单个 streaming kernel；在本机无 CUDA 的条件下先测 Torch compile/C++ extension 的 CPU 下界，GPU 只保留 runner，不声称已验证。

---

## 2026-07-18：E3-S0 预注册 — exact-reset cumulative-charge SNN 与时间并行训练

### 数学构造 / 边界

F0 已证明等价 kernel fusion 不能消除 E2 的非线性时间链。S0 先验证一个可精确并行、带真实离散 spike 与 hard reset 的最小 SNN 基元。对每个 E/I neuron，令归一化单步电荷 `q_t=ρ·sigmoid(d_t)`，冻结 `ρ=0.95<1`，初始膜电位 `u_0∈[0,1)`：

`Q_t = u_0 + Σ_{k≤t} q_k`；`C_t=floor(Q_t)`；`s_t=C_t-C_{t-1}`；`u_t=Q_t-C_t`。

因为每步电荷小于一个阈值，`s_t∈{0,1}`；`u_t∈[0,1)` 正是每次越阈值后减 1 的 hard-reset IF serial dynamics。训练时 `Q` 由 `torch.cumsum` 并行生成，推理时只保存 residual `u` 并 O(1) 更新。S0 是 additive affine monoid `A_t=1` 的严格特例，不宣称已经具有泄漏、选择性遗忘或同层 recurrent feedback；这些属于 S1 动态衰减/reset-correction 扩展。

为避免长序列 float32 prefix reduction、batched GEMM 与逐 token GEMV 的加法顺序在阈值边界产生不同 spike，forward 先把 drive 冻结量化到 `2^-10` 网格，再把 `q_t` 量化为 `2^-12` 网格（4,096 levels）；backward 对两次 round 都使用 straight-through identity。drive 量化误差上界为 `4.8828e-4`，单步 charge 量化误差上界 `1/(2×4096)=1.2207e-4`。在 T≤2,048 的预定范围内，二进制 charge 分数的累计值仍可由 float32 精确表示，使 scan/streaming spike 等价成为可检验的工程不变量；这是明确的数值表示选择，不冒充连续电荷方程完全无误差。

网络冻结为两层：第一层从 dense multimodal token 产生 E/I charge；第二层只接收第一层的离散 E/I spikes，使用四个 row-softmax 非负 magnitude 与固定 `E→E + / I→E - / E→I + / I→I -` 符号。最终 readout 同时读取末层 `[s_E,-s_I,u_E,-u_I]`。因此时间轴并行、层轴串行，不能把它描述成 ANN recurrent hybrid。

### 训练梯度语义

- forward 的 `floor/difference/modulo` 必须保持精确二值 spike 与 hard reset；不得用连续 activation 冒充 spike。
- backward 使用冻结的 periodic surrogate-floor：以最近整数阈值的距离构造有界 surrogate，scale `5.0`；膜电位 reset count 在 backward 中 stop-gradient，避免把 hard discontinuity 当解析导数。
- `serial` reference 与 `scan` 必须使用同一 cumulative-charge surrogate 图；另用逐 token `step` 验证实际 hard-reset streaming forward。surrogate gradient 一致性只在这一冻结训练语义内成立，不外推为生物真实性。

### 假设 / 硬门

- **H-S0-EQ**：覆盖 `B∈{1,4},T∈{1,32,512}`、1/2 层和外部 state；scan 与 serial 的 spike 必须逐元素完全相同，sequence/state/input-gradient/全部 parameter-gradient 通过 `atol=2e-6,rtol=1e-5`；full scan 与逐 token hard-reset streaming 同样通过，且所有 spike 只含 0/1、所有 residual 在 `[0,1)`。
- **H-S0-PAR**：`B=1,T=512,D=32` 的 scan/serial forward+backward p50 加速至少 `10×`；autograd node 随 T 不得线性增长到 serial 的 25% 以上。
- **H-S0-ANN**：相同输入/输出维度、batch、T、loss 和线程设置下，S0 scan train p50 至少在一档预注册线程数达到 LSTM；连续 step streaming p95 也必须在同一档达到 LSTM。否则不得宣称 S0 已达到 ANN 训推速度。
- **H-S0-MEM**：在冻结 delayed-event/copy task 上，参数量差≤2%，S0 的 held-out token accuracy 不低于 LSTM 2 个百分点，且 T=512 train p50 更快；再进入真实 TextWorld next-event。真实任务仍须同时报告 LSTM/Transformer，S0 若只快不准则失败。

### 实验顺序 / 产物

1. 先实现 `E3CumulativeScanCore`、serial/scan/step 与严格单测；EQ 失败则停止速度/质量宣称。
2. 运行 1/4/16 CPU thread；CUDA runner 必须支持 synchronize 与 peak memory，但本机 CUDA unavailable 时明确记 null。microbenchmark 产物冻结为 `results/e3_scan/e3_s0_scan_benchmark.json`。
3. 只有 H-S0-EQ 通过后才运行 synthetic memory；只有 H-S0-PAR 或 H-S0-ANN 至少一项显示方向性收益后才花预算接 TextWorld/HomeGrid。
4. S0 之后五条互不混写的路线依次为：S1 dynamic-decay affine scan + reset correction；PRF 共轭振荡 scan；FPT 固定点时间并行；eligibility/online local gradient；exact sparse-event forward/backward。每条都另写预注册和负面结果。

---

## 2026-07-18：E2-F0 结果 — 等价融合成立，但实现速度门与 ANN 训推速度门均失败（明确负面结果）

### 证据 / 可复现性

- 正式命令：`.venv-wsl/bin/python experiments/e2_f0_fusion_benchmark.py --output results/e2_acceleration/e2_f0_fusion_benchmark.json`；墙钟约 56 秒。
- 产物：`results/e2_acceleration/e2_f0_fusion_benchmark.json`，SHA-256 `12D34D225A41FD7FC8E6F6244A8D7E00D52D98075C1521C91B212F9ABB26CB19`。runner 记录基线 commit `67d657767f1c2469cbba956533b0d4b56cd8b06b` 及精确 dirty-file 列表。
- 环境：WSL2、Python 3.12.3、PyTorch `2.13.0+cpu`、oneDNN enabled、Ryzen 9 7950X、32 logical CPU；CUDA unavailable，GPU 指标为 null/unavailable，没有用 CPU 结果冒充 GPU。
- 等价矩阵 8/8 case PASS，覆盖冻结的四类配置与 `(B,T)=(1,1)/(4,32)`；sequence/state/input/state-gradient/全部 parameter-gradient、streaming、参数键/shape/count/state bytes 均通过 `atol=2e-6, rtol=1e-5`。全矩阵最大原始绝对差为 `2.0266e-6`。

### Core forward+backward：图变小，但 T=32 的硬门未达到

`B=1,T=32,D=32`；p50 ms。E2 参数/state 分别与 reference 完全相同（8,416 / 256 bytes），autograd node 从 `1,294` 降到 `616`（减少 52.4%）。

| CPU threads | E2 reference | E2 fused | fused speedup | LSTM | Transformer |
|---:|---:|---:|---:|---:|---:|
| 1 | 9.016 | 3.663 | 2.461× | 0.548 | 0.947 |
| 4 | 9.303 | 3.541 | **2.627×** | 0.556 | 1.047 |
| 16 | 21.803 | 7.870 | 2.770× | 0.975 | 1.818 |

- canonical T=32 的最好加速只有 `2.770×`，没有达到预注册 `4×`。16 线程反而比 1/4 线程慢，说明许多小 GEMM 之间仍被 Python/非线性时间依赖串行，线程调度开销超过单步并行收益。
- 在长序列、16 线程上，reference/fused 相对加速到 T=128 的 `4.355×`、T=512 的 `4.612×`，但绝对 p50 仍为 `19.65/71.09 ms`；同条件 LSTM 为 `2.25/4.64 ms`。因此更大的相对加速只是 amortize 旧实现冗余，不能掩盖 fused E2 仍慢 `8.72×/15.32×`。

### 完整 HomeGrid 训练步与 streaming

训练步范围为共同 encoder/heads + forward + 冻结 weighted CE + backward + clip-grad + AdamW.step；`B=1,T=32,D=32`。

| CPU threads | E2 reference p50 ms | E2 fused p50 ms | speedup | LSTM p50 ms | Transformer p50 ms |
|---:|---:|---:|---:|---:|---:|
| 1 | 12.967 | 7.524 | 1.723× | **4.186** | 4.628 |
| 4 | 13.636 | 8.100 | 1.684× | **4.515** | 4.935 |
| 16 | 23.573 | 11.822 | **1.994×** | **5.684** | 7.644 |

16 线程的最好值 `1.994×` 仍低于预注册 `2×`，不向上取整为 PASS。fused E2 在三档线程上分别比 LSTM 慢约 `1.80×/1.79×/2.08×`。

连续 core streaming p95（ms）如下；所有模型使用同 token 流与有界显式 state。

| CPU threads | E2 reference | E2 fused | LSTM | Transformer |
|---:|---:|---:|---:|---:|
| 1 | 0.137 | 0.149 | **0.142** | 0.323 |
| 4 | 0.180 | 0.176 | **0.153** | 0.338 |
| 16 | 0.356 | 0.403 | **0.400** | 0.793 |

F0 的 sequence-level hoist 在 T=1 streaming 无法摊销，fused 并不稳定快于 reference；三档均未同时赢 LSTM 的完整训练步与 streaming p95。

### 判定 / 数学方向

- **H-F0-EQ PASS；H-F0-IMPL FAIL；H-F0-ANN FAIL。** fused 可作为等价默认执行图保留，但不能宣称已经达到 ANN 训推速度。
- 负面结果定位很明确：softmax/input/readout hoist 与 signed block 已移除约一半 autograd 图并取得 1.68–2.77× 实际收益；剩余 6.7–38× core 差距来自每个 token 都必须等上一个 sigmoid/Euler state 的非线性时间链。继续做相同层级的 kernel 拼接不会把复杂度变成时间并行。
- 下一轮进入 E3，而不是微调 F0 门槛：构建带离散 spike/reset 与 surrogate-gradient 语义的 signed selective affine recurrence，把 subthreshold dynamics 写成 associative pair `(A_t,b_t)` 并用 prefix scan 训练；推理仍保持 O(1) state。先在合成 copy/delay 与真实 token next-event 上验证 scan/serial 等价、梯度、速度和记忆质量，再升级到 HomeGrid 多模态 rollout。

---

## 2026-07-18：E2-F0 预注册 — 等价 signed-block 融合、CPU 多核缩放与 ANN 速度硬门

### 动机 / 研究问题

E2-M0 的正式训练吞吐只有 LSTM 的 26.2%，但当前 `E2SignedCore` 每个 token 都在 Python 中重复计算四次整矩阵 softmax、两次输入投影、四次 recurrent GEMM 和一次逐 token readout；LSTM 则使用融合内核。因此，本轮先回答一个受控问题：**在不改变 E2 的方程、参数、状态、训练数据或优化器的前提下，等价图融合能消除多少实现性开销，剩余差距是否仍要求改变时间动力学的数学形式？**

这不是“纯 SNN 已成立”的证据：E2 仍是连续 sigmoid E/I recurrence。F0 只清除实现混杂，为后续真正离散 spike/reset/surrogate-gradient 且可时间并行的 E3 提供可信基线。

### 冻结改动（仅执行图，不改模型数学）

1. 每次 sequence forward 只计算一次四个 row-softmax channel；参数及梯度路径全部保留。
2. 把两路 input projection 在完整 `[B,T,D]` 上预计算，再按时间索引读取。
3. 令 `z=[E,I]`，把四个有符号通道组成单个 `2D×2D` block weight：`[[gEE·WEE, -gIE·WIE], [gEI·WEI, -gII·WII]]`，每个 micro-step 用一次 recurrent GEMM；不删除 gain 为零的参数。
4. 循环内只保留不可避免的非线性状态更新；先收集 E/I state，再对完整序列批量执行 LayerNorm 与 output projection。
5. 保留可调用的 reference execution mode，用相同 state dict 做逐项等价验证和交错计时；公开默认改为 fused。`policy/no_positive/state_reset/micro_steps/detach_state/streaming step` 语义不得改变。

### 预注册假设与判据

- **H-F0-EQ（必要门）**：覆盖 `exact/margin/hybrid`、`no_positive`、`state_reset`、`micro_steps∈{1,2}`、初始/外部 state、full-sequence/逐 token streaming。float32 的 sequence、末状态、输入梯度与每个参数梯度均须通过标准 `torch.allclose(atol=2e-6, rtol=1e-5)`，并另存未经隐藏的 `max_abs`；所有参数键、shape、数量和 state bytes 必须完全一致。梯度 probe 冻结为逐元素线性权重的 mean-normalized sequence loss，加 `0.17×` 末状态 E/I mean 差，避免绝对梯度误差随 `B×T×D` 任意缩放。任一失败则 F0 REJECT，不能报告速度收益。
- **H-F0-IMPL**：`B=1,T=32,D=32` 的 core forward+backward fused/reference p50 加速至少 `4×`，完整 synthetic HomeGrid train-step 至少 `2×`；两项分别须在至少一个预注册线程设置上达到。未达到则记录负面结果，不调低门槛。
- **H-F0-ANN**：在至少一个相同的预注册线程设置上，fused E2 必须同时达到 LSTM 的 train-step p50 和 streaming inference p95，才可宣称“仅工程融合已达到 ANN 训推速度”。只赢 Transformer、只赢单项或从不同线程设置拼接胜项均不得判 PASS。
- 即使 H-F0-ANN 通过，也不能宣称 SNN 替代 ANN；还必须在 E3 以后满足真实语言、多模态、action-conditioned rollout 与闭环任务质量门。

### 测量设计 / 公平性

- 环境冻结：WSL Ubuntu、Python 3.12.3、PyTorch `2.13.0+cpu`、oneDNN 可用；主机 Ryzen 9 7950X，16 physical / 32 logical core。当前无 `nvidia-smi` 且 `torch.cuda.is_available=False`，故本机 GPU 结果必须写成 unavailable；runner 仍实现 CUDA synchronize/peak-memory 路径，留待 GPU 环境复跑。
- CPU 线程为 `1/4/16`；至少覆盖 `(B,T,D)=(1,32,32),(8,32,32),(1,128,32),(1,512,32)`。先 warmup，随后按固定 seed 交错执行 reference/fused/LSTM/causal-Transformer，报告 p50/p95、tokens/s、speedup、autograd node 数；CUDA 可用时额外报告峰值显存。
- 所有实现使用相同输入 tensor、初始 state 与标量 loss；每次 backward 前清梯度。禁止在看到结果后改变重复次数、线程集合或成功阈值。
- 产物冻结为 `results/e2_acceleration/e2_f0_fusion_benchmark.json`；正式命令和实际环境/耗时在结果段补录。

### 预先决定的后续分支

- H-F0-EQ 通过后才允许把 fused 设为默认；H-F0-ANN 若失败，差距归入“非线性时间递归”而不是继续微调 benchmark。
- 下一数学主线固定为 E3：signed selective affine scan（训练期 prefix-scan / 推理期 O(1) state），并保留离散 spike/reset 与 surrogate-gradient 语义；并行振荡 PRF、固定点 FPT、eligibility/local-gradient、exact sparse-event 作为独立分支，必须分别预注册、分别与 LSTM/Transformer 比质量和训推速度，禁止把混合 ANN 内核当最终 SNN 成功。

---

## 2026-07-18：E2-M0 结果 — 官方 HomeGrid 管线 READY；E2 仅有短视距变化预测信号，未成为世界模型首选（混合/负面结果）

### 官方数据、provenance 与 READY gate

- 官方 `homegrid-dynamics 0.1.1`、Gym `0.26`；严格隔离的 train/valid/test 为 `32/8/8` episode、`3072/768/768` transition，没有 synthetic/fallback。train/test action phase 分别为 `2,656/664`，test changed patch 为 `8,735`；超过预注册的 `2,000/1,000` 门槛。
- RGB 按预注册的 `12×12` patch、64 类冻结量化进入模型；train-only language vocabulary 为 26，三个 split 当前/下一 language OOV 都为 0。summary、manifest、transitions 的 SHA-256/size、版本、seed、episode 连续性和统计均由 fail-closed loader 复验。
- train 中 reward 三类 `0/0.5/1` 都存在，故 reward loss 启用；done 只有 0 类，按预注册禁用。test 没有非零 reward 或 done，因此 test reward accuracy 即使为 1.0 也只是全零类命中，**不构成奖励预测成功证据**。
- 三模型的训练、held-out、rollout、streaming 指标均有限；三个 gate 全通过，正式 `pipeline_status=READY`。该状态只证明多模态动作条件实验管线可用，模型排序不参与 READY 判定。

### 参数与训练公平性

每个模型/seed 都按同一 episode 顺序训练 3 epoch，共消费 `9,216` transition；参数 spread 为 `0.0537%`，通过 2% 门槛。

| 核心 | 参数 | weighted train loss | train transition/s |
|---|---:|---:|---:|
| LSTM | 357,697 | **2.4412** | **5,941** |
| causal Transformer | 357,857 | 2.4683 | 5,166 |
| E2 hybrid-0.8 | 357,665 | 2.6338 | 1,555 |

E2 reference kernel 的训练吞吐仅为 LSTM 的 26.2%、Transformer 的 30.1%，即慢约 `3.82×/3.32×`；这是延续 TextWorld 的明确工程负债。

### 一步预测：能识别变化，但没有稳健的 E2 优势

三 seed test mean 如下；changed macro-F1 只对 target 中实际出现的类做宏平均。

| 核心 | visual overall acc | changed acc | changed NLL | changed macro-F1 | next-language acc |
|---|---:|---:|---:|---:|---:|
| LSTM | 0.5830 | 0.3063 | 1.7307 | **0.2202** | 0.7053 |
| causal Transformer | **0.5874** | 0.2980 | 1.7276 | 0.2144 | **0.7305** |
| E2 hybrid-0.8 | 0.5132 | **0.3115** | **1.7032** | 0.1969 | 0.3194 |

- “复制当前帧” baseline 的 overall accuracy 为 `0.9210`、changed accuracy 为 `0`；train 全局频率 baseline overall 为 `0.2206`。三模型确实学到了复制 baseline 完全不会的变化信号，但当前 decoder/瓶颈没有保住大量静态背景，因此 overall 远低于简单复制。
- E2 changed accuracy 均值只比 LSTM 高 `0.00515`（0.515 个百分点），逐 seed 赢 2/3；changed NLL 最低，但 macro-F1 反而最差。证据更像对少数高频变化类的集中预测，而非稳健的变化动力学领先，不能据此宣称 E2 胜出。
- E2 next-language accuracy 只有 `0.3194`，显著低于 LSTM/Transformer 的 `0.7053/0.7305`；这直接否定了当前融合头已经形成统一视觉—语言状态的说法。

### 受控 action-conditioned rollout：一步信号没有延伸为长时世界模型

下表每格为 `overall / changed accuracy`。rollout 从真实 anchor 出发，后续 action、language、read flag 仍使用真实序列，只递归回输预测视觉，所以它不是自主规划或闭环任务成功率。

| horizon | LSTM | causal Transformer | E2 hybrid-0.8 |
|---:|---:|---:|---:|
| 1 | 0.5334 / 0.3509 | **0.5368** / 0.3356 | 0.4517 / **0.3564** |
| 3 | **0.4808 / 0.3666** | 0.4790 / 0.3577 | 0.4506 / 0.3544 |
| 5 | **0.4679 / 0.3614** | 0.4678 / 0.3541 | 0.4473 / 0.3468 |
| 10 | 0.4619 / **0.3629** | **0.4623** / 0.3533 | 0.4409 / 0.3385 |

E2 只在 horizon 1 的 changed mean 第一；到 3/5/10 步均由 LSTM 的 changed accuracy 领先。horizon 10 上 E2 三个 seed 都没有赢，且均值最差，因此没有长时想象优势。

### 完整 transition 实时性

96-step history、batch 1，计时覆盖共同视觉/语言/action encoder、时序核心和所有 heads，而非只测 core。

| 核心 | p50 / p95 / p99 ms | transition/s | core state bytes |
|---|---:|---:|---:|
| LSTM | **0.308 / 0.395 / 0.516** | **3,422** | **256** |
| causal Transformer | 0.556 / 0.729 / 0.894 | 1,889 | 24,576 |
| E2 hybrid-0.8 | 0.319 / 0.583 / 0.780 | 2,940 | **256** |

- E2 对 Transformer 的 p95 低 `19.93%`，略低于方向性 H-RT 的 20% 线；state 少 `98.96%`。但 H-RT 还要求质量非劣，而且正式历史门槛是 2048，不是本轮 96，所以不能判 PASS。
- E2 对 LSTM 的 p95 高 `47.5%`，state 同为 256 bytes；Transformer 对 LSTM 的 p95 高 `84.3%`、state 为 96 倍。本轮实际 transition 上，LSTM 是更强的实时基线。

### 判定 / 下一步

- **E2-M0 PIPELINE READY；H-WM 与 H-RT 均不通过。** 正面结果仅限于：E2 以固定小状态获得一步 changed accuracy/NLL 的微弱均值信号，并相对 tiny Transformer 大幅节省状态；负面结果是该信号不稳定、macro-F1 最差、语言融合明显落后、3–10 步 rollout 不领先、训练很慢且实时性输给 LSTM。
- 继续保留三条核心：LSTM 作为当前主基线与可部署候选；Transformer 作为语言/长上下文上界但需解决 KV 成本；E2 只作为待验证的生物约束 recurrent 分支，不再默认升级为主干。
- M1 不允许在 M0 上事后调参冒充复现。下一轮应预注册结构修复：显式 `next = current + sparse change` 的残差视觉头、保留 `12×12` 空间 latent 而非一次 flatten 到 32 维、把 counterfactual action ranking 和真正闭环任务成功率作为首要指标；仍须同时跑 LSTM/Transformer/E2，并把 horizon 10 与完整 transition 延迟作为硬约束。

### 可复现信息

- 正式结果：`results/e2_world_model/homegrid_dynamics_pilot_s0_s1_s2.json`；数据：`results/e2_world_model/homegrid_dynamics/`。
- 命令：`.venv-wsl/bin/python experiments/e2_homegrid_world_model.py --corpus-dir results/e2_world_model/homegrid_dynamics --output results/e2_world_model/homegrid_dynamics_pilot_s0_s1_s2.json --seeds 0 1 2 --d-model 32 --visual-embedding-dim 8 --num-heads 4 --sequence-length 32 --epochs 3 --learning-rate 0.001 --cache-window 128 --streaming-warmup-steps 32 --streaming-steps 64 --rollout-horizons 1 3 5 10`。
- 环境：WSL Ubuntu，Python 3.12.3、PyTorch 2.13.0+cpu、HomeGrid 0.1.1、Gym 0.26、NumPy 2.5.1；正式运行约 93 秒，无异常。

---

## 2026-07-18：E2-T0′ 结果 — provenance 闭环通过；质量完全复现，LSTM 延迟优势翻转暴露计时噪声（混合结果）

### 证明链与 READY gate

T0′ 按紧邻下方预注册只增加 fail-closed provenance/decision，没有改变模型或预算。新 runner 在训练前逐项验证：TextWorld `1.7.0`、冻结且跨 split 唯一的 6 个 seed、split summary 对 manifest/episodes/token-events 的 SHA-256/size、manifest 对真实 `.z8` 的 SHA-256/size、6 个 episode 的 `won=True/return=1.0/game_sha`，以及 event header 对 episode seed/split/count。所有检查通过。

- available shifted target：train `10,979`、valid `2,750`、test `2,782`；100-update 单遍训练实际消费 `6,345`，没有 cycle/repeat；valid/test 均全量消费。
- `official_dataset_provenance_verified`、冻结 seed、episode boundary、跨模型/seed等量消费、held-out 完成、所有指标有限等 7 项 gate 全为 true。
- **正式 `pipeline_status=READY`**；该状态只证明官方事件 LM 数据/训练管线可用，不改变“尚未验证结构化世界预测与规划”的边界。

### 确定性质量复现与计时复跑

三模型 valid/test NLL、PPL、逐 seed 值与 provenance-incomplete T0 产物**逐位一致**：LSTM test `37.19±0.98`、Transformer `74.68±6.50`、E2 `67.76±2.11`。因此 T0 的负面质量结论被正式确认：LSTM 三 seed 全胜，E2 比 LSTM 高 82.2%，只在平均上比 tiny Transformer 低 9.3%。

| 核心 | verified train token/s | verified stream p50 / p95 ms | state bytes |
|---|---:|---:|---:|
| LSTM | 33,599 | 0.148 / **0.192** | 256 |
| causal Transformer | 27,253 | 0.359 / 0.453 | 28,160 |
| E2 hybrid-0.8 | 2,184 | **0.145** / 0.257 | 256 |

- E2 对 Transformer 仍有清楚的流式结构优势：p95 低约 43%、状态少 99.1%；但对 LSTM，p50 只低约 2.6%，p95 反而高约 34%。
- 完全相同质量复跑中，T0 原始 p95 曾显示 E2 `0.274` vs LSTM `0.303`，T0′ 变为 E2 `0.257` vs LSTM `0.192`。**LSTM/E2 尾延迟排序翻转**，证明 100 次、固定模型顺序的 sub-ms CPU microbenchmark 不足以声称二者谁更实时；M0 必须测完整 transition，并把重复/交错计时列为后续工程修复。
- E2 verified 训练吞吐仍仅为 LSTM 的约 6.5%、Transformer 的约 8.0%，负面 kernel 结论稳定。

### 决定

- T0′ 正式恢复 T0 的 **EVENT PIPELINE READY**，同时把旧结果 JSON 永久保留为 provenance-incomplete 审计案例；后续引用质量时以 verified 产物为准。
- 不再把本轮 E2-vs-LSTM sub-ms p95 当结构收益。当前唯一稳定的实时证据是 recurrent state 相对 Transformer KV 的固定内存，以及 E2 相对本轮 Transformer 的延迟；LSTM 仍是质量、状态和工程效率的最强基线。
- 继续执行已预注册的 HomeGrid M0；其 changed-patch 与 rollout 结果决定是否值得进入 planner，而不是由 event PPL 或一次 microbenchmark 决定。

### 可复现信息

- verified 产物：`results/e2_world_model/textworld_event_lm_pilot_s0_s1_s2_verified.json`；旧产物不覆盖。
- 与 T0 相同命令，仅 output 改为 `_verified.json`；运行前 provenance hard gate、运行后 `pipeline_status=READY`，无异常。

---

## 2026-07-18：E2-T0′ 预注册 — TextWorld provenance 闭环修复（预注册）

### 触发证据

T0 模型数值、episode reset 和 token 公平检查已完成，但发布前红队审计发现结果 runner 只读取 split manifest 字段与 manifest 自身 SHA，没有把 `summary.json → episodes.jsonl/token_events.txt/manifest.json → .z8` 的 SHA/size、官方版本、精确 seed、获胜 episode 和 event header 串成闭环。现有真实文件可由人工逐项核对，但原结果 JSON 中的 `synthetic=false/fallback=false` 尚不能由 runner 自证；因此 T0 的质量数字保留为 provenance-incomplete 观测，PIPELINE READY 暂缓为正式判定。

### 冻结唯一修复

- 不改变语料、tokenizer、词表、模型、初始化、E2 策略、seed、100/50 step 预算、优化器、KV window 或指标；只增加 fail-closed provenance 与 READY 状态计算。
- 强制 TextWorld `1.7.0`；精确 seeds 为 train `{20260718..20260721}`、valid `{20260722}`、test `{20260723}`，跨 split 唯一。
- 逐 split 校验 summary 中 manifest/episodes/token-events 的 path、SHA-256、size；manifest 中每个真实 `.z8` 的 path/SHA/size；episodes 与 manifest 的 seed/split/game SHA 一致且 `won=True, return=1.0`；event episode header 与 episodes 的 seed/split/count 一致。任一缺失、篡改、错误版本或 fake fixture 都硬失败。
- 结果新增 available shifted train targets、实际 consumed targets 和 `pipeline_status`。只有 provenance 全通过、available targets ≥10,000、episode/reset/参数公平通过且全部训练/held-out/streaming 数值有限时才写 `READY`；该状态仍只指数据/事件 LM 管线，不等于 H-WM。

### 复跑与接受边界

- 使用与 T0 完全相同命令/预算写新产物 `results/e2_world_model/textworld_event_lm_pilot_s0_s1_s2_verified.json`，不覆盖 provenance-incomplete 原产物。
- 质量 NLL/PPL 应在确定性 CPU 容差内复现；吞吐/尾延迟允许受系统噪声变化。若质量排序改变或 provenance 失败，记录为 REVISE，不回写旧结果。

---

## 2026-07-18：E2-T0 结果 — 官方 TextWorld 事件流管线 READY，但 LSTM 质量显著领先（混合/负面结果）

### 真实数据与公平性证据

- 官方 TextWorld `1.7.0` `tw-coin_collector` level 5 共 6 个 seed-disjoint 游戏：train 4、valid 1、test 1；全部由官方 interpreter 执行 `extra.walkthrough` 并以 `won=True, return=1.0` 结束。每个真实步骤另由 `Environment.copy()` 记录最多 2 个候选动作的真实反事实，没有 replay/synthetic fallback。
- 事件语料为 train `4 episode / 10,983 token`、valid `1 / 2,751`、test `1 / 2,783`；train-only vocab 344，corpus fingerprint `10e4599b792fe59c7af779bb38c010a44692410dbde76fd9d87f290ac54e7e34`。episode 首 chunk reset 审计全部通过，chunk 不跨 episode。
- 三模型每 seed 均训练 100 update / `6,345` target token；valid `2,750`、test `2,782` target 全量评估。总参数 LSTM `19,864`、Transformer `20,024`、E2 `19,832`，spread `0.9645%`，通过 2% 门槛。

### 三 seed 原始汇总（mean；括号为 population std）

| 核心 | valid PPL | test PPL | train token/s | stream p50 / p95 ms | 110-token state bytes |
|---|---:|---:|---:|---:|---:|
| LSTM | **35.66 (1.06)** | **37.19 (0.98)** | **32,067** | 0.179 / 0.303 | 256 |
| causal Transformer | 72.45 (6.36) | 74.68 (6.50) | 24,938 | 0.366 / 0.459 | 28,160 |
| E2 hybrid-0.8 | 65.52 (2.19) | 67.76 (2.11) | 1,913 | **0.172 / 0.274** | 256 |

- LSTM 在三个 seed 的 test PPL 均第一；E2 相对 LSTM 平均高 `82.2%`，明确不满足 5% 非劣解释。这个负结果必须随 E2 一起保留，不能只引用 WikiText 上约 1% 的均值优势。
- E2 平均 test PPL 比 Transformer 低约 `9.3%`，但只在 seed 1/2 更好；seed 0 为 Transformer `66.65`、E2 `69.66`。因此只支持“小数据事件流中 E2 平均优于本轮 tiny Transformer”，不支持普遍架构排序。
- 流式 reference 实现中，E2 p95 比 Transformer 低约 `40.2%`、状态少 `99.1%`（256 vs 28,160 bytes），但 LSTM 状态同为 256 bytes，且 E2 对 LSTM 的 p95 优势仅约 `9.5%`、逐 seed 只赢 2/3。
- E2 训练吞吐仅为 LSTM 的约 `6.0%`、Transformer 的约 `7.7%`，即慢约 `16.8×/13.0×`；这是比 WikiText 更明显的 kernel 工程负债。LSTM 吞吐跨首次运行有较大系统噪声，但不影响数量级结论。

### 判定 / 决定

- **E2-T0 EVENT PIPELINE READY；质量结果为负面。** 官方数据超过 10,000 train token，三核心有限训练并完成 held-out 游戏评估，数据/状态/公平性管线达到预注册 READY；但 E2 未接近最佳 LSTM，不能据此升级 H-WM 或声称动作世界建模有效。
- 当前 event LM 把目标、观察、动作、反事实、奖励等序列化后做 teacher forcing；它验证真实任务语言事件兼容性，却没有单独测 next-state、reward/done、counterfactual ranking 或闭环规划。下一阶段不能再用事件 PPL 代替世界模型指标。
- **保留 LSTM 为强基线**：M0 HomeGrid 必须同时比较 changed-patch、开放环 rollout 与整 transition 延迟；若 E2 仍明显落后 LSTM，即使比 Transformer 省 KV 内存，也不能作为首选基底。

### 可复现信息

- 数据产物：`results/e2_world_model/textworld_l5/{train,valid,test}/`；结果：`results/e2_world_model/textworld_event_lm_pilot_s0_s1_s2.json`。
- 命令：`.venv-wsl/bin/python experiments/e2_textworld_lm.py --corpus-dir results/e2_world_model/textworld_l5 --output results/e2_world_model/textworld_event_lm_pilot_s0_s1_s2.json --seeds 0 1 2 --d-model 32 --num-heads 4 --batch-size 1 --sequence-length 64 --steps 100 --eval-steps 50 --learning-rate 0.001 --cache-window 128 --streaming-warmup-steps 10 --streaming-steps 100`。
- 环境：WSL Ubuntu，Python 3.12.3、PyTorch 2.13.0+cpu、TextWorld 1.7.0；运行完成无异常。相关事件/CLI 测试 `9 passed, 3 subtests`。

---

## 2026-07-18：E2-M0 预注册 — HomeGrid 官方多模态动作条件视觉动力学 pilot（预注册）

### 任务选择与边界

按已确认的“语言校准 → 文本动作世界 → 多模态世界”路线，Gate M 首个任务冻结为官方 HomeGrid `0.1.1` 的 `homegrid-dynamics`，而不是序列分类或自造 gridworld。该环境同时给出 `96×96×3` RGB、逐 token 动力学语言、动作、奖励与终止；M0 只验证**动作条件下一视觉/语言预测和有限开放环想象**，尚无统一 planner、任务成功率或自主选择 imagined action，因此无论结果多好都不判完整 H-WM，也不把内部激活称为“思考”。

### 冻结真实数据

- split 按环境 seed 严格隔离：train `2026071800..2026071831`（32 episode），valid `2026071900..2026071907`（8），test `2026072000..2026072007`（8）；每 episode 最多 96 个官方 transition，真实终止则提前停止。
- 动作仅由独立的 `random.Random(seed + 1_000_003).randrange(10)` 产生，禁止从 observation/test 指标选择动作；HomeGrid 环境 RNG 仍走已审计的 `0.1.1` seed 兼容路径。记录 preread `is_read_step=True` 与真实 action phase，分层报告，不把 preread 中被环境忽略的动作混作动力学证据。
- RGB 只通过冻结的无学习编码进入模型：把 `96×96` 切为 `12×12` 个 `8×8` patch，每通道均值按 `[0,64,128,192,256]` 量化为 4 档，合成 `r*16+g*4+b` 的 64 类视觉 token。每帧保存原始 RGB SHA-256 和 144 token；结果目录不保存/提交原始 RGB，也不允许 synthetic fallback。
- 保存 current/next visual token、当前/下一官方 language token、human-readable language、read flag、action、reward、terminated/truncated；manifest 固化 HomeGrid/Gym/Python/NumPy 版本、量化定义、seed mode、artifact SHA-256 和各 split 的 episode/transition/read/action/changed-patch/reward/done 计数。

### 冻结模型与训练

- 三模型只替换时序核心：一层 stateful LSTM、一层 causal Transformer（真实 KV cache，window 128）、E2 signed E/I `hybrid + positive_factor=0.8`；`d_model=32`，共同视觉 token/patch-position encoder、train-only language-token vocabulary、action/read embeddings、输出 LayerNorm 和 next-visual/next-language/read/reward-done heads。
- 总参数（含所有共享形式的 encoder/head）spread ≤2%；模型 seed `{0,1,2}`。每个 episode 以 sequence length 32 顺序训练，batch 1、3 个 epoch；episode 首 chunk reset，chunk 间保留并 detach state，禁止跨 episode。AdamW `lr=1e-3`；每个模型消费完全相同的 transition 次序与数量。
- 总 loss 冻结为 next-visual 144 patch 平均交叉熵 + `0.25×` next-language CE + `0.10×` next-read CE；reward/done 仅在训练 split 同时存在正负/多类别时各加 `0.10×`，否则标为不可判而不制造类别。禁止依据 test 排名重调 loss、量化阈值或模型宽度。

### 指标与判定

- 视觉：test overall/changed/unchanged patch NLL 与 accuracy、macro-F1；必须同时给“复制当前帧”与 64 类频率基线，overall 高分不得掩盖 changed-patch 失败。
- 语言/事件：next-language accuracy、next-read accuracy；reward/done 只有类别可识别时报告 Brier/accuracy，并按 preread/action phase 分层。
- 想象：在 test action phase 从真实 anchor 出发，后续 action 与 language token 固定为真实序列，仅把模型预测视觉递归回输，报告 horizon `{1,3,5,10}` changed/overall accuracy。它是受控 action-conditioned visual rollout，不是自主规划。
- 实时：batch=1 整个 transition update 的 p50/p95/p99、transitions/s 与 core state bytes；96-step history 仍不足 H-RT 的 2048 门槛，只作工程数据。
- **M0 PIPELINE READY** 只要求：官方数据与 split/边界验证通过、train action-phase ≥2,000 transition 且 test changed patch ≥1,000、三模型 loss/held-out/rollout/latency 全为有限值。模型排序与 E2/LSTM 非劣仅作为下一轮 planner/反事实任务的设计证据，不自动判 H-WM/H-RT。

### 预定产物

- 数据清单：`results/e2_world_model/homegrid_dynamics/`，真实轨迹缓存可放 ignored `data/e2_world_model/homegrid_dynamics/`。
- 比较结果：`results/e2_world_model/homegrid_dynamics_pilot_s0_s1_s2.json`。

---

## 2026-07-18：E2-T0 预注册 — TextWorld 官方事件流 LM 校准（预注册）

### 目的与严格边界

E2′ 已修复共享 LM 输出仪器并完成 WikiText pilot（见下条）。进入完整 H-WM 前，先用官方 TextWorld Coin Collector 检验同一 causal core 是否能消费“目标—观察—可行动作—反事实—真实动作—下一观察—奖励—终止”的事件流。该实验仍是 **teacher-forced event LM 校准**：它是实际 LLM-agent 任务数据，但不等于下一状态结构化预测、自由 rollout、统一 planner 或闭环成功率，因而无论结果多好都不能判 H-WM。

### 冻结数据

- 官方 TextWorld `1.7.0`、challenge `tw-coin_collector`、level 5；只调用虚拟环境同目录的 `tw-make`，生成 `.z8` 后由官方 interpreter 执行 `extra.walkthrough` 并要求终局 `won=True`。
- 按游戏 seed 隔离：train `{20260718,20260719,20260720,20260721}`，valid `{20260722}`，test `{20260723}`；禁止同 seed 跨 split。
- 每个真实步骤保存当前 observation、admissible actions、walkthrough action、next observation/reward/done，并用 `Environment.copy()` 记录最多 2 个候选动作的真实反事实；禁止 replay 或自造环境 fallback。
- 数据集逐 split 保存 manifest、游戏 SHA-256、canonical JSONL 和 token-event text。LM corpus 以 `<|episode|>` 开头严格切 episode，train-only 建 vocab；每个 episode 首 chunk reset state，chunk 不跨 episode。

### 冻结比较

- 复用 E2′ 已接受的共同 wrapper：`Normal(0,d^-0.5)` tied embedding + shared output LayerNorm。
- LSTM / causal Transformer / E2 `hybrid + positive_factor=0.8`；`d_model=32`、1 layer、Transformer KV window 128、参数 spread ≤2%。
- seed `{0,1,2}`；batch 1、sequence length 64；对 train 事件流做一个确定性 pass、最多 100 update；valid/test 各最多 50 chunk。AdamW `lr=1e-3`，数据顺序、step 和 token 数必须逐模型相同。
- 报告 train/valid/test token-weighted NLL/PPL、训练 token/s、streaming p50/p95/p99 与状态 bytes；数据过小或某 split 不足预算时按实际 token 数报告，不重复 episode 填满预算。

### 判定

- 本轮不设 H-WM PASS。若三模型都能有限 loss 训练并完成真实 held-out game event PPL，则事件表示/训练管线 **READY**；若任一核心数值失败、episode reset 泄漏或真实数据不足 10,000 train token，则 **PIPELINE REVISE**。
- LSTM/Transformer/E2 排序只作为下一轮结构化 next-state / reward-done / counterfactual ranking 设计证据；不得声称规划或世界模型已验证。

### 预定产物

- 数据：`results/e2_world_model/textworld_l5/`；游戏二进制只缓存于 ignored `data/e2_world_model/textworld/games_l5/`。
- 结果：`results/e2_world_model/textworld_event_lm_pilot_s0_s1_s2.json`。

---

## 2026-07-18：E2′ 结果 — 共享尺度修复通过；WikiText pilot 中 E2≈LSTM、Transformer 较差（正面但非确认性）

### 仪器接受

按紧邻下方的 E2′ 预注册，只改变共同 LM wrapper 后，训练前同一真实 batch 的尺度为：

| 核心 | embedding std | hidden std | logits std | initial NLL |
|---|---:|---:|---:|---:|
| LSTM | 0.1768 | 0.9989 | 1.0187 | 8.695 |
| Transformer | 0.1765 | 1.0001 | 1.0150 | 8.813 |
| E2 | 0.1764 | 1.0000 | 1.0175 | 8.539 |

`log(4096)=8.318`；三者 NLL 最大差 0.273 nat、logits std 比 1.004，全部通过预注册门槛。加入共同 LayerNorm 后总参数为 LSTM `143,680`、Transformer `143,840`、E2 `143,648`，spread `0.001336`（0.134%）。因此 E2′ 仪器 ACCEPT，随后原预算重跑有效；P0 的极端 Transformer PPL 被确认是读出尺度伪差异。

### 修复后原始结果（3 seed mean；括号内为 population std）

| 核心 | valid PPL | test PPL | train token/s | stream p50 / p95 ms | 80-token state bytes |
|---|---:|---:|---:|---:|---:|
| LSTM | 255.91 (4.80) | 273.38 (3.04) | 17,404 | 0.180 / 0.234 | 256 |
| Transformer | 347.62 (11.96) | 372.84 (15.68) | 18,454 | 0.342 / 0.430 | 20,480 |
| E2 hybrid-0.8 | 261.23 (3.97) | 270.71 (11.92) | 5,042 | 0.251 / 0.322 | 256 |

- test PPL：E2 比 LSTM 低约 0.98%，但逐 seed 为 E2 `[253.85,279.23,279.04]`、LSTM `[269.11,275.05,275.98]`，只在 seed 0 更好；valid 上 E2 比 LSTM 高 2.08%。因此当前证据是**两者近似、E2 满足 5% 非劣 pilot 线**，不是 E2 稳定胜出。
- Transformer 在尺度修复后从 P0 的约 31,529 恢复至 372.84，证明修复有效；它仍比 LSTM/E2 高约 36–38%，但该结论只适用于 143k 参数、25.6k 训练 token 的小数据 pilot，不外推到成熟 Transformer 或 LLM 规模。
- 训练效率：Transformer≈18.45k、LSTM≈17.40k、E2≈5.04k token/s；当前 reference E2 kernel 慢约 3.5–3.7 倍，是明确工程负债。
- 流式：E2 p95 比 Transformer 低约 25%，但比 LSTM 高约 37%；E2/ LSTM 的 state 都为 256 bytes，Transformer 在 80-token 历史为 20,480 bytes。E2 只相对 Transformer有内存/延迟优势，不支配 LSTM。

### 判定 / 决定

- **E2′ SCALE INSTRUMENT ACCEPT；WikiText Gate-L pilot 支持 E2 非劣，但不作 confirmatory PASS。** 三 seed、真实 test、参数与 token 公平下，E2 与 LSTM 同一量级，满足继续进入动作条件任务的最低语言兼容条件。
- **H-RT 未判定**：只测 80-token history，不是预注册 2048；且 E2 虽快于 Transformer，却慢于 LSTM。不得用 256 vs 20,480 bytes 单独宣称总体实时胜出。
- **效率结论保留负面**：E2 训练吞吐明显落后；下一阶段必须同时保留 LSTM 基线，不能只与 Transformer 比内存。
- 下一步执行 E2-T0 官方 TextWorld 事件流校准；通过后再增加结构化 next-state/reward-done/counterfactual ranking 与闭环 planner，才触及 H-WM。

### 可复现信息

- 原始结果：`results/e2_world_model/wikitext_pilot_scale_fixed_s0_s1_s2.json`；仪器：`results/e2_world_model/wikitext_scale_diagnostic.json`。
- 配置除共同 wrapper 两项修复外与 P0 完全一致；WSL Python 3.12.3、PyTorch 2.13.0+cpu、NumPy 2.5.1、16 CPU threads。

---

## 2026-07-18：E2′ 预注册 — 共享 LM 输出尺度仪器修复（预注册）

### 触发证据 / 唯一问题

紧邻下条 E2-P0 在真实 WikiText-2 上满足数据、参数量、token 和 seed 公平，但未训练诊断显示共享 tied embedding 仍使用 PyTorch `nn.Embedding` 默认 `std≈1`。Transformer 的末层 LayerNorm 使 hidden `std≈1.000`，从而得到 logits `std=5.679`、初始 NLL `26.278`；LSTM 的 hidden `std=0.145`，初始 NLL 仅 `8.573`。因此 P0 的 Transformer test PPL≈31,529 主要混入了**共享读出仪器对核心输出尺度不等价**，不能作为架构结论。

### 冻结修复与禁止项

E2′ 只允许修改所有核心共用的 `CausalLanguageModel` 包装器：

1. tied embedding 初始化改为宽度感知的 `Normal(0, d_model^-0.5)`，不再使用 `std≈1`；padding 行保持 0；
2. 在共同 LM head 前增加同一个可训练 `LayerNorm(core.output_dim)`，使 LSTM、Transformer、E2 进入 tied head 前使用同一种输出尺度仪器。

三模型都增加相同形式的 LayerNorm；tokenizer、WikiText archive、4096 train-only vocab、模型宽度、核心结构、E2 `hybrid/positive_factor=0.8`、seed `{0,1,2}`、每模型 100 step/25,600 target token、AdamW `lr=1e-3`、valid/test 20 batch、KV window 128 与 streaming 预算全部不变。禁止根据 E2′ 结果再选择 embedding std、额外温度、学习率或核心宽度。

### 仪器接受与结果边界

- 训练前、同一首 batch 上，三模型初始 NLL 均须落在 `log(4096) ± 1.0 nat`，且最大两两差 ≤0.5 nat；否则 E2′ 仍以仪器失败停止。
- 三模型初始 logits std 均须在 `[0.5,1.5]`，且最大/最小 ≤2；参数量 spread 仍须 ≤2%。
- 仪器通过后才运行与 P0 完全相同的三 seed pilot。它仍标为 `pilot_not_confirmatory`：只比较修复后的 test PPL、训练 tokens/s、流式 p50/p95/p99 和当前 80-token 状态字节；不据此判 H-LM/H-RT，也不把 80-token cache 结果外推为 2048-token 门槛。
- 若 E2′ Transformer 恢复而排序改变，P0 永久保留为无效仪器负例；不得删除或回写成成功实验。

### 预定产物

- 代码：`vpsc/world_model/lm.py` 及对应单元测试。
- 结果：`results/e2_world_model/wikitext_pilot_scale_fixed_s0_s1_s2.json`；另保存训练前尺度诊断。

---

## 2026-07-18：E2-P0 结果 — 真实 WikiText pilot 运行完成，但共享输出尺度仪器失败（无架构判定）

### 数据与公平性证据

- 数据为 WikiText-2 raw，archive SHA-256 `ef7edb566e3e2b2d31b29c1fdb0c89a4cc683597484c3dc2517919c615435a11`、4,721,645 bytes；原 MetaMind S3 URL 当日返回无可用跳转的 HTTP 301，改用 ggml-org Hugging Face 上**同 byte size、同 SHA-256**镜像，内容判据未改变，且没有 synthetic fallback。
- train-only regex word+punct vocab 4096；实际 token 数 train `2,158,836`、valid `225,370`、test `254,046`。
- seed `{0,1,2}`；每模型每 seed 100 step × batch 4 × length 64 = 25,600 target token；valid/test 各 20 batch。
- 总参数：LSTM `143,616`、Transformer `143,776`、E2 `143,584`，relative spread `0.001337`（0.134%），通过 2% 公平门槛。
- E2 冻结低增益策略为 `hybrid + positive_factor=0.8`，有效增益 `E→E=5.89, I→E=0, E→I=9.5, I→I=5.985`。

### 原始结果（3 seed mean；仅作故障定位）

| 核心 | test PPL | train token/s | streaming p50 / p95 ms | 80-token 后状态 bytes |
|---|---:|---:|---:|---:|
| LSTM | 1,075.7 | 18,336 | 0.129 / 0.211 | 256 |
| Transformer | 31,528.7 | 19,305 | 0.275 / 0.488 | 20,480 |
| E2 hybrid-0.8 | 877.3 | 5,193 | 0.118 / 0.239 | 256 |

E2 的 pilot test PPL 数值最好、Transformer 最差约 29 倍；但以下训练前诊断否定了把该排序解释为时序核心能力：

| 核心 | embedding std | hidden std | logits std | initial NLL |
|---|---:|---:|---:|---:|
| LSTM | 0.9999 | 0.1451 | 0.8340 | 8.573 |
| Transformer | 1.0021 | 1.0001 | 5.6789 | 26.278 |
| E2 | 0.9985 | 0.7020 | 3.9978 | 13.892 |

### 判定 / 决定

- **E2-P0 INSTRUMENT REJECT；不作 LSTM/Transformer/E2 架构判定。** tied embedding 的默认 `std≈1` 与各核心天然输出幅度相乘，导致模型在第一个优化步骤前就处于完全不同的 softmax 温度；“共享同一个 head”在数值仪器上并不等于公平。
- P0 仍保留两个可用工程观察，但都不是预注册 H-RT 结论：当前 reference kernel 中 Transformer 训练吞吐最高，E2 训练吞吐最低；80-token 流式状态下 recurrent state 为 256 bytes，Transformer KV 为 20,480 bytes。历史尚未达到 2048，禁止对 H-RT 判定。
- 开 E2′ 单一仪器修复：所有模型共同采用宽度感知 embedding 初始化与共同输出 LayerNorm；其余预算冻结后原样重跑。

### 其他真实任务可运行证据（尚非训练结果）

- TextWorld 1.7.0：官方 `tw-coin_collector` level 5 / seed 20260718 成功生成 `.z8`，SHA-256 `93e02d9fcd540040d29f52160c2046c09b4ad9d546f4c9b4924b4853745f06a4`；真实 interpreter reset、`Environment.copy()` 三个候选反事实和 live step 均通过。
- HomeGrid 0.1.1：四个官方 ID 均完成固定 seed reset/step，观察含 `96×96×3` RGB、语言 token/embedding；两个独立 `homegrid-dynamics` 实例在 seed 20260720 + action 3 下初始/下一图像哈希、语言与 reward 完全一致。
- Messenger 未安装；官方旧 Gym/Python/SDL 栈留作独立兼容环境，不以自造任务替换。

### 可复现信息

- 命令：`python experiments/e2_world_model.py wikitext-pilot --cache-dir data/e2_world_model/wikitext2 --seeds 0 1 2 --d-model 32 --num-heads 4 --vocab-size 4096 --batch-size 4 --sequence-length 64 --steps 100 --eval-steps 20 --learning-rate 0.001 --cache-window 128 --e2-policy hybrid --positive-factor 0.8 --streaming-warmup-steps 16 --streaming-steps 64`。
- 原始产物：`results/e2_world_model/wikitext_pilot_s0_s1_s2.json`；环境产物：`results/e2_world_model/environment_probe.json`。
- 环境：WSL Ubuntu，Python 3.12.3，PyTorch 2.13.0+cpu，NumPy 2.5.1，16 CPU threads；运行无异常。

---

## 2026-07-18：E2 预注册 — 统一 token-event 流式语言世界模型（预注册）

### 背景 / 动机

用户目标不是再验证一个序列分类器，而是判断 VPSC/E1 的有符号反馈机制能否成为可演化到**多模态、实时响应、可通过内部世界预测进行思考**的模型基底。因此 E2 将比较对象改为真实语言模型与动作条件世界模型任务，并把“思考”冻结为可被环境核验的 imagined rollout / 反事实预测是否改善行动，而不是生成的 chain-of-thought 是否流畅。

本条只预注册问题、协议和判官，**不包含 E2 结果**；首次结果必须另起日志条目，失败、依赖不兼容和未完成阶段同样记录。

### A–E 可继承的正面效果与不可越界结论

| 阶段 | 可继承的成功效果 | 本轮不得扩大解释的边界 |
|---|---|---|
| A | 小温度时转移算子可逼近恒等映射（A2 `||S-I||_max=1.01e-6`），大温度时行分布可逼近平稳分布（误差 `5.59e-8`）；环形拓扑谱隙 `0.0979` 小于稠密拓扑 `0.8213`，提供较慢混合的结构候选 | A1 任务结果无效；A3 的拟合 `R^2<0`，没有证明持续模态、功能记忆或任务收益 |
| B | STDP 时间窗拟合 `R^2=0.82, tau≈4`；纯生成自由能单调；深层任务出现 `beta*=0.80`、`beta_c≈0.81`；MNIST 达 `96.95%`；Fixation 的 Poisson/静态输入分别由 `59.01/52.96%` 提升至 `96.45/96.40%`；训练约比 CNN 快 `2.2x` | MNIST 仍落后 CNN/MLP，且推理更慢；本轮必须同时报告质量、延迟、吞吐和内存，不能只挑训练速度 |
| C | 完整 E/I 环在冻结协议下形成有界振荡：`late_std=0.3206`、谱纯度 `0.9596`、状态范围 `[0.0136,0.9152]`，局部网格 `21/27` 通过 | “正反馈=记忆、负反馈=门控”的强命题 `0/27`，不得作为已证机制；E2 必须用消融重新建立任务级因果 |
| D | D2 信息峰 `1.736 bit`；去正反馈在 30/30 信息案例中使校正 MI 为 0，支持“正反馈是信息载体”的局部工程结论；负反馈作用随正反馈增益分区翻转；D3 的敏感区稳定落在 `beta*rho∈[0.95,1.00]` | D1 cue probe 及 shuffle/ablation 都接近 1，属于被动残留；D2 拓扑置换 29/30 保持；D3 两类回响时间仪器均失败，精确临界也从未严格最优 |
| E | E1 外推 60 案例通过：hybrid 名义 median `1.663` vs exact `1.583`；漂移最坏 median `1.489` vs `1.241`，崩溃 `4` vs `12`；去正反馈全为 0。分层上 `g_ee×0.8` 与 `×1.0` 有效，`×1.25` 有害 | E1 只证明冻结 MI 协议上的工程组合，不证明语言建模或世界建模；E2 必须冻结按层策略并分层报告，禁止 pooled 结果掩盖高增益失败 |

E2 的最小新意是把 E1 规则实现为**显式分离的 excitatory / inhibitory 动态状态和有符号通道**，而不是在现有单一对称 `W_rec` 上贴标签。冻结策略为：低增益层使用 `hybrid = 0.95 margin + 移除负反馈`，中增益层使用 `0.95 margin/full E/I`，高增益层保留 `exact/full E/I`。

### 研究问题与预注册假设

- **H-LM（语言兼容）**：在共享 tokenizer、embedding、LM head 和数据顺序下，E2 能完成真实语料的因果 next-token 建模；test perplexity 不比最佳 LSTM/Transformer 基线差超过 5%。
- **H-WM（动作条件世界预测）**：在 TextWorld 的未见游戏上，E2 对下一观察、奖励、终止和动作后果的预测，以及统一 planner 下的闭环成功率，不比最佳基线差超过 3 个百分点。
- **H-RT（流式结构收益）**：在质量非劣前提下，E2 至少满足一项：2048-token 历史下 batch=1 的 p95 在线更新延迟降低 ≥20%；在线状态/解码缓存降低 ≥30%；或长轨迹闭环成功率提高 ≥5 个百分点。
- **H-MECH（机制归因）**：`no_positive` 或 episode 中途 `state_reset` 至少一项必须显著破坏长程预测、反事实排序或闭环表现；否则即使 E2 数值更好，也不得把收益归因于 A–E 的正反馈载体机制。
- **总边界**：只通过 WikiText 不能称为世界模型；通过 TextWorld 仍只支持文本世界模型；至少在 HomeGrid 或 Messenger 的官方多模态任务上复现世界预测/规划收益后，才允许称为“多模态世界模型技术基底候选”。

### 冻结任务路线与 token-event 表示

1. **Gate L — WikiText-2 causal LM 校准**：使用真实 WikiText-2 raw train/valid/test，词汇仅由 train 构建；禁止 synthetic fallback。先做可复现 CPU pilot，再在资源允许时扩大预算。
2. **Gate T — TextWorld 动作条件因果预实验**：使用官方 TextWorld 生成器/环境和未见游戏；输入目标、文本观察、上一动作和奖励，预测下一观察、奖励、done、可行动作及候选动作后果；报告 teacher-forcing 与自由闭环。
3. **Gate M — HomeGrid / Messenger 多模态主实验**：沿 Dynalang 的官方任务接口，把文本、视觉/符号观察、动作、奖励串成同一事件流；只替换时序核心。若 Windows/依赖阻断，明确记为 BLOCKED/未运行，不能换成自造 gridworld 后仍称官方任务。

统一逻辑流冻结为：

```text
<goal> text
<obs:text> ... <obs:vision> latent_tokens
<prev_action> ... <reward> ...
<predict:next_obs> ... <predict:reward_done> ... <predict:next_action> ...
```

WikiText 只使用其中的文本子流；后续任务追加模态 token，而不更换核心接口。

### 模型、资源公平与禁止项

- 三个核心：stateful LSTM、causal Transformer、E2 signed E/I recurrent core；共享 tokenizer、输入 embedding、位置/模态编码方案、输出 heads、优化器、训练样本顺序和 planner。
- 主比较匹配**总参数量（含 embedding/head）±2%**、训练 token/transition 数、BPTT/attention 窗口、超参数搜索次数和随机种子 `{0,1,2}`；另报告实测训练时间、吞吐和峰值内存。若 FLOP 不能同时严格匹配，明确列为限制，不用“等参数”冒充“等计算”。
- Transformer 增量推理必须使用真实 KV cache；主实时泳道使用与 LSTM/E2 在线状态字节匹配的固定/滑动窗口，完整上下文 Transformer 仅作为单独上界，不与固定内存结论混池。
- LSTM/E2 必须跨 chunk 保留状态；只在真实 document/episode 边界 reset。训练截断长度与 Transformer 窗口相同。
- 同时报告 reference PyTorch 实现与可用优化实现的边界；若 E2 没有优化 kernel，不能把成熟 Transformer kernel 的差异直接解释为算法差异，反之亦然。
- 禁止：静默 synthetic fallback、从 test 建词表/调增益、只做 teacher forcing、把生成文本的可读性当作思考、按测试地图切换 E2 策略、只报 pooled 均值或单 seed。

### 指标、分层与判官

- **语言**：train/valid/test NLL 与 perplexity；128/512/2048 历史长度的退化；连续流式与每 chunk reset 对照。
- **世界预测**：下一观察 token NLL/F1、reward/done Brier 或 calibration、1/3/5/10 步开放环误差、同一状态下候选动作的反事实排序。
- **规划**：统一 planner 下闭环成功率、步数、非法动作率、无 imagination 与 1/3 步 imagination 的 planning gain，以及 predicted/realized return gap。
- **实时性**：TTFT；逐事件/逐 token p50、p95（必要时 p99）延迟；稳态 tokens/s；峰值 CPU/GPU 内存；持久状态/KV-cache 字节。
- **分层**：任务/游戏难度、轨迹长度、预测 horizon、正常/OOD、E2 增益层分别给出；`g_ee×1.25` 必须独立列出，不得由低增益层抵消。
- **结论规则**：H-LM 失败则停止世界模型主张但仍记录负结果；H-LM 通过后进入 H-WM；H-WM 与 H-RT 同时通过且 H-MECH 有效，才把 E2 提升为多模态 Gate M 候选。LSTM 与 Transformer 的真实任务对比无论 E2 成败都必须保留。

### 最小实现与产物

- 代码：`vpsc/world_model/`；统一入口预定为 `experiments/e2_world_model.py`。
- 原始产物：`results/e2_world_model/` 下的配置、数据清单/SHA-256、逐 seed JSONL、汇总 JSON 和实时基准；图表只能由原始结果生成。
- 测试：离线 tiny fixture 只验证 tokenizer、mask、状态连续性、KV cache、E/I 符号约束和适配接口；它不替代真实任务结果。
- 首轮运行环境现状：系统 Python `3.12.7` 未安装 PyTorch；已有 `atri` 环境为 Python `3.13.2`、`torch 2.7.0.dev20250209+cpu`、16 CPU threads。显卡为 AMD Radeon RX 7800 XT，但该环境只暴露 CPU。正式运行前必须把实际解释器、依赖版本、设备与命令写入结果条目。

---

## 2026-07-17：E1 结果 — D2 载体规则 × D3 5% 裕量外推通过（正面结果，有限采用）

### 背景 / 动机

执行紧邻下条的 E1 预注册。案例格只从 D2 旧产物冻结；判官数据全部来自 D2/D2′ 未使用的动态 seed `{3,4,5}` 与两套新刺激库，因此本条检验的是组合规则的外推，而不是在旧结果上重打分。

### 结果（原始运行）

60 个外推案例 × 三种全局增益漂移（$-5\%,0,+5\%$），预注册判官三项全通过：

| 判据 | 预注册门槛 | 结果 | 判定 |
|---|---:|---:|---|
| E1-U：名义效用 | hybrid 信息案例率 ≥80%；median ≥ exact−0.10 bit | **60/60 = 100%**；1.663 vs 1.583 bit | PASS |
| E1-R：漂移最坏值 | worst median ≥ exact+0.10 bit；崩溃数不增加 | **1.489 vs 1.241（+0.248 bit）**；4 vs 12 | PASS |
| E1-C：正反馈载体 | ≥90% 满足 no-positive ≤0.5×hybrid | **60/60 = 100%**；no-positive MI 全为 0 | PASS |

**总判定：E1 ADOPT。** 未进入总判据的置换 sanity 也通过：名义 `hybrid` 的 raw MI 在 60/60 案例中均高于各自 16 次 label-shuffle 最大值。

### 分解：成功来自“组合”，不是单一 0.95 缩放

- pooled 名义 median：`exact_full=1.583`、`margin_full=1.459`、`hybrid=1.663`。**只加 5% 裕量会降低名义中位 MI**；加入 D2 的负反馈分区规则后才超过原始完整 E/I。
- pooled 漂移 worst-case median：`exact_full=1.241`、`margin_full=1.366`、`hybrid=1.489`；崩溃案例数依次为 12、10、4。0.95 裕量贡献一部分稳健性，分区规则进一步减少崩溃。
- 按 $g_{ee}$ 行分解（括号为“名义 median / 漂移最坏 median”）：
  - $g_{ee}\times0.8$：exact `1.583/1.243` → hybrid **`1.696/1.565`**；关闭负反馈是主要增益来源。
  - $g_{ee}\times1.0$：exact `1.536/1.208` → hybrid **`1.470/1.446`**；名义略降 0.066 bit，但最坏值提高 0.238 bit，符合安全裕量的效用—鲁棒权衡。
  - $g_{ee}\times1.25$：exact `1.612/1.373` → hybrid **`1.413/1.018`**；5% 裕量在这一行反而有害。该行只有 6/60 案例，不足以推翻预注册 pooled 判官，但构成明确部署边界。
- 名义逐案例 `hybrid > exact` 为 30/60；因此 ADOPT 不是“每一点都更优”，而是跨冻结案例分布的整体非劣与最坏漂移改善。

### 观察与解释

- **观察**：D2 的“低正反馈区关闭负反馈”规则在两套新刺激、三个新动态 seed 上稳定复现；$g_{ee}\times0.8$ 行的 hybrid 名义 MI 均值约 1.69，而单纯 margin 在 `(0.8,1.0)` 一列几乎崩溃。
- **观察**：在 $g_{ee}\times1.0$ 行，0.95 缩放牺牲少量名义 MI，换得更高 worst-case；这是 D3 混合证据首次转化为可量化的工程安全裕量收益。
- **解释**：正反馈提供信息放大/传播，负反馈不是固定必需部件而是随正反馈增益切换的调制器；5% 裕量对中等增益区抑制参数漂移跨界有效，但在更高 $g_{ee}$ 区会把系统推离其信息工作区。
- **重要边界**：Wilson–Cowan 的统一增益缩放不等于 D3 的严格 $\beta\rho$；E1 证明的是这个冻结工程映射有效，不是 D3 回响假说复活。

### 结论 / 决定

- **按预注册判官采用 E1**：将“正反馈保留 + $g_{ee}\times0.8$ 关闭负反馈 + 0.95 全局反馈裕量”作为冻结 10 格分布上的组合实践候选。
- **实际部署采取更保守边界**：$g_{ee}\in\{0.8,1.0\}$ 可进入下一阶段任务验证；$g_{ee}=1.25$ 暂保留 exact_full，不把本轮 pooled ADOPT 外推成高增益通用默认。
- **D2/D3 原结论不变**：D2、D2′、D3、D3′ 仍为 REJECT；成功的是从其混合结果中抽取边界后建立的独立 E1 工程策略。
- 下一步若继续，应检验 E1 是否改善真正任务指标（而不只是冻结 MI 协议），并把 `g_ee=1.25` 的例外作为预注册分层而非事后删除。

### 可复现信息

- 命令：`python3 lab/ring_feedback/e1_hybrid_margin.py`。
- 原型：`lab/ring_feedback/e1_hybrid_margin.py`；产物：`lab/ring_feedback/results/e1_hybrid_margin.{json,png}`。
- JSON SHA-256：`8a7cd3e168ec7c1505a44dfbaa231c238bbf166b117da3987a909bf94138a569`。
- 运行：CPU，NumPy float64，RK4 `dt=0.04`，60 外推案例；运行无报错。

---

## 2026-07-17：E1 预注册 — D2 载体规则 × D3 近临界安全裕量（预注册）

### 背景 / 动机

D2 与 D3 方向均已按各自迭代上限关闭，不能再事后改判据寻找“成功”。但两条线留下了可组合、且没有被反证的工程边界：D2′ 的 30 个信息案例中，去正反馈 30/30 使 $\mathrm{MI}_{corr}=0$；负反馈在 $g_{ee}\times0.8$ 区压制信息、在 $g_{ee}\geq1.0$ 区成为必要调制器。D3/D3′ 则在 3/3 seed 上都没有发现精确临界严格占优，$\chi$ 与一阶收缩最慢点集中在 $\beta\rho\in[0.95,1.00]$，但两种回响时间仪器均失败。因此本条不恢复 D2/D3 原命题，而把这些**混合结果中的正面边界**压缩成一个可部署候选：保留正反馈载体、按正反馈增益开关负反馈，并留 5% 全局反馈增益裕量。

### 假设与成功标准

- **E1（组合实践）**：相对“D2 原始完整 E/I、名义增益 1.00”，以下固定策略在新 seed 与新刺激上保持信息效用，并提高对 $\pm5\%$ 全局反馈增益漂移的最坏情形鲁棒性：
  1. 正反馈始终保留；
  2. $g_{ee}$ 因子为 0.8 时关闭 E←I 负反馈（`g_ei=0`），$g_{ee}\geq1.0$ 时保留负反馈；
  3. 四个反馈增益 `g_ee/g_ei/g_ie/g_ii` 统一乘 0.95；外部输入、阈值、时间常数不变。
- **固定案例格**：D2 旧产物中 seed 平均 $\mathrm{MI}_{corr}\geq0.5$ 的 10 个格；案例选择只用旧产物，不看本轮结果。
- **真正外推集**：未在 D2/D2′ 使用的动态 seed `{3,4,5}` × 两套新的 8-pattern 刺激库（seed `20260718/20260719`），共 60 个（格、动态 seed、刺激库）案例；每个 pattern 仍激活 8 节点中的 3 个，且库内组合唯一。响应窗、4-bin 量化、4 instance、16 次 label-shuffle 校正全部复用 D2。
- **增益漂移**：$\delta\in\{-0.05,0,+0.05\}$ 同乘四个反馈增益。比较 `exact_full`（名义尺度 1.00、完整 E/I）、`margin_full`（尺度 0.95、完整 E/I）与 `hybrid`（尺度 0.95 + 上述负反馈规则）；`no_positive` 仅在名义漂移 0 下作载体负面对照。
- 预注册判据：
  - **E1-U 效用非劣**：名义漂移下，`hybrid` 的 60 案例中 $\mathrm{MI}_{corr}\geq0.5$ 的比例 $\geq80\%$，且 pooled median 不低于 `exact_full` pooled median 0.10 bit 以上（`median_hybrid >= median_exact - 0.10`）。
  - **E1-R 漂移鲁棒**：逐案例取三种漂移中的最小 MI；`hybrid` 的 worst-case pooled median 至少比 `exact_full` 高 0.10 bit，且 worst-case MI<0.5 的崩溃案例数不多于 `exact_full`。
  - **E1-C 载体对照**：名义漂移下，$\geq90\%$ 案例满足 $\mathrm{MI}_{no\_positive}\leq0.5\,\mathrm{MI}_{hybrid}$。
- 总判定：E1-U、E1-R、E1-C 全通过 → **ADOPT**；E1-U 与 E1-C 通过但 E1-R 失败 → **MIXED（可用但 5% 裕量未证明鲁棒收益）**；E1-U 或 E1-C 失败 → **REJECT**。

### 边界

- 0.95 是由 D3 混合证据提出的**工程候选裕量**，不是已经证明的最优临界点；Wilson–Cowan 增益统一缩放也只是 D3 `$\beta\rho$` 的工程类比，不声称两套动力学数学等价。
- E1 是新实践验证，不重开 D2/D3，不改变它们的 REJECT 结论；即使 E1 通过，也只支持这套冻结协议下的信息效用与参数漂移鲁棒性，不支持注意力、涌现或生物回响声称。

### 可复现信息

- 预定原型：`lab/ring_feedback/e1_hybrid_margin.py`。
- 预定产物：`lab/ring_feedback/results/e1_hybrid_margin.{json,png}`。
- 运行：CPU，NumPy float64，RK4 `dt=0.04`。

---

## 2026-07-17：D3′ 结果 — 松弛回响仍不可测，D3 关闭；近临界收缩/χ 形状保留为工程线索（负面结果）

### 背景 / 动机

执行紧邻下条的 D3′ 预注册实验。这是 D3 方向约定的唯一迭代轮；本条之后 D3 关闭。

### 结果（原始运行）

- 3 个训练网的 $\beta_c$ 与 D3 首轮一致：seed 0/1/2 分别为 0.9333、0.9249、0.8128。
- **$\tau_{relax}\equiv1$**：3 seed × 7 个 $\beta\rho$ 点全部在第一次松弛迭代即跌破 $1/e$；3 个去递归对照同样均为 1。按预注册仪器验证，松弛时间仍不可测。
- Q1 平台仅 seed 0 通过；Q2 因常数序列 3/3 均记“不可测/失败”；Q3“精确临界不严格占优”3/3 通过；精确临界严格占优 0/3。
- 判官：Q1∧Q2∧Q3 为 **0/3**；REJECT-rule 为 **2/3**；**D3′ REJECT**。
- 未进入判据的连续诊断仍有一致形状：第一步扰动比在 seed 0/1 于 $\beta\rho=1.00$ 最大（0.3484/0.3475），seed 2 于 0.95 最大（0.3332）；$\chi$ 峰位相同（seed 0/1 在 1.00，seed 2 在 0.95）。但它们不能替换已经冻结的 $\tau_{relax}$ 判据。

### 观察与解释

- **观察**：D3 的跨时间步回响与 D3′ 的离散首达时间都退化为常数 1；完整网和去递归对照相同。两次独立仪器均无法建立“长回响平台”。
- **观察**：Q3 连续两轮均为 3/3，且 0/3 出现精确临界严格占优；有效的 $\chi$/一阶收缩形状把敏感区限定在 0.95–1.00，而非超临界区。
- **解释**：当前 `leak=1.0 + n_relax=8` 架构是每时间步快速固定点求解器，不是回响储备池。可保留的工程启发是“近临界但不越界的增益裕量”，不是“次临界回响态已被证明”。

### 结论 / 决定

- **不采用 D3′，D3 方向关闭**：D3 REJECT、D3′ REJECT，迭代额度已用完。
- **禁止事后换连续衰减拟合重新判通过**；第一步收缩与 $\chi$ 只作为后续独立工程实验的候选信号。
- 将 0.95 作为安全裕量候选，与 D2 的“正反馈载体 + 负反馈分区调制”组合成新 E1 实践；E1 有独立预注册、外推 seed/刺激与漂移判据，不回写 D2/D3 结论。

### 可复现信息

- 命令：`python3 lab/criticality/d3prime_relax_tau.py`。
- 原型：`lab/criticality/d3prime_relax_tau.py`；产物：`results/d3prime_relax_tau.{json,png}`。
- 运行：CPU，PyTorch，seed 0/1/2；未修改 D3 训练与前向管线。

---

## 2026-07-17：D3′ 预注册 — 松弛尺度的回响平台 vs 精确临界（预注册）

### 背景 / 动机

紧邻下条 D3 首轮结果：跨时间步回响在 leak=1.0 架构中不存在（$\tau\equiv 1$ 且对照相同，仪器失效）；$\chi$ 通道有效（峰在 $\beta\rho\in[0.95,1.00]$）；任务/信息通道在 1/3 seed 有真实信号且峰在次临界 0.95。D3′ 把回响仪器替换为**松弛迭代尺度的收缩测量**（定点映射的临界减速——在该架构中唯一可操作的"回响"定义），其余协议与判据完全复用 D3 首轮。这是 D3 方向按约定仅有的一轮迭代；本轮结束后 D3 关闭。

### 假设与成功标准

- 命题与判据结构同 D3 首轮，**唯一变更是回响时间定义**：
  - **新仪器 $\tau_{relax}$（松弛尺度临界减速）**：在每 (seed, $\beta\rho$) 点，取正常前向最后时间步的顶层状态 $m_0$（近似定点）与该步顶层输入 $I$；从 $m_0$ 加扰动 $\delta$（范数 0.01，8 个固定随机方向），对 $m_0$ 与 $m_0+\delta$ 同步施加与层内一致的松弛映射 $m\leftarrow\tanh(\beta(m W_s + I-\theta))$（leak=1.0），记录 $d_k=\mathrm{mean}\|m_{pert}(k)-m_{base}(k)\|$（8 次迭代 × 8 方向 × 全部 300 测试样本）；$\tau_{relax}$ = $d_k/d_0$ 首次 $<1/e$ 的 $k$（未达则取 8）。
  - **Q2 的 Spearman 改作用于 $\tau_{relax}$ 的次临界点**（$\beta\rho\in\{0.80,\dots,0.95\}$，阈值 $\geq 0.8$ 不变）。
- Q1（平台）、Q3（精确临界不严格占优）、acc/MI/$\chi$ 通道、verdict 规则（$\geq 2/3$ seeds）与全部容差同 D3 首轮，逐字不变。
- **仪器验证**（预注册，非判据）：去递归对照（$\mathrm{W_{rec}}=0$）的 $\tau_{relax}$ 应恒为 1（映射与状态无关，扰动一步消失）。若完整网的 $\tau_{relax}$ 同样恒 1 或不随 $\beta\rho$ 呈系统趋势，则宣布该架构在两个时间尺度上回响均不可测，D3 以"仪器失效"为最终结论关闭。
- 常数序列的秩相关伪影（首轮 Q2 的 +1.000）在本轮显式处理：计算前检查序列方差，方差 $<10^{-12}$ 时 Q2 记为"不可测"（不算通过）。

### 边界

- 与 D3 首轮相同：工作区形状之争，不涉及涌现/功能声称，不构成 C3。
- $\tau_{relax}$ 单位为松弛迭代（1–8），是"回响"在该架构中唯一可操作的定义；结果条目连同此定义陈述，不外推为生物时间尺度的回响。

### 可复现信息

- 预定原型：`lab/criticality/d3prime_relax_tau.py`（复用 `lab/criticality/d3_reverberation.py` 的训练与前向管线）。
- 预定产物：`results/d3prime_relax_tau.{json,png}`。
- 运行：CPU，PyTorch，seed 0/1/2。

---

## 2026-07-17：D3 结果 — 预注册判官输出 REJECT，但 3/5 通道仪器失效；唯一有效通道支持临界 χ 峰（混合结果）

### 背景 / 动机

执行紧邻下条的 D3 预注册实验。本条报告原始结果，并如实标注仪器失效通道；D3 方向约定的一轮迭代（D3′，修复回响仪器）见后续条目。

### 结果（原始运行）

3 个训练网（CE 100 epoch，$\beta_{train}=0.8$）：seed=0 $\beta_c=0.933$，seed=1 $\beta_c=0.925$，seed=2 $\beta_c=0.813$（$\beta_c$ 随训练权重变化，与 deep_critical 的 0.81 同区）。

逐点摘要（acc / $\mathrm{MI}_{corr}$ / $\chi$ / $\tau_{rev}$ / drop）：

- **acc 通道弱**：seed 0/1 全部 $\beta\rho$ 点上 acc∈[0.223, 0.300]（chance=0.25），信号贴近噪声底；seed 2 有真实信号，acc 峰 0.437 在 **$\beta\rho=0.95$（次临界）**，$\beta\rho=1.00$ 时降至 0.320。
- **$\mathrm{MI}_{corr}$ 通道弱**：除 seed 2 在 $\beta\rho=0.95$ 的 0.214 bit 外，全部点 ≤0.1 bit。
- **$\tau_{rev}\equiv 1.00$**：全部 seed、全部 $\beta\rho$、以及去递归对照（$\mathrm{W_{rec}}=0$）——跨时间步扰动在一个时间步内消失，与递归无关。**回响仪器在此架构（leak=1.0、每时间步 8 次全松弛）下不可测**。
- **drop≡0.000**：与 $\tau\equiv 1$ 一致，扰动恢复通道无内容。
- **$\chi$（唯一有效通道）**：seed 0/1 随 $\beta\rho$ 升至峰于 $\beta\rho=1.00$（0.86→1.13）后于超临界点回落；seed 2 峰于 0.95。方向与 Theorem 3 的临界磁化率一致。

判官输出：seed 0 判 pass（但三个判据均经失效/平庸通道通过——见下），seed 1/2 走 REJECT 规则（Q1 无平台）；**D3 REJECT（1/3 pass，2/3 reject-rule）**。

### 观察与解释

- **观察 1（仪器失效）**：$\tau_{rev}$ 与 drop 两个通道不含任何随 $\beta\rho$ 变化的信号（常数 1 与 0），去递归对照与完整网完全相同。leak=1.0 + 8 次松弛使每时间步完全重收敛，跨时间步回响在该架构中不存在——这本身是关于架构的实证发现。
- **观察 2（Q2 的 Spearman=+1.000 是伪影）**：继承自 `deep_critical.py` 的秩相关函数对常数序列按稳定排序赋秩 1..n，虚假输出 +1.000。三个 seed 的 Q2=True 均由此产生，不反映任何单调性。判官总结论（2/3 走 reject 规则）不经 Q2，不受影响。
- **观察 3（seed 0 的 pass 是空洞的）**：其 Q1"平台"是噪声底（acc≤0.287 全部点在容差内、MI≈0），Q2 为伪影，Q3 因无信号平庸成立。
- **观察 4（有效信号）**：$\chi$ 在 $\beta\rho\in[0.95, 1.00]$ 达峰后超临界回落（3/3 seed 峰位在 0.95–1.00）；seed 2 的 acc/MI 峰在 0.95（次临界），与 deep_critical 的 $\beta^*=0.80<\beta_c=0.81$ 同向。
- **解释**：精确临界的"功能占优"未获支持（无任何 seed 严格占优），但"次临界平台"也只在一个有信号的 seed 上成立——证据不足以下结论，瓶颈在回响仪器与训练网任务信号强度，不在判据。

### 结论 / 决定

- **按预注册判据：D3 REJECT**。但本条明确标注：该 REJECT 的 2/3 来自"无平台"（seed 1 信号弱、seed 2 平台仅 1 点），而非"精确临界占优"（0/3 seed 严格占优）——证据强度弱于一个干净的证伪。
- **启动 D3 方向约定的一轮迭代（D3′）**：修复回响仪器——改测**松弛迭代尺度**的扰动收缩（定点映射的临界减速，跨时间步回响在此架构不存在的实证结论之上，这是唯一可测的"回响"定义），其余判据与通道不变。预注册见紧邻上条；该轮结束后 D3 关闭。
- **保留**：leak=1.0 架构跨时间步无回响（$\tau\equiv 1$、对照相同）作为严格的架构边界记录。

### 可复现信息

- 命令：`python3 lab/criticality/d3_reverberation.py`。
- 原型：`lab/criticality/d3_reverberation.py`；产物：`results/d3_reverberation.{json,png}`。
- 运行：CPU，PyTorch，seed 0/1/2；复用 `experiments/deep_critical.py` 与 `vpsc/recurrent.py`（未修改）。

---

## 2026-07-17：D3 预注册 — 略次临界回响态 vs 精确临界（预注册）

### 背景 / 动机

调研条目方向 D3（认识论标签：Cross-domain analogy）。Wilting–Priesemann 2018 针对强 subsampling 的稳健估计表明，鼠、猫、猴记录更符合"略次临界但可回响"的中间/reverberating 状态，而非精确临界。本项目已有同方向暗示：`deep_critical.py` 的任务精度峰 $\beta^*=0.80$ 略低于网络 $\beta_c=0.81$。本条把"$\beta\rho=1$ 精确临界最优"与"$\beta\rho<1$ 回响平台最优"作为**竞争假设**直接比较，不预设临界点必然最优。

### 假设与成功标准

- **D3（次临界平台假说）**：VPSC 深递归网的最优工作区是 $\beta\rho<1$ 的连续平台——任务与信息指标在近最优水平连续延伸，回响时间随 $\beta\rho$ 增长——而非 $\beta\rho=1$ 单点。
- **系统与数据**：`RecurrentVPSCNet`（`sizes=[12,40,24]`，4 类时序任务），与 `experiments/deep_critical.py` 完全相同的 `make_data`、80/20 划分与超参；CE 训练（$\beta_{train}=0.8$，100 epoch，`lr=3e-3`）后冻结权重；seed $\{0,1,2\}$ 各训一网。推理期扫描 $\beta\rho=\beta/\beta_c\in\{0.80, 0.85, 0.90, 0.95, 1.00, 1.05, 1.10\}$，$\beta_c$ = 各训练网实测 `critical_beta()`。
- **指标**（每 (seed, $\beta\rho$)，测试集 300 样本）：
  1. **任务指标**：test accuracy。
  2. **$\mathrm{MI}_{corr}$**：类标签（4 类）与顶层状态 `x_top` 的置换校正 MI——24 维按符号 2 值量化，16 次 label-shuffle 零分布（沿用 D2 修正案 2 的估计器）。
  3. **susceptibility $\chi$**：顶层驱动加 $h=0.05$（固定随机单位向量方向，逐网固定）的有限差分响应 $\chi=\mathrm{mean}\|\Delta x_{top}\|_2/h$。
  4. **回响时间 $\tau_{rev}$**：探测输入为测试集前 8 个样本；在 $t_p=8$ 对顶层**持续状态**（warm-start 缓冲区）加固定扰动 $\delta$（范数 0.1，方向逐网固定随机）；$d(k)=\|m_{pert}(t_p+k)-m_{base}(t_p+k)\|_2$（$d(0)=\|\delta\|$）；$\tau_{rev}$ = $d(k)$ 首次 $<d(0)/e$ 的 $k$（未达则取最大 $k$），8 样本平均。
  5. **扰动恢复**：全部测试样本在 $t_p=8$ 受同样扰动后的准确率下降 $\mathrm{drop}=\mathrm{acc}_{base}-\mathrm{acc}_{pert}$。
- **通过标准**（D3 采用需 $\geq 2/3$ seeds 满足以下全部）：
  - **Q1 平台**：$\geq 3$ 个连续次临界 $\beta\rho$ 点满足 $\mathrm{acc}\geq\max_{all}\mathrm{acc}-0.03$ 且 $\mathrm{MI}_{corr}\geq\max_{all}\mathrm{MI}_{corr}-0.25$ bit（max 取全部 7 个扫描点）。
  - **Q2 回响**：Spearman$(\beta\rho\in\{0.80,\dots,0.95\},\ \tau_{rev})\geq 0.8$。
  - **Q3 精确临界不严格占优**：$\mathrm{acc}(1.00)\leq\max_{sub}\mathrm{acc}+0.01$ 且 $\mathrm{MI}_{corr}(1.00)\leq\max_{sub}\mathrm{MI}_{corr}+0.1$。
- **REJECT**：$\geq 2/3$ seeds 满足「$\beta\rho=1$ 严格占优」（$\mathrm{acc}(1.00)>\max_{sub}\mathrm{acc}+0.01$ 且 $\mathrm{MI}_{corr}(1.00)>\max_{sub}\mathrm{MI}_{corr}+0.1$）或 Q1 失败（无平台）。其余情形 MIXED。
- **负面对照与仪器检验**（预注册，非判据，用于解释与仪器验证）：
  - MI 的 label-shuffle 零分布（已含在 $\mathrm{MI}_{corr}$ 内）；
  - **去递归对照**（$\mathrm{W_{rec}}=0$）在 $\beta\rho=1.00$ 的 $\tau_{rev}$ 与 acc：$\tau_{rev}$ 应显著小于完整网（验证 $\tau_{rev}$ 确由递归产生），acc 应下降；
  - susceptibility 应随 $\beta\rho$ 非减（Theorem 3 方向），报告 Spearman 作为仪器 sanity。

### 边界

- D3 比较的是**工作区形状**（单点 vs 平台），与涌现/功能声称无关；任务指标复用既有 4 类时序任务（deep_critical 同源），不构成 C3（双环注意力的任务实验，仍冻结）。
- $\beta\rho=1$ 与次临界平台是竞争假设；任一结果均为严格验证的成果。
- 若 $\chi$ 或 $\tau_{rev}$ 不随 $\beta\rho$ 呈任何系统趋势（仪器失效），在结果条目中如实标注并相应降级结论强度。

### 可复现信息

- 预定原型：`lab/criticality/d3_reverberation.py`（复用 `experiments/deep_critical.py` 的 `make_data`/`train_CE` 与 `vpsc/recurrent.py`，**不修改二者**）。
- 预定产物：`results/d3_reverberation.{json,png}`。
- 运行：CPU，PyTorch（与 deep_critical 相同环境），seed 0/1/2。

---

## 2026-07-17：D2′ 结果 — 载体全称命题否决（N1 18/30）；正反馈必要、负反馈角色分区、拓扑无关（混合结果）

### 背景 / 动机

执行紧邻下条的 D2′ 预注册实验（D2 方向约定的一轮迭代）。本条只报告按预注册判据得到的结果；N1/N2 判据未按结果修改。本轮结束后 D2 方向关闭。

### 结果（原始运行）

案例集：30 个（格， seed）对（full $\mathrm{MI}_{corr}\geq 0.5$ bit，取自 D2 产物）。

- **N1（增益结构必要）：18/30 → FAIL**（预注册要求 100%）。
- **N2（环拓扑非必要）：29/30 = 96.7% → PASS**（预注册要求 ≥80%，唯一失败案例为 (1.0,0.6) seed=1，shuffle=0.097）。
- **总判定：D2′ REJECT。**

逐条件分解（30 案例）：

| 条件 | 结果 | 解读 |
|---|---|---|
| no_positive（去正反馈） | 30/30 → MI=0.000 | 正反馈传播**无条件必要** |
| no_negative（去负反馈） | 18/30 → 0.000；12/30 → 1.726–1.746 | 必要性**随增益区反转**（见下） |
| single_ring | 30/30 → 0.197–0.317 | 单环不足够 |
| 连接 shuffle | 29/30 ≥ 0.5×full（均值 ≈ full） | 环拓扑**非必要** |

N1 的 12 个失败案例**恰好是 $g_{ee}\times 0.8$ 整列**：这些格子去掉负反馈后 MI 升至 1.726–1.746，**高于** full 的 1.36–1.59——负反馈在该增益区主动压制刺激信息。在 $g_{ee}\geq 1.0$ 区，去负反馈则归零（负反馈必要）。

### 观察与解释

- **观察 1**：正反馈（E 环 + 局部传播）在全部信息格必要；单环（无 E/I 分工、无振荡）在任何格都不足够。
- **观察 2**：负反馈的必要性依赖增益区——$g_{ee}\times 0.8$ 列去之 MI 反升，$g_{ee}\geq 1.0$ 区去之归零。负反馈是**增益调制器**，不是信息的构成部件。
- **观察 3**：任意置换的邻居结构在 96.7% 案例中同样承载信息——相干行波/环几何不是信息载体。
- **解释**：刺激信息的载体是"正反馈 + 局部传播"的增益结构，与振荡、行波相干性、环拓扑均无关。这解释了 D2 的 P2/P3 为何在峰格 (1.0,1.25) 同时成立——峰格恰好落在负反馈必要区，单格证据无法区分"E/I 整体必要"与"仅正反馈必要"。
- **解释（连接 C2″）**：C2″ 显示"任何动力学都保留痕迹"（被动残余），D2′ 显示"主动承载刺激信息只需正反馈传播"——两者共同否定了"E/I 耦合整体是信息载体"的陈述；信息载体的答案比两条预注册假设都更简单、更依赖参数区。

### 结论 / 决定

- **不采用** D2′（全称命题"E/I 增益结构必要"被 $g_{ee}\times 0.8$ 列否决）。
- **D2 方向关闭**：两轮预注册实验完成（D2 REJECT、D2′ REJECT），迭代轮已用完。
- D2 线净产出（均为严格验证的边界）：(i) Shew 式中间信息峰在本系统不存在（峰触界、动态范围峰值在低增益非振荡区）；(ii) 信息不依赖环拓扑（96.7%）；(iii) 正反馈传播在全部信息格必要，负反馈必要性随增益区反转；(iv) 信息"是否存在"高度依赖测量协议（与 C2″ 对照）——任何后续信息声称必须连同协议一起陈述。

### 可复现信息

- 命令：`python3 lab/ring_feedback/d2_prime_carrier.py`。
- 原型：`lab/ring_feedback/d2_prime_carrier.py`；产物：`lab/ring_feedback/results/d2_prime_carrier.{json,png}`；源产物：`lab/ring_feedback/results/d2_information_peak.json`（案例集来源）。
- 运行：CPU，NumPy float64，RK4 `dt=0.04`，seed 0/1/2，全程无警告。

---

## 2026-07-17：D2′ 预注册 — E/I 增益结构是刺激信息的载体（非振荡、非环拓扑）（预注册）

### 背景 / 动机

紧邻下条 D2 结果的分解：P2（峰格去 E/去 I/单环 → MI=0）与 P3（峰格连接 shuffle → MI 不降）联合表明，**在峰格**，刺激信息的必要载体是 E/I 增益结构而非环拓扑。但该证据只覆盖一个格点。D2′ 把这一陈述推广为可在全部信息格上判官的命题。这是 D2 方向按约定仅有的一轮迭代；本轮结束后 D2 关闭。

### 假设与成功标准

- **D2′**：在全部信息格上，刺激信息（$\mathrm{MI}_{corr}$）的必要载体是 E/I 增益结构——去正反馈、去负反馈或 A3 单环则信息崩溃；而环拓扑的相干传播不必要——连接 shuffle 后信息保留。
- **案例集**：D2 网格中全部满足 full $\mathrm{MI}_{corr}\geq 0.5$ bit 的（格， seed）对（从 D2 产物 JSON 的 per-seed 值直接选取，不重跑 full）。
- 通过标准（两条全满足 → 采用）：
  - **N1 增益结构必要**：100% 案例中 $\max(\mathrm{MI}_{noE}, \mathrm{MI}_{noI}, \mathrm{MI}_{single}) \leq 0.5\times \mathrm{MI}_{full}$（同格同 seed，各自置换校正）。
  - **N2 环拓扑非必要**：$\geq 80\%$ 案例中 $\mathbb{E}[\mathrm{MI}_{shuffle}]$（5 次重复）$\geq 0.5\times \mathrm{MI}_{full}$。
- 失败条件：任一信息格中消融系统保留 $>0.5\times$ MI（C2″ 提示这可能发生在非振荡格——若然，N1 被否决），或 $\geq 20\%$ 信息格 shuffle 后 MI 跌破一半（环拓扑实则必要，N2 被否决）。

### 实验设计（预注册，执行不得修改）

- **完全复用 D2 冻结管线**：同一网格、刺激集合、响应窗、量化与置换校正 MI 估计器、同一批实例种子。只对案例集中的（格， seed）补测 no_positive、no_negative、single_ring、连接 shuffle（5 次重复，置换种子与 D2 相同规则：seed×100+rep）。
- 判据 N1/N2 的 MI 均为同管线置换校正 MI（16 次 label-shuffle 零分布）。
- 不再引入新指标、新刺激或新阈值以外的任何自由参数。

### 边界

- D2′ 只回答"信息的必要载体是什么"；不恢复 D2 已被否决的"中间峰"声称，不支持任何功能/注意力声称。
- 若 N1 通过而 N2 失败，结论为"信息需要 E/I 增益结构与环拓扑"；反之亦然；两者均为严格验证的结果。

### 可复现信息

- 预定原型：`lab/ring_feedback/d2_prime_carrier.py`。
- 预定产物：`lab/ring_feedback/results/d2_prime_carrier.{json,png}`。
- 运行：CPU，NumPy float64，RK4 `dt=0.04`，seed 0/1/2。

---

## 2026-07-17：D2 结果 — E/I 信息峰不成立（REJECT）；峰触界、动态范围在低端、拓扑无关（混合结果）

### 背景 / 动机

执行紧邻下条的 D2 预注册实验（含运行前修正案 1、2）。本条只报告按修正后预注册判据得到的结果；判据 P1–P4 未按结果修改。

### 结果（原始运行）

36 格 × 3 seed 全网格（括号内为该格 seed 平均 $\mathrm{MI}_{corr}$，bit）：

- 振荡格 7 个（≥2/3 seed 过 C1）：(0.8,0.4)、(0.8,0.6)、(0.8,0.8)、(1.0,1.0)、(1.0,1.25)、(1.0,1.5)、(1.25,1.5)。
- 信息格（$\mathrm{MI}_{corr}\geq 0.5$）10 个：上述振荡格中除 (1.25,1.5)=1.409 外全部，外加非振荡格 (0.8,1.0)=1.474、(1.0,0.6)=0.717、(1.0,0.8)=1.644；峰值 1.736 在 (1.0,1.25) 与 (1.0,1.5)。其余格子 $\mathrm{MI}_{corr}=0$。
- 非振荡格 $\mathrm{MI}_{corr}$ 均值 0.132。

P1 中间峰（三指标）：

| 指标 | 峰值 | 峰值格 | 含内部振荡格 | 触边界 | 边际 | 判定 |
|---|---:|---|---|---|---|---|
| MI | 1.736 | (1.0,1.25), (1.0,1.5) | 是 | **是** | 满足（≥0.132+0.5） | FAIL |
| pattern entropy | 3.000 | (0.8,0.4), (0.8,0.8), (1.0,1.0), (1.0,1.25), (1.0,1.5) | 是 | **是** | 满足（≥0.220+0.5） | FAIL |
| 动态范围 | 0.565 | (0.6,0.4) | **否** | **是** | — | FAIL |

对照（MI 峰值格 (1.0,1.25)，3 seed）：full $\mathrm{MI}_{corr}$ = 1.726–1.746；no_positive=0.000、no_negative=0.000、single_ring=0.197–0.317 → **P2 PASS（3/3）**；连接 shuffle=1.591–1.651（$>0.5\times$ full=0.87）→ **P3 FAIL（0/3）**；$\mathrm{MI}_{raw}$=3.000 $>$ label-shuffle max 1.375–1.562 → **P4 PASS（3/3）**。

**总判定：D2 REJECT**（P1 三指标全失败 + P3 失败）。

### 观察与解释

- **观察 1**：信息区是一条以 $g_{ee}\in\{0.8,1.0\}$ 为主的带，同时覆盖振荡格与非振荡格（如 (0.8,1.0)、(1.0,0.8) 非振荡但 $\mathrm{MI}_{corr}\approx 1.5$）；信息并非振荡区特产。
- **观察 2**：MI 与熵的峰值平台延伸到扫描边界（$g_{ei}=1.5$），在扫描范围内不衰减——"峰在中间"不成立（严格地说：峰是否中间超出本网格可判范围，预注册规则计 FAIL）。
- **观察 3**：动态范围与增益反相关：最大值在低增益角 (0.6,0.4)=0.565，振荡格 ≈0.000——自发振荡淹没诱发响应（正是修正案 1 修复的基线问题的另一面）。Shew 式"中间 E/I 动态范围峰"在本系统不成立。
- **观察 4**：峰格上去 E、去 I、单环的 MI→0（P2），但连接 shuffle 几乎不降 MI（P3）：**信息需要 E/I 增益结构，但不需要环拓扑的相干传播**。
- **解释（与 C2″ 的表面对立）**：C2″ 在 (1.0,1.0) 用单节点 cue + 长窗 $[12,42)$ + 40 维特征，消融后仍可解码；D2 在 (1.0,1.25) 用 3 节点随机刺激 + 短窗 $[4,16)$ + 量化均值 + 置换校正 MI，消融后 MI=0。协议不同、结论不同——信息"是否存在"高度依赖测量协议，这本身是本轮最重要的方法学观察。
- **解释**：D2 把 Shew 命题分解为三句可分别检验的话：信息在中间振荡区达峰（**否**）、信息需要完整 E/I（在峰格：**是**）、信息需要环拓扑（**否**）。

### 结论 / 决定

- **不采用** D2（Shew 式中间信息峰）：P1 三指标全失败（MI/熵峰触界、动态范围峰值在低增益非振荡区），P3 失败（信息不依赖环交互结构）。
- **启动 D2 方向约定的一轮迭代**：P2+P3 的分解直接给出一个新的可证伪命题——D2′"E/I 增益结构（非振荡、非环拓扑）是刺激信息的载体"，把峰格的 P2/P3 判据推广到全部信息格。预注册见紧邻上条；该轮结束后 D2 关闭，无论结果。
- **保持边界**：D2′ 只回答"信息的必要载体"，不恢复"中间峰"声称，不支持任何功能/注意力声称。

### 可复现信息

- 命令：`python3 lab/ring_feedback/d2_information_peak.py`。
- 原型：`lab/ring_feedback/d2_information_peak.py`；产物：`lab/ring_feedback/results/d2_information_peak.{json,png}`。
- 运行：CPU，NumPy float64，RK4 `dt=0.04`，seed 0/1/2，全程无警告。
- 预注册与修正案 1（量化/动态范围基线）、修正案 2（置换校正 MI + 置换检验 P4）：见紧邻下条，均在首次全网格运行之前写入。

---

## 2026-07-17：D2 预注册 — E/I 信息峰（预注册）

### 背景 / 动机

执行调研条目的方向 D2（认识论标签：Established direction，VPSC 迁移未证）。Shew et al. 2009/2011 的药理干预证据表明：动态范围、Shannon entropy、stimulus-response MI 在中间 E/I / avalanche 区达峰。本条目检验该"中间信息峰"是否存在于本项目 C 线的 E/I 双环。D1 的两条边界（振荡 ≠ 相位编码；可解码 ≠ 振荡维持）直接塑造了本设计：刺激集合必须打破旋转对称，指标必须是信息量的而非可解码性的。

### 假设与成功标准

- **D2**：在固定刺激集合下，pattern entropy、stimulus-response MI、动态范围三个连续信息指标在 E/I 增益网格的中间（振荡）区达峰，且峰依赖完整 E/I 与环拓扑交互。
- 通过标准（P1–P4 全部满足 → 采用）：
  - **P1 中间峰**：三个指标各自满足——(a) seed 平均指标的峰值格集合（指标 $\geq$ 网格最大值 $-10^{-9}$ 的全部格子）含至少一个**内部振荡格**，且不含任何边界格（内部 = 两因子均不在各自扫描端点）；(b) 峰值格指标 $\geq$ 全部非振荡格的 seed 平均值 + 边际（MI 0.5 bit，entropy 0.5 bit，动态范围 0.2）。峰值格触及边界即视为峰不在中间区（对应该指标假设为假）。
  - **P2 消融降指**：在 MI 峰值格，no_positive、no_negative、single_ring 的 MI 均 $\leq 0.5\times$ full（3 个 seed 中 $\geq 2$ 满足）。
  - **P3 交互打乱**：在 MI 峰值格，连接 shuffle（两环核施加同一随机节点置换，5 次重复）平均 MI $\leq 0.5\times$ full（3 个 seed 中 $\geq 2$ 满足）。
  - **P4 无估计偏差**：label-shuffle MI $\leq 0.15$ bit（3 个 seed 中 $\geq 2$ 满足）。
- P1 失败 → 不采用（无中间信息峰）；P2/P3 失败 → 不采用（峰不依赖完整 E/I 或环交互）；P4 失败 → 实验无效（估计器偏差）。

### 实验设计（预注册，执行不得修改）

- **系统**：完整 E/I 双环，`Params()` 基准。网格：`g_ee` factor × `g_ei` factor $\in\{0.4, 0.6, 0.8, 1.0, 1.25, 1.5\}^2$（36 格）× seed $\{0,1,2\}$（108 点）。每格用 C1 判据（autonomous 协议）标记振荡/定点。
- **刺激集合**（固定，rng seed `20260717` 一次生成并连同网格坐标存入 JSON）：8 个随机模式，每个激活 8 节点中的 3 个（每激活节点幅度 5.0，$t\in[0,2)$）。**不用单节点刺激**：D1 证明旋转对称环上单节点刺激互为循环移位，任何动力学下都可被平庸解码，无法暴露饱和区刺激特异性的崩溃。每模式 4 个抖动实例（仅初始状态抖动不同）。
- **响应**：响应窗 $t\in[4,16)$（`total_time=16`）；特征 = 窗内每节点 mean($e$)（8 维）。
- **指标定义**（量化与动态范围基线见下方修正案）：
  - pattern entropy：8 个刺激各取实例平均模式，逐维 2 值量化（固定边界 0.30），对 8 个量化模式（等权）求 Shannon 熵（bit）。
  - stimulus-response MI：$I(K;R)$，$R$ = 单实例 8 维 2 值量化模式（同一边界），由 32 个（刺激 × 实例）对的列联表估计；label-shuffle（4 次重复）为偏差对照。
  - 动态范围：固定刺激模式 0，幅度因子 $\in\{0.1, 0.2, 0.4, 0.6, 0.8, 1.0, 1.3, 1.6\}$；响应幅度 = 各节点窗内均值之和 $-$ 同窗无刺激基线。$\Delta=\log_{10}(I_{90}/I_{10})$，$I_x$ 为响应首次达到最大响应 $x\%$ 的幅度因子（线性插值；90\% 未达到则 $I_{90}=1.6$ 封顶；10\% 未达到则 $\Delta=0$）。

### 修正案（2026-07-17，首次全网格运行之前）

冒烟检查（仅 $(1.0,1.0)$ 与 $(0.4,0.4)$ 两格的机械验证，未运行任何判据相关网格，未产生任何判据数据）发现原定义两处**结构性退化**，与假设真假无关地使实验无效：

1. **2 值量化（边界 0.30）在振荡点把全部刺激模式压成同一比特串**——行波扫过所有节点，窗内每节点均值对刺激几乎不变，实测 MI=0.000。改为 **4 值固定量化，边界 $\{0.2, 0.4, 0.6, 0.8\}$**（绝对边界，跨格可比，无数据依赖）。
2. **动态范围基线（无刺激运行）在自发振荡点本身高度活跃**，evoked − baseline ≤ 0，实测 Δ=0.000。改为：基线 = 同一幅度序列的 **0 因子成员**（幅度因子集合加入 0.0），超额响应 $=\max(0,\,r(a)-r(0))$，在超额曲线上求 $I_{10}/I_{90}$（插值、封顶规则不变）；最大超额 $\leq 10^{-9}$ 时 $\Delta=0$。

其余定义（网格、窗口、刺激集合、对照、P1–P4 判据与边际）**不变**。本修正案仅修复测量工具，判定规则未动。

### 修正案 2（2026-07-17，首次全网格运行之前）

修正案 1 后的冒烟检查（5 个代表性格点的机械验证，仍未运行判据网格，未产生判据数据）发现 **MI 估计的有限样本正偏差不可忽略**：4 值量化下，32 样本的 label-shuffle MI 在信息丰富格点达 0.86–1.31 bit，原始 MI 不可跨格直接比较，预注册 P4（label-shuffle $\leq 0.15$ bit）在该量化精度下必然失败。修正：

1. 所有 MI 量一律改为**偏差校正 MI**：$\mathrm{MI}_{corr}=\mathrm{MI}_{raw}-\mathbb{E}[\text{label-shuffle MI}]$；label-shuffle 重复由 4 次增至 **16 次**以稳定偏差估计（仅列联表重算，不增加模拟量）。
2. 凡引用 MI 的判据（P1 峰值格认定与 0.5 bit 边际、P2/P3 的 $0.5\times$ 比值）均作用于 $\mathrm{MI}_{corr}$；$\mathrm{MI}_{corr}<0$ 截断为 0 仅用于展示，判据用未截断值。
3. **P4 重述为置换检验**：在 MI 峰值格，每个 seed 的 $\mathrm{MI}_{raw}$ 须大于该 seed 16 次 label-shuffle MI 的最大值（$p<1/17$），3 个 seed 中 $\geq 2$ 满足。校正后的偏差本身不再设绝对上限——置换检验直接检验显著性，不因量化精度惩罚估计器。
4. pattern entropy（由实例平均模式计算，无此偏差）与动态范围定义不变。

本修正案仍只触及估计器的偏差处理与显著性检验，P1–P3 的判定结构、边际与全部实验参数不变。
- **对照**（在 MI 峰值格测量）：no_positive（`g_ee=0`）、no_negative（`g_ei=0`）、single_ring（默认参数 A3 单环）、连接 shuffle（对两个环核的邻居映射施加同一随机节点置换，保留每节点出度为 1 但破坏环几何，5 次重复）。
- **失败条件**（与调研条目一致）：振荡区无中间信息峰（P1），或交互打乱后指标不降（P3），或峰不需要完整 E/I（P2）。

### 边界

- D2 通过仅说明"中间 E/I 振荡区携带更丰富的刺激信息"，不声称注意力/门控功能；D1 已表明可解码性本身不构成功能证据。
- 三个指标与全部对照以本条预注册定义为准，不事后更换估计器或边界。

### 可复现信息

- 预定原型：`lab/ring_feedback/d2_information_peak.py`。
- 预定产物：`lab/ring_feedback/results/d2_information_peak.{json,png}`。
- 运行：CPU，NumPy float64，RK4 `dt=0.04`，seed 0/1/2。

---

## 2026-07-17：C2″ 结果 — 痕迹维持不依赖完整 E/I（0/27 REJECT，D1 方向关闭）（负面结果）

### 背景 / 动机

执行紧邻下条的 C2″ 预注册实验（D1 方向约定的一轮迭代）。本条只报告按预注册判据得到的结果；判据 S1–S3 与网格规则未按结果修改。本轮结束后 D1 方向关闭。

### 结果（原始运行）

基准点（seed=0）：full acc=1.0000，no_positive=1.0000，no_negative=1.0000，single_ring=0.3438，label-shuffle=0.2109；S1 PASS、S2 FAIL、S3 PASS。

27 点网格（21 点振荡，与 C1/C2′ 完全同一批点；6 点 `no_oscillation` 直接 FAIL）：

| 条件 | acc 范围 | 均值 | S2 子项（≤0.325） |
|---|---:|---:|---:|
| full | [0.969, 1.000] | 0.999 | —（S1：21/21 通过） |
| no_positive | [1.000, 1.000] | 1.000 | 0/21 |
| no_negative | [1.000, 1.000] | 1.000 | 0/21 |
| single_ring | [0.313, 0.469] | 0.403 | 5/21 |

label-shuffle ∈ [0.078, 0.273]，S3 通过 17/21。网格通过率 **0/27** → 按预注册规则（≤1/3）：**C2″ REJECT**。决定性失败是 S2（消融即毁）：0/21。

### 观察与解释

- **观察 1**：完整 E/I 的延迟深部痕迹总是可解码（S1 21/21，acc≥0.969）。
- **观察 2**：去正反馈、去负反馈后痕迹在全部 21 点仍 **100%** 可解码。消融系统的延迟状态不是 cue 无关定点，而是 cue 依赖的（多稳态/慢残余）。
- **观察 3**：连 A3 单环的饱和定点都保留约 0.40 的部分痕迹（16/21 点高于 0.325 门槛）。
- **解释**：延迟窗内的 cue 痕迹是**被动残余**——任何不从外部擦除状态的动力学都保留它，与振荡无关。C2″ 的因果预测（完整 E/I 是痕迹维持的必要条件）被否决。
- **解释（S3 的 4 个失败点）**：label-shuffle 0.23–0.27 属 $n_{\text{test}}=32$ 的有限样本波动（chance 0.125），非系统性泄漏——若解码器泄漏，全部点应一致偏高而非零星越界。
- **解释（方法学，连接 C2′）**：在这个近确定性小系统里，"延迟期可解码"几乎平庸成立。可解码性既不能证明相位编码（C2′），也不能证明振荡维持记忆（C2″）；有意义的判官必须测干预效应（C2 式）或载体特异性（C2′ 式 shuffle）。

### 结论 / 决定

- **不采用** C2″：痕迹维持不依赖完整 E/I（0/27 REJECT）。
- **D1 方向关闭**：C2′（相位载体，0/27）与 C2″（E/I 因果维持，0/27）均被否决；迭代轮已用完，不再提出第三轮。
- D1 的三条净产出（均为严格验证的边界）：(i) 振荡 ≠ 相位编码；(ii) 可解码 ≠ 振荡维持；(iii) shuffle 与消融对照是判别信息载体与因果维持的最低门槛。
- 保留的基线观察（"延迟期痕迹普遍可解码"）不构成任何功能声称，不进入 C3。

### 可复现信息

- 命令：`python3 lab/ring_feedback/d1_trace_ablation.py`。
- 原型：`lab/ring_feedback/d1_trace_ablation.py`；产物：`lab/ring_feedback/results/d1_trace_ablation.{json,png}`。
- 运行：CPU，NumPy float64，RK4 `dt=0.04`，seed 0/1/2。
- 数值卫生：标准化下限 1e-6（特征 O(1)，更低方差视为数值噪声）+ `errstate` 局部屏蔽 umath_linalg 良性警告；重跑全程无警告，产物 JSON 无 NaN（已逐点核验）。

---

## 2026-07-17：C2″ 预注册 — 痕迹维持的 E/I 因果依赖（预注册）

### 背景 / 动机

紧邻下条 D1 结果：延迟期 cue 痕迹高度可解码（acc≈1.0），但载体是洗牌不变的空间/边际模式而非相位（C2′ 0/27 REJECT）。可解码性是观察而非因果——痕迹可能由完整 E/I 的振荡动力学维持，也可能任何动力学（包括收敛到 cue 无关定点的消融系统）都同样保留 cue 依赖。C2″ 用预注册消融做因果判别。这是 D1 方向按约定仅有的一轮迭代；本轮结束后 D1 关闭。

### 假设与成功标准

- **C2″**：完整 E/I 环把 cue 痕迹维持到延迟深部（$t\geq 12$），且该维持依赖完整 E/I 耦合——去正反馈、去负反馈与 A3 单环的延迟深部状态收敛到 cue 无关定点，解码掉到 chance 附近。
- 单点通过标准（以下全部满足）：
  - **S1 痕迹可解码**：完整 E/I 解码 acc $\geq 0.60$（与 C2′ R1 同一门槛）。
  - **S2 消融即毁**：no_positive、no_negative、single_ring 三个条件 acc 均 $\leq 0.325$（= chance $0.125+0.20$）。
  - **S3 无泄漏**：完整 E/I 的 label-shuffle acc $\in[0.025,\,0.225]$。
- 网格判定：与 C2′ 同一 27 点网格（`g_ee`、`g_ei` $\times 0.9/1.0/1.1$ × seed 0/1/2）；通过率 $\geq 2/3$ → 采用；$\leq 1/3$ → 不采用；介于之间 → 混合。完整 E/I 不满足 C1 振荡判据的点自动 FAIL（无痕迹载体，与 C2′ 同一处理）。

### 实验设计（预注册，执行不得修改）

- **完全复用 C2′ 冻结管线**：同一积分器与参数、同一 64 条轨迹/点（8 cue 节点 × 8 实例，实例仅初始状态抖动不同）、同一读出窗 $t\in[12,42)$、同一 40 维特征（每节点 [mean, std, 主频振幅, $\sin\varphi$, $\cos\varphi$]）、同一闭式 ridge 解码器（$\lambda=1.0$）与同一分层划分（实例 0–3 训练 / 4–7 测试）。
- **条件**：full、no_positive（`g_ee=0`）、no_negative（`g_ei=0`）、single_ring（A3 tanh 单环，定义同 `c2_verify.py`）。消融条件用与 full 相同的 cue 节点、实例抖动与读出窗。
- **主频**：全部四个条件统一使用该点**完整 E/I** 自主运行的 dominant frequency（消融条件无自有振荡，特征仍按同一频率投影计算；其预期结果是延迟深部状态近常数、特征近 cue 无关、解码 ≈ chance——这正是被检验的预测，不需特殊处理）。
- **对照**：label-shuffle（4 次重复取均值，仅完整 E/I）。
- **失败条件**：完整 E/I 不可解码（S1 失败），或任一消融条件的延迟深部状态仍携带可解码 cue 信息（S2 失败，例如去负反馈若进入 cue 依赖的多稳态饱和，则 C2″ 被否决）。

### 边界

- C2″ 只检验"痕迹维持是否依赖完整 E/I"，不检验功能可分离性（C2 已 0/27 否决，不回写），不支持任何"双环注意力"声称。
- 若 C2″ 通过，结论是"振荡 E/I 动力学维持刺激痕迹、定点动力学不能"——一个关于**动力学维持信息**的因果陈述，仍非任务功能证据。

### 可复现信息

- 预定原型：`lab/ring_feedback/d1_trace_ablation.py`。
- 预定产物：`lab/ring_feedback/results/d1_trace_ablation.{json,png}`。
- 运行：CPU，NumPy float64，RK4 `dt=0.04`，seed 0/1/2。

---

## 2026-07-17：D1 结果 — C2′ 相位特异读出 0/27 否决；痕迹高度可解码但载体非相位（混合结果）

### 背景 / 动机

执行紧邻下条的 C2′ 预注册实验。本条只报告按预注册判据得到的结果，预注册文本（判据 R1–R4、网格规则）未按结果修改。

### 结果（原始运行）

基准点（seed=0）：phase acc=1.0000，fixed acc=1.0000，time-shuffle=1.0000，delay-shuffle=1.0000，label-shuffle=0.1094（chance=0.125）；R1–R3 FAIL、R4 PASS。

27 点网格（与 C2 同一局部网格 × seed 0/1/2）：

- 21 点通过 C1 振荡判据、6 点 `no_oscillation` 直接 FAIL（(g_ee×1.1, g_ei×0.9) 与 (g_ee×1.1, g_ei×1.0) × 3 seed）。经逐点比对，这 21 点与 C2 条目 C1 通过的 **21/27 完全同一批点**。
- 21 个振荡点上：phase acc ∈ [0.969, 1.000]；fixed acc 18/21 点为 1.000、最低 0.781；time-shuffle ∈ [0.977, 1.000]；delay-shuffle ∈ [0.977, 1.000]；label-shuffle ∈ [0.031, 0.203]。
- R1（phase ≥ 0.60 且 ≥ fixed + 0.30）：**0/21**（phase 与 fixed 几乎打平）。
- R2/R3（shuffle 后优势至少减半）：**0/21**（shuffle 后解码仍 ≈1.0）。
- R4（label-shuffle ∈ [0.025, 0.225]）：21/21 PASS（无泄漏）。
- 网格通过率 **0/27** → 按预注册规则（≤1/3）：**C2′ REJECT**。

### 观察与解释

- **观察 1**：延迟期轨迹确实携带完整 cue 信息——40 维相位特征与 8 维延迟末平均注意力的训练线性探针均以 ≈100% 解码 8 类 cue（chance 12.5%）；8 类在 $(\sin\varphi,\cos\varphi)$ 平面上聚成 8 个分离簇（见产物图右上）。
- **观察 2**：time-shuffle（破坏全部时间结构、保留边际统计）与 delay-shuffle（破坏节点间相对相位）之后解码仍 ≈1.0；label-shuffle 掉到 chance。
- **解释**：cue 信息的载体是**洗牌不变的空间/边际活动模式**（每节点窗内均值/方差构成的旋转模板），不是相位或相对时序。旋转对称环 + 近确定性极限环使 cue $k$ 的轨迹 ≈ cue 0 轨迹的 $k$ 步循环移位，均值模式即充分解码。预注册的三类 shuffle 正是为区分这两种载体而设，判别有效。
- **解释（对 C2 低 $M$）**：C2 的低 $M$ 并非"信息藏在相位里"，而是其读出的两个具体选择所致——固定目标节点标量 $a[\text{target}]$ + 窗口 $[10,12)$；换用延迟末窗 $[40,42)$ 的 8 维模式训练探针即可达 1.0。**此发现不回写 C2**：C2 检验"正/负反馈功能可分离性"，其 0/27 与预注册读出保持不变。
- **推测**：在旋转对称环上，"相位编码"与"空间模式编码"本就不可区分（一类轨迹是另一类的循环移位）；分离两者需打破对称的拓扑。此为可选后续，本条不展开。

### 结论 / 决定

- **不采用** C2′（相位特异读出）：预注册 R1–R3 全失败，0/27 REJECT。
- **保留**负面边界：轨迹可解码 ≠ 相位/时序编码；shuffle 对照是区分载体的必需手段，缺之会把边际模式误判为相位编码。
- **启动 D1 方向允许的一轮迭代**：痕迹可解码性 ≈1.0 是观察而非因果——它可能由完整 E/I 振荡维持，也可能定点动力学同样保留 cue 依赖。新命题 C2″（痕迹维持的 E/I 因果依赖）见紧邻上条预注册；该轮结束后 D1 方向关闭，无论结果。

### 可复现信息

- 命令：`python3 lab/ring_feedback/d1_phase_readout.py`。
- 原型：`lab/ring_feedback/d1_phase_readout.py`；产物：`lab/ring_feedback/results/d1_phase_readout.{json,png}`。
- 运行：CPU，NumPy float64，RK4 `dt=0.04`，seed 0/1/2。
- 实现备注：闭式 ridge（λ=1.0）用 SVD-lstsq 求解；umath_linalg 对近共线特征矩阵发出的良性 divide/overflow 警告已局部 `errstate` 屏蔽（输入有限性已逐点验证，屏蔽前后数值一致）。

---

## 2026-07-17：D1 预注册 — C2′ phase/polychronous 读出（预注册）

### 背景 / 动机

承接 C2 混合结果与上方涌现调研条目：C1 通过（21/27）、C2 失败（0/27）。对 C2 低 $M$ 的唯一保留解释是"cue 可能编码在环上传播轨迹的相位/相对时序中，而非固定 target 节点"。本条目把该解释形式化为新命题 **C2′** 并预注册判官实验（调研条目方向 D1）。C2′ 不回写 C2 结论：无论结果如何，C2 的 0/27 保持不变。

### 假设与成功标准

- **C2′**：完整 E/I 环的延迟期轨迹把 cue 位置编码在节点间相对相位/时序结构中；phase-aware 线性读出能跨初始条件抖动泛化解码 cue 位置，且该优势依赖完整时间结构。
- 单点通过标准（以下全部满足）：
  - **R1 可解码性**：phase decoder 测试准确率 $\geq 0.60$，且 $\geq$ 固定位置读出 $+0.30$（chance $=1/8=0.125$）。
  - **R2 时间结构依赖**：time-shuffle 后 $(\mathrm{acc}-0.125)\leq 0.5\times(\text{phase acc}-0.125)$。
  - **R3 相对相位依赖**：delay-shuffle（逐节点独立循环移位）后判据同 R2。
  - **R4 无泄漏**：label-shuffle 准确率 $\in[0.025,\,0.225]$。
- 网格判定：在 C2 同一局部网格（`g_ee`、`g_ei` $\times 0.9/1.0/1.1$）$\times$ seed 0/1/2 共 27 点上，通过率 $\geq 2/3$ → C2′ 采用；$\leq 1/3$ → 不采用；介于之间 → 混合结果。若某点完整 E/I 不满足 C1 振荡判据（无相位载体），该点直接计 FAIL 并标注 `no_oscillation`。

### 实验设计（预注册，执行不得修改）

- **系统**：完整 E/I 双环，参数与 `c2_verify.py` 的 `Params()` 一致（基准点同 C2 探索性中心），同一 RK4 积分器与 `dt=0.04`。
- **刺激**：cue 节点 $k\in\{0..7\}$，幅度 5.0，$t\in[0,2)$；随后无输入至 `total_time=44`。每节点 8 个实例，实例间仅初始状态抖动不同（rng 由 seed、节点、实例派生）。每参数点 64 条轨迹。
- **读出窗口**：$t\in[12,42)$（撤 cue 后 10 个时间单位起，排除输入残留与初始瞬态；约覆盖 2 个自主周期）。固定位置读出窗口：$t\in[40,42)$（延迟末 2 单位，对应 C2 的延迟末窗）。
- **特征（仅用 $e$ 状态）**：
  - phase-aware（40 维）：每节点 [mean, std, 主频振幅, $\sin\varphi$, $\cos\varphi$]；主频取该点自主运行的 dominant frequency（同一频率用于全部条件与全部对照；$\varphi$ 以 sin/cos 成对给出，消除相位原点任意性）。
  - 固定位置（8 维）：延迟末窗平均注意力 $a(t)$（与 C2 相同的 softmax 读出，$\kappa=6,\lambda=1$）。
- **解码器**：闭式 ridge 回归（$\lambda=1.0$）到 one-hot 标签，argmax；特征用训练集均值/方差标准化；分层划分实例 0–3 训练、4–7 测试（32/32）。所有条件共用同一解码器、划分与标准化。
- **负面对照**（各 4 次重复取均值，复用同一批轨迹只重算特征）：
  - time-shuffle：读出窗内统一随机置换时间轴（保留边际统计，破坏全部时间结构）；
  - delay-shuffle：逐节点独立随机循环移位（保留单节点时间结构，破坏节点间相对相位）；
  - label-shuffle：置换训练标签（泄漏检查）。
- **失败条件**（与调研条目一致）：轨迹不可重复（R1 失败），或 decoder 优势在 shuffle 后仍存在（R2/R3 失败）。

### 与 C2 的关系 / 边界

- C2′ 是独立新命题：检验"cue 信息在哪里"，不检验"正/负反馈功能可分离"；C2 的 0/27 结论不受本条结果影响。
- 若 C2′ 通过，仅说明"相位携带 cue 信息"；仍不能声称"双环注意力"（需功能任务证据，属后续，本实验不涉及）。

### 可复现信息

- 预定原型：`lab/ring_feedback/d1_phase_readout.py`。
- 预定产物：`lab/ring_feedback/results/d1_phase_readout.{json,png}`。
- 运行：CPU，NumPy float64，RK4 `dt=0.04`。

---

## 2026-07-17：神经群体涌现证据调研 — 新方向与反证边界（调研结论）

### 背景 / 动机

在 Reynolds/Boids 的“局部规则经群体交互产生宏观行为”启发下，继续核查神经科学中是否存在相似的实验或理论论证。此时本项目已经有一个重要混合结果：C1 中 E/I 双环在探索性中心附近的局部网格有 **21/27（77.8%）** 产生有界振荡，但 C2 的“正反馈=记忆、负反馈=门控”功能可分离性为 **0/27（0%）**。因此本次调研的目的不是为 C2 寻找事后解释，而是回答两个新问题：

1. 局部递归、E/I、时间延迟和局部可塑性在神经系统中能涌现出哪些**已实测**的群体变量？
2. 哪些证据足以支持新的 VPSC 命题，哪些反证必须保留，防止把振荡、幂律或规模效应误称为功能涌现？

本条是文献证据与方向收敛记录，不是本地实验结果；不同论文的数字来自不同制备、记录尺度与指标，**不可横向当作同一基准比较**。

### 证据分层

| 证据 | 类型 | 局部机制与群体现象 | 能支持什么 | 不能支持什么 |
|---|---|---|---|---|
| [Hopfield 1982](https://papers.baulab.info/papers/also/Hopfield-1982.pdf) | 理论/模拟 | 二值神经元异步阈值更新 + 对称递归连接，整体形成内容寻址吸引子 | 简单、异步单元可通过能量地形实现群体记忆 | 权重由 Hebbian 处方写入，不证明真实脑或自发学习出吸引子 |
| [Song–Miller–Abbott 2000](https://www.seti.net/Neuron%20Lab/NeuronReferences/Competitive%20Hebbian%20learning%20-%20Song%202000%20.pdf) | 基于实验窗的模型 | 单突触 STDP 通过争夺 postsynaptic spike timing 形成竞争，短时相关输入成组增强 | 局部 timing rule 可产生全局突触选择与不规则平衡态 | 稳定依赖 STDP 窗总体略偏 depression；沉默神经元仍需 homeostasis/scaling |
| [Izhikevich 2006](https://www.izhikevich.org/publications/spnet.pdf) | SNN 模拟 | 脉冲神经元 + 轴突延迟 + STDP 自组织出毫秒精度的 polychronous groups | 信息可能编码在相对时序/群组合，而非固定节点或同步率 | 非直接生物验证；群会生长、竞争、衰退，注意力/意识论述是推测 |
| [Beggs–Plenz 2004](https://pubmed.ncbi.nlm.nih.gov/15175392/) | 鼠皮层切片 MEA | 自发 neuronal avalanches 聚成稳定时空模式家族；10 h 后保留 >98% 模式间 mutual information，时间精度约 ±4 ms | 局部级联可形成丰富、可重复且长时稳定的群体模式 | 体外切片 + LFP 阈值事件；只满足部分“记忆基底”条件，不是行为任务证明 |
| [Shew et al. 2009](https://pubmed.ncbi.nlm.nih.gov/20007483/) / [2011](https://pubmed.ncbi.nlm.nih.gov/21209189/) | 皮层培养物药理干预 + 模型 | 改变 E/I 后，动态范围、Shannon entropy 和 stimulus-response mutual information 在中间 E/I / avalanche 区附近达峰 | 对“过弱传播与过强同步之间存在功能工作区”提供因果干预证据 | 不证明系统精确停在数学临界点，也不证明临界性自动产生复杂认知 |
| [Eytan–Marom 2006](https://pubmed.ncbi.nlm.nih.gov/16914671/) / [Pasquale et al. 2017](https://pubmed.ncbi.nlm.nih.gov/28749937/) | 培养神经网络 MEA | 少量 early-to-fire 神经元和局部高递归社区先放大活动，再点燃全网 network burst | 宏观事件可由局部拓扑成核，而非由所有单元均匀贡献 | 培养网络的全局 burst 也可能是过强同步，不能直接等同于有用计算 |
| [Feller et al. 1996](https://pubmed.ncbi.nlm.nih.gov/8638165/) / [retinal-wave collective model](https://pmc.ncbi.nlm.nih.gov/articles/PMC6782231/) | 发育视网膜实验 + 模型 | 随机局部激活、胆碱能邻域传播与 refractory history 形成有限区域的自发波 | 无外部感知任务时，局部活动也可生成组织连接的内部训练信号 | 属于发育期回路塑形，不证明同机制能提升成熟网络任务能力 |

### 必须保留的反证 / 禁止越界

1. **幂律不等于临界性。** [Touboul–Destexhe 2010](https://pmc.ncbi.nlm.nih.gov/articles/PMC2820096/) 表明阈值化随机过程可产生表观幂律；[Touboul–Destexhe 2017](https://journals.aps.org/pre/abstract/10.1103/PhysRevE.95.012413) 进一步表明远离临界的自维持不规则网络乃至独立随机 surrogate 也可产生类似 scaling。后续不能以单个 avalanche exponent、log-log 直线或 spectral purity 作为“临界性已证”的证据。
2. **精确临界点不是唯一解释。** [Wilting–Priesemann 2018](https://www.nature.com/articles/s41467-018-04725-4) 针对强 subsampling 提出稳健估计，鼠、猫、猴记录更符合能让输入回响数百毫秒的中间/reverberating 状态，而非简单的异步态或精确临界态二选一。VPSC 的 `βρ=1` 必须与“略次临界但可回响”的竞争假设比较。
3. **有振荡不等于有功能。** 本项目 C1 通过而 C2 失败：完整 E/I 能形成极限环，仍不能推出固定位置记忆、distractor 门控或注意力。任何新指标必须直接测信息或任务目标，不能用轨迹好看代替功能。
4. **新读出不能回写旧结论。** “cue 可能编码在相位/传播轨迹中”只能建立新命题 C2′；它不能挽救 C2 的 0/27。C2′ 必须预注册 phase-aware 指标，并加入 time-shuffle、delay-shuffle、label-shuffle 等负面对照，排除输入残留和后验挑指标。
5. **STDP 窗形不等于记忆形成。** 当前 `deep_stdp.py` 只验证窗口形状（R²≈0.82、τ≈4），且符号为 anti-Hebbian。它可能承担去相关/稳定功能，但在没有群形成、重现性与因果消融证据前，不能称为 Hebbian 记忆规则。
6. **涌现不等于智能，也不等于参数量阈值。** 波、雪崩、吸引子、同步簇和 polychronous group 是新的群体变量；它们是否改善分类、记忆、控制或适应必须分别验证。增加神经元数量本身不是充分机制，邻域数、拓扑、E/I、噪声、延迟和可塑性规则才是独立变量。
7. **体外结果的外推有限。** 培养物与切片允许干净的药理干预和长时记录，但缺少完整感觉—行为闭环；不能把 in vitro 信息指标直接外推为动物认知能力。

### 由证据导出的候选方向

以下均为待选研究命题，不是已成立结论：

| 方向 | 认识论标签 | 核心假设 | 最小判官实验 | 失败条件 |
|---|---|---|---|---|
| D1 phase/polychronous 读出 | **Cross-domain analogy** | C1 的 cue 信息位于相对相位、循环位移等变轨迹或重复时空群中，而非固定 target 节点 | 预注册 C2′；比较 phase-aware decoder 与固定位置读出，并做 time/delay/label shuffle | 轨迹不可重复，或 decoder 优势在 shuffle 后仍存在 |
| D2 E/I 信息峰 | **Established direction（VPSC 迁移未证）** | E/I 振荡区只有在中间增益范围才提高 pattern entropy、stimulus-response MI 与动态范围 | 在固定刺激集合和 E/I 网格上测连续信息指标，并与去 E、去 I、单环对照 | 振荡区无中间信息峰，或交互打乱后指标不降 |
| D3 略次临界回响态 | **Cross-domain analogy** | VPSC 的最佳工作区可能是 `βρ<1` 的长回响平台，而非精确 `βρ=1` 单点 | 扫 `βρ≈0.8–1.1`，同时测回响时间、susceptibility、MI、扰动恢复与任务指标 | 精确临界在多 seed、多指标上稳定全面占优，或不存在可重复平台 |
| D4 局部成核社区 | **Cross-domain analogy** | 少量稀疏高递归单元可预测并因果启动全局状态转换 | 从自发轨迹识别 early-to-fire/高预测力社区，做定向 lesion 与等规模随机删除 | 无稳定 seed，或定向 lesion 不强于随机删除 |
| D5 自发波预训练 | **Cross-domain analogy** | 任务训练前的内部传播波可用局部可塑性预组织时空拓扑 | 与等 spike 数、等能量随机输入比较后续样本效率、鲁棒性与连接结构 | 只产生病理同步，或不优于 matched-noise 控制 |
| D6 吸引子/quorum 分类 | **Established theoretical direction** | 分类可由局部异步状态更新形成群体承诺，减少对集中式 linear readout 的依赖 | 等参数比较 attractor/quorum 与 linear readout 的残缺输入恢复、校准和损伤鲁棒性 | 吸引子必须手工写入，或出现错误早期共识/容量崩溃 |

### 观察、解释与推测

- **观察**：神经科学中最强的共同结构不是“神经元足够多”，而是局部递归 + 相反反馈 + 时间尺度分离 + 活动依赖可塑性在中间参数区产生新的群体变量。
- **解释**：C1/C2 的混合结果与这些证据并不冲突。E/I 首先能生成动力学，但动力学是否承载信息需要独立的群体观测量；Shew 的 E/I 干预和 polychronization 模型分别提供了“测什么”与“信息可能在哪里”的候选答案。
- **推测**：当前 anti-Hebbian 更新可能类似 Boids 的 separation/homeostasis，而 Hebbian 或其他正相关机制承担 cohesion/群形成。该分工尚无本项目证据，必须通过组合规则和消融验证。

### 结论 / 决定

- **采用**上述文献集作为“局部规则 → 神经群体涌现”的证据地图；严格区分真实组织实验、计算模型和理论论证。
- **设为硬约束**：幂律、振荡、单点临界峰或规模增长均不能单独证明功能涌现；后续至少需要负面对照、因果干预和功能信息指标。
- **保持 C2 负面结论**：不修改 0/27 的预注册结果，不进入 C3。
- **优先候选**：若继续 C 线，先在 D1 与 D2 中选择并预注册新的 C2′；D1 检查相位/轨迹是否携带信息，D2 用 entropy/MI/dynamic range 作为功能判官。二者未通过前，不声称“双环注意力”。
- **将精确临界降为竞争假设**：后续把 `βρ=1` 与 D3 的略次临界回响区直接比较，而不是预设临界点必然最优。
- 本条只记录证据与方向，**未修改模型、未新增实验、未改变现有运行结果**。

### 可复现信息

- 本地事实源：本日志紧邻的 C2 条目、`lab/ring_feedback/results/c2_verify.json`、`results/deep_stdp.txt`、`results/deep_critical.txt`。
- 外部事实源：本条表格与反证段落中的原始论文或 PubMed/PMC 页面；检索日期 `2026-07-17`。
- `[待补]`：若选定 D1/D2，需要另写预注册条目，固定数据生成、刺激集合、decoder 容量、统计检验、seed 数与通过阈值。

---

## 2026-07-17：C2 双环判官实验 — C1 振荡通过，功能可分离性失败（混合结果）

### 背景 / 动机

紧接下条 C-v0.1 形式化，先检验两个逻辑上独立的问题：E/I 双环能否修复 A3 单环冻结（C1），以及“正反馈=记忆、负反馈=门控”的功能解释能否通过交叉消融（C2）。C2 预注册阈值与读出窗口见下条，本条不按结果修改。

### 实验设计

原型 `lab/ring_feedback/c2_verify.py`，$N=8$。完整系统使用 Wilson–Cowan E/I 双环和 RK4（`dt=0.04`）；对比去正反馈（`g_ee=0`）、去负反馈（`g_ei=0`）和 A3 式单 tanh 环。输入协议为 target cue（节点 0，$t\in[0,2)$）、延迟期和等幅 distractor（节点 4，$t\in[6,8)$）。

基准参数先经**探索性搜索**找到一个可振荡点，再固定为 `g_ee=7.75, g_ei=6.70, g_ie=10.0, g_ii=6.30, tau_i=5.80`。因此 C1 基准点只证明存在性，不是无偏的参数空间命中率。固定后另扫 `g_ee`、`g_ei` 的 $0.9/1.0/1.1\times$ 局部网格与 seed 0/1/2，共 27 组，用于检查是否只是孤立精调点。

### 结果

**C1 自主动力学（基准 seed=0）：**

| 条件 | late std | spectral purity | 主频 | 判定 |
|---|---:|---:|---:|---|
| 完整 E/I | 0.3206 | 0.9596 | 0.0667 | PASS |
| 去正反馈 | 0.000021 | 0.9946 | 0.0333 | FAIL（近似定点） |
| 去负反馈 | $1.45\times10^{-12}$ | 0.0001 | 0.0333 | FAIL（饱和定点） |
| A3 单环 | $2.68\times10^{-11}$ | 0.0315 | 0.0333 | FAIL（饱和定点） |

完整 E/I 的相图形成闭轨，状态保持在 `[0.0136, 0.9152]`。局部网格 C1 通过 **21/27（77.8%）**：振荡不是单一数值点，但该比例仍以探索性选中的中心为条件，不能外推为全局参数空间比例。

**C2 交叉消融（$M$=延迟末目标注意力，$G$=目标减 distractor 注意力）：**

| 条件 | $M$ | $G$ |
|---|---:|---:|
| 完整 E/I | 0.0203 | -0.1760 |
| 去正反馈 | 0.1044 | -0.2699 |
| 去负反馈 | 0.1000 | -0.0647 |
| A3 单环 | 0.1260 | -0.0019 |

干预效应为：

```
Δ_positive^M = M_full - M_noPositive = -0.0841  （要求 > +0.10）
Δ_negative^M = M_full - M_noNegative = -0.0797  （交叉效应）
Δ_negative^G = G_full - G_noNegative = -0.1113  （要求 > +0.10）
Δ_positive^G = G_full - G_noPositive = +0.0939  （交叉效应）
```

四项判据全部失败；局部网格 C2 通过 **0/27（0%）**。

### 观察与解释

- **观察（C1）**：只有完整 E/I 保持大幅、窄带、有限振荡；去任一反馈都收敛到定点。A3 的“单环会冻结”边界被复现，而双环确实补上了产生极限环所需的迟滞负反馈。
- **观察（C2）**：完整 E/I 在固定延迟读出时的 target attention 只有 0.0203，低于均匀基线 $1/N=0.125$；$G<0$ 表示 distractor 占优。去掉负反馈反而把 $G$ 从 -0.1760 改善到 -0.0647，效应符号与“负反馈负责门控”的预测相反。
- **解释**：E/I 在本模型里首先是一个**不可分的振荡器对**，不是两个可独立贴上“记忆/门控”标签的模块。负反馈既限制增益也决定相位；正反馈既放大 cue 也放大 distractor，两个环对两个指标均有强耦合。
- **限制**：$M$ 是固定时间窗、固定位置的瞬时注意力。完整系统可能把 cue 编码在环上传播的相位/位置中，而不是保留在原 target；这可以解释 $M$ 很低，但不能挽救当前 C2，因为 C2 明确定义的就是目标位置记忆和 distractor 门控，且 $G$ 的符号也反向。

### 结论 / 决定

- **C1 暂时采用**：E/I 双环能在一个局部参数邻域产生 A3 单环没有的有界极限环；这是 C 线第一项正面结果。
- **C2 强解释不采用**：在 C-v0.1 与预注册读出下，“正反馈=记忆、负反馈=门控”被 0/27 的结果否定。不能由“有振荡”直接推出“是注意力机制”。
- **暂停 C3**：不进入任务准确率实验。若继续，应先提出新的可观测量（例如 phase-aware/循环移位等变读出），并把它作为新命题 C2' 重新预注册，不能事后替换当前失败指标。

### 可复现信息

- 命令：`python3 lab/ring_feedback/c2_verify.py`
- 原型：`lab/ring_feedback/c2_verify.py`
- 原始数值：`lab/ring_feedback/results/c2_verify.json`
- 图：`lab/ring_feedback/results/c2_verify.png`
- 运行：CPU，NumPy float64，RK4 `dt=0.04`，seed 0/1/2。

---

## 2026-07-17：C 线形式化 — 兴奋/抑制双环反馈注意力（形式化完成；C2 结果见上条）

### 背景 / 动机

A 线给出了三个负面边界：A2 表明一般 softmax 转移矩阵不能严格写成连续时间生成元的单位时间指数；A3 表明有向单环无论在线性半群还是单调 tanh 动力学下，都不能单独产生持续时序；A1 的近似算子实验则受 toy 任务限制而无效。A3 的失败把缺失结构定位到**兴奋/抑制（E/I）双环**：单一正反馈会衰减或饱和，持续动力学需要负反馈提供相位滞后与增益限制。

C 线不再把注意力视为一次性权重矩阵，而把它视为一个由正、负反馈共同调谐的动态增益场：**正反馈积累并传播被选中的痕迹（记忆），负反馈抑制无关扰动并限制增益（门控）**。

### 数学形式化

令 $e(t),i(t)\in[0,1]^N$ 分别为 $N$ 个 token/位置上的兴奋与抑制状态，$R$ 为有向循环移位矩阵（$R^N=I$）。定义两个闭环传播核

$$
K_E=(1-\rho_E)I+\rho_E R,\qquad
K_I=(1-\rho_I)I+\rho_I R^\top,
$$

并采用 Wilson–Cowan 型连续动力学

$$
\tau_E\dot e=-e+\sigma\!\left(g_EK_Ee-g_IK_Ii+Bu(t)-\theta_E\right),
$$

$$
\tau_I\dot i=-i+\sigma\!\left(h_EK_Ee-h_IK_Ii-\theta_I\right).
$$

其中 $g_EK_Ee$ 是正反馈环，$g_IK_Ii$ 是回到兴奋态的负反馈环；$\tau_I>\tau_E$ 使抑制反应滞后，从而可能把“放大后冻结”改成有界振荡。动态注意力读出定义为

$$
a(t)=\operatorname{softmax}\bigl(\kappa[e(t)-\lambda i(t)]\bigr),
\qquad y(t)=\sum_j a_j(t)v_j.
$$

这一定义中的“环”是状态递归闭环，不等于 token 图中仅仅存在拓扑环；C 线的最小新增主张是 **E/I 两个符号相反、时间常数不同的闭环产生可控时序增益**。

### 可证伪命题

- **C1（双环动力学）**：存在一个非零体积的参数区域，使 E/I 完整系统在脉冲输入撤去后收敛到有界、非定点的极限环；去掉 E 或 I 反馈后，该持续振荡消失。单个精调参数点不算通过。
- **C2（功能可分离性，判官实验）**：在同一组固定参数和输入下，去掉正反馈主要破坏延迟期记忆，去掉负反馈主要破坏干扰期门控；交叉影响必须显著更小。
- **C3（注意力意义，后续）**：若 C1/C2 成立，动态读出 $a(t)$ 应在有干扰的时序选择任务上优于无环、单环和静态 softmax 基线。C2 未通过前不进入 C3。

### C2 预注册实验

原型采用固定的脉冲—延迟—干扰协议，并比较四个条件：完整 E/I、去正反馈（`g_E=0`）、去负反馈（`g_I=0`）、A3 单环基线。所有条件共享初态、输入、积分器和其余参数。

1. **自主动力学检验**：输入撤去后的后半段需满足 `late_std > 0.02`、状态有界，且主频谱功率占比 `spectral_purity > 0.5`；完整 E/I 需通过，两个消融需失败。
2. **记忆指标 $M$**：cue 撤去后延迟末端，目标位置相对基线的注意力质量。定义 $\Delta_E^M=M_{full}-M_{noE}$、$\Delta_I^M=M_{full}-M_{noI}$。
3. **门控指标 $G$**：注入等幅 distractor 后，目标相对 distractor 的注意力优势。定义 $\Delta_I^G=G_{full}-G_{noI}$、$\Delta_E^G=G_{full}-G_{noE}$。
4. **可分离通过标准**：$\Delta_E^M>0.10$、$\Delta_I^G>0.10$，且 $\Delta_E^M>2|\Delta_I^M|$、$\Delta_I^G>2|\Delta_E^G|$。同时在预先固定的局部参数网格与多个 seed 上报告通过比例，避免只展示单点。

若完整 E/I 没有极限环，则 C1 失败；若有极限环但上述 2×2 干预矩阵不呈对角占优，则“正反馈=记忆、负反馈=门控”的强解释失败。两者必须分开报告，不能用任务准确率掩盖机制失败。

### 结论 / 决定

- **采用**上述 E/I 双环方程作为 C 线第一版严格对象。
- **已执行**：`lab/ring_feedback/c2_verify.py` 完成 C1 自主动力学与 C2 交叉消融，结果见上条。
- **边界保持**：本条只定义实验前命题，不根据结果改写阈值；有效性结论以上条实测为准。

### 可复现信息

- 形式化版本：C-v0.1（Wilson–Cowan E/I 双环 + 动态 softmax 读出）。
- 预定原型：`lab/ring_feedback/c2_verify.py`。
- 预定产物：`lab/ring_feedback/results/c2_verify.{json,png}`。

---

## 2026-07-17：A1 近似热核注意力 — 实验无效（任务设计失败，非理论证伪）

### 背景 / 动机

A2 严格等式被嵌入性证伪后，A1 退而求其次：放弃"softmax=exp(τ(P-I))"等式，问 exp(τ(P-I)) 作为**近似注意力算子**（τ 可学习）有没有工程价值。本条记录原型结果：实验无效，原因是任务设计而非理论。

### 假设与成功标准

- **H_A1**：exp(τ(P-I)) 作为近似注意力，τ 可学习时能与标准 softmax 竞争；固定 τ 扫描应出现精度峰（τ 是有意义旋钮）。
- 成功：R1（τ 峰存在）、R2（可学习 τ 匹配/超越固定 τ=1 且接近 softmax）、R3（大 τ 过平滑显现）。

### 实验设计

原型 `lab/attn_diffusion/a1_approx.py`。多头 semigroup 注意力 exp(τ(P-I))，P=softmax(QK)。τ 固定扫描 vs 可学习（每头一个 log τ）。对比标准 softmax baseline。任务：32 token、8 维、10 类，cue token 的类 = 样本标签，其余 token 是随机类 distractor，mean-pool 读出。

### 结果

```
固定 τ 扫描: 0.10→0.176, 0.30→0.178, 0.50→0.173, 0.80→0.180, 1.0→0.181,
             1.5→0.186, 2.0→0.186, 3.0→0.180, 5.0→0.184, 8.0→0.188
  峰 τ*=8.0 acc=0.188, interior=False, rises=False, falls=False
标准 softmax:        0.165
semigroup 固定 τ=1:  0.181 (gap vs softmax: -0.016)
semigroup 可学习 τ:  0.176, learned τ=[1.07, 1.76, 1.31, 1.45]
R1 (τ 峰): FAIL   R2 (可学习竞争): PASS(假象)   R3 (过平滑): FAIL   R4: PASS(假象)
```

### 观察与解释（关键：实验无效诊断）

- **观察**：所有方法都在 17–19%（随机 10%），精度对 τ 几乎不敏感（0.173–0.188 平坦）。
- **解释**：**任务设计失败，非理论证伪**。32 token × 8 维、每个都是 prototype+噪声、mean-pool 读出——**架构上无法定位 cue token**（mean-pool 把所有 token 混匀，attention 选择性被 pool 抹掉）。任务在当前架构下无解，τ 怎么调都接近随机+一点。R2/R4 的 PASS 是阈值过松的假象（差距在噪声内）。
- **观察（微弱真实信号）**：semigroup 固定 τ=1 (0.181) 略高于标准 softmax (0.165)，符合 A2 发现——exp(P-I) 是 softmax 的近似变体，性能相近但不同。差距太小，不能下结论。
- **解释（A1 本质局限）**：A1 的核心问题（近似热核替代 softmax 有没有用）在 toy 规模无法回答——需要真实任务（NLP/视觉）+ 大模型才能判别 τ 可学习的收益。原型规模不足。

### 什么改变了

- A1 **未被证伪也未被证实**——实验无效（任务设计失败）。
- 不再继续调 A1 任务设计：继续调是工程打磨，不是理论验证；且 A1 是 A 线最弱分支（近似算子，理论增量薄）。

### 结论 / 决定

- **搁置** A1（实验无效，原型规模不足以下结论）。代码 `a1_approx.py` + results 保留。
- **转 C 线**：C 线有明确的理论判官实验（E/I 双环是否产生极限环、正/负反馈是否可分离），不依赖难任务设计。A3 证伪已指向 C（单 tanh 环冻结→需 E/I 双环）。
- **A 线总结**：A2（嵌入性证伪）、A3 线性+非线性（均证伪）、A1（搁置）。A 线三个子方向无正面结果，但产出三个有价值的负面边界。

### 可复现信息

- 原型：`lab/attn_diffusion/a1_approx.py`（`python lab/attn_diffusion/a1_approx.py --epochs 50`）。
- 产物：`lab/attn_diffusion/results/a1_approx.{json,png}`。
- bug 修正：`semigroup_attn` 3D→4D 支持；`train_eval` 处理 tau=None。

---

## 2026-07-17：涌现机制调研 — "加参数让 SNN 涌现"支持度极低（调研结论）

### 背景 / 动机

回溯第一轮对话的原始问题："脉冲网络低参数量表现不佳，是不是还没出现像 GPT-3 那样的涌现？" A2/A3 证伪后，用户重提此问：有没有可能增加参数量就出现涌现、实现 SNN 突破？先调研，再测 A1，最后转 C。

### 假设与成功标准

- **待评估直觉**："增加参数量 → SNN 涌现 → 突破"。
- 成功标准：文献核实涌现机制，判断参数量是否关键变量，给出 SNN 涌现路径的诚实判断。

### 调研结果（subagent 核实，arXiv/Transformer Circuits/Wikipedia 直连）

**涌现机制（问题1）：**
- Wei 2022（arXiv:2206.07682）定义涌现为"小模型无、大模型有、不可外推"，但归因笼统的"scale"，**未区分参数/数据/算力**。
- Schaeffer 2023（arXiv:2304.15004，NeurIPS 2023）：部分涌现是**评估指标非线性造成的假象**，换连续指标即消失；并在视觉任务上故意制造出"前所未见"的涌现。学界有反驳但未核实原文。
- **induction heads（Olsson 2022）**：最强机制解释——注意力+≥2 层形成 induction head 电路时出现相变（loss bump），**任意大小、层数>1 即出现**，关键在结构非尺寸。作者自述 circumstantial。
- Kaplan 2020（arXiv:2001.08361）：loss 对 N/D/C 平滑幂律，**无阈值**。Hoffmann 2022 Chinchilla（arXiv:2203.15556）：compute-optimal 需 N×D 等比（~20 token/参数），欠训练大模型被小模型反超。
- grokking（Power 2022, arXiv:2201.02177）：训练动力学相变，与涌现是类比非源证明。

**参数量是否关键（问题2）：文献整体反对参数量决定论。**
- Chinchilla 直接反对：N×D 等比，70B Chinchilla > 280B Gopher。
- Kaplan 平滑幂律无参数阈值；Schaeffer 质疑表观阈值是假象；induction heads 阈值在结构非尺寸。
- Faith and Fate（Dziri 2023, arXiv:2305.18654）：transformer 组合推理靠线性化子图匹配，堆参数未必换系统性能力。

**SNN 涌现路径（问题3）：**
- 最大 SNN（Spikformer V2 16层，arXiv:2401.02020）≈172M 参数，比 LLM 涌现阈值小 3-4 个数量级。
- **没有任何经同行评审的 SNN 能力涌现报告**；所有"SNN emergent"用词均为生物动力学（振荡/同步）含义。
- SNN 训练障碍（源证明）：替代梯度不精确（arXiv:2605.27412）、二值激活信息瓶颈（arXiv:2606.23761）、BPTT 显存（arXiv:2602.22259）、在线无监督精度低（arXiv:2606.30926）。
- **SNN scaling law 文献极稀疏**（仅 arXiv:2601.14961，小规模 LIF，发现精度主要随类别数幂律、神经元数影响小）；**SNN vs ANN scaling 斜率比较——完全空白**，这是决定"加参数能否涌现"的核心未知量。
- Hala Point（1152 颗 Loihi 2，11.5 亿神经元/1280 亿突触）规模够大但精度/ANN-SNN 转换受限，未证实能跑 transformer 规模。

### 观察与解释

- **观察**：涌现机制文献指向"结构电路 + 训练阶段相变"（induction heads），非绝对参数量；Chinchilla 证明 N×D 等比才算 compute-optimal。
- **解释**："加参数就涌现"是**朴素直觉，与现有证据方向相反**。LLM 侧参数量都不是独立驱动因素；SNN 侧连平滑 scaling law 都未建立，遑论离散涌现。
- **解释（SNN 障碍排序）**：训练方法（替代梯度/BPTT/二值激活）> 结构（缺注意力残差/归一化成熟配方、induction-head 式电路未被识别）> 硬件（Hala Point 精度受限、GPU 生态不兼容）> 参数量（前三者受限的结果）。
- **推测**：SNN 的 scaling curve 斜率可能低于 ANN（因替代梯度信息损失），若真如此，即使加参数到 LLM 规模也不会涌现——但这是未测试的核心未知量。

### 关键经验（transferable）

1. **涌现≠参数量的函数**。任何"加参数就能涌现"的论证都需先回答：compute-optimal 数据量够吗？结构有 induction-head 式电路吗？评估指标是否连续？
2. **SNN 涌现的核心未解问题是 scaling law 斜率**。在没建立 SNN 平滑 scaling law、并和 ANN 比斜率之前，"SNN 能否涌现"是空谈。
3. **指标陷阱**：测涌现必须用连续指标 + 置信区间，预注册负面对照，排除 ANN→SNN 蒸馏泄漏（SpikeBERT 模式）和生物动力学"涌现"的语义偷换。

### 结论 / 决定

- **不采纳** "增加参数量就能让 SNN 涌现"的直觉——文献支持度极低（Speculative，与证据反向）。
- **采纳** 调研给出的诚实实验设计作为远期参照：先建 SNN 平滑 scaling law（扫 N×D×C，比 ANN 斜率）→ 连续指标 → compute-optimal frontier 扫到 1B-10B → 机制探针 → 预注册负面对照。这是"测 SNN 能否涌现"的唯一诚实路径，但规模远超当前原型能力。
- **当前可做**：A2/A3 证伪已表明，SNN 注意力的连续扩散形式化走不通（嵌入性障碍 + 单环不涌现时序）。下一步按用户计划测 A1（连续热核作近似算子），再转 C 线（E/I 双环）。
- **待补**：SNN vs ANN scaling 斜率比较是决定性实验，但需大规模训练，超出本原型范围，标记为远期。

### 可复现信息

- 调研 subagent：agentId ac9e1507713d47ef5，输出存会话 task 目录。
- 关键文献：Wei arXiv:2206.07682、Schaeffer arXiv:2304.15004、Olsson induction-heads（transformer-circuits.pub）、Kaplan arXiv:2001.08361、Hoffmann arXiv:2203.15556、Spikformer V2 arXiv:2401.02020、SNN scaling arXiv:2601.14961、Loihi/Hala Point Wikipedia。

---

## 2026-07-17：A3 非线性版也被证伪 — tanh 单环收敛到饱和不动点（负面结果，A3 终结）

### 背景 / 动机

A3 线性版证伪后（见下条），log 标注"真正不收敛需非线性（tanh 饱和）"。本条测非线性版——A3 最后机会。

### 假设与成功标准

- **H_A3（非线性版）**：tanh 均场环节点 + 有向环产生持续振荡（极限环），轨迹承载时序结构。
- 成功：Q1（环持续振荡、链塌缩）、Q2（环轨迹拟合时序 R²>0.5 且 >> 链）、Q4（late_std>0，非冻结）。

### 实验设计

原型 `lab/attn_diffusion/a3_nonlinear.py`。两版动力学：
1. **离散** $x_{t+1}=(1-\alpha)x_t+\alpha\tanh(Wx_t)$ — 首版，全部 FAIL。
2. **连续 ODE** $\dot{x}=-x+\tanh(Wx)$，RK4 — 修正版（离散版 leak 平均掉了旋转）。
有向环 W（`W[i,(i+1)%N]=scale`），扫 scale，对比 chain（DAG）。线性化 Jacobian $J=-I+W$，Hopf 分岔需 `Re(eig(J))>0`。

### 实现与踩坑

- **离散版失败**：Q4 `late_std=0.0000`，tanh 饱和把状态推向 ±1 角落后冻结到不动点。leak 平均掉了环的旋转。
- **改 ODE**：连续时间保住旋转，能量不再衰减到 0，但仍未达极限环。
- **参数扫描**：scale 1.0/1.5/2.0/3.0/5.0，total_time 30→120。

### 结果（ODE 版 + 长时诊断）

```
线性化 max Re(eig(-I+W)): scale=1→0.0, 1.5→0.5, 2→1.0, 3→2.0, 5→4.0  (全部 >=0, Hopf 已跨过)
长时能量 (total_time=120): scale=2 E(end)=14.669 late_std=0.0000
                          scale=3 E(end)=10.158 late_std=0.0302
                          scale=5 E(end)=15.997 late_std=0.0000
Q1 cycle sustained? True  chain sustained? True  → FAIL（链也振荡，判别失效）
Q2 best cycle R^2=-0.038  chain R^2=-0.392  → FAIL（接近 0 但未达 0.5）
Q3 corr(N,freq)=+0.796  → FAIL（应为负）
Q4 late_std=0.0034  → FAIL（冻结）
```

### 观察与解释（决定性）

- **观察**：线性化在原点全部不稳定（`Re(eig)>0`，Hopf 分岔已跨过），能量不衰减到 0（scale=2 稳在 14.67）——**未塌缩到原点**。但 `late_std=0.0000`：能量稳在常数，轨迹收敛到**饱和不动点**，**非极限环**。
- **解释**：tanh 是**单调饱和**。产生极限环需"旋转 + 限幅"，但 tanh 单调性把 spiral-out 拉到一个固定饱和角，而非周期轨道。标准神经振荡器（Wilson-Cowan、FitzHugh-Nagumo）需**兴奋-抑制双群体（E/I）**才产生极限环——单 tanh 均场环不行。
- **解释（连回 C 线）**：A3 单环证伪，恰因缺 E/I 结构。**C 线的核心预测——正反馈（兴奋）+ 负反馈（抑制）双环配对才能产生持续动力学——得到间接支持**。A3 的失败不是否定"环产生时序"，而是否定"单环 + 单调非线性产生时序"。

### 什么改变了

- A3 **非线性版证伪**。A3 线性 + 非线性**两版均证伪**，A3 **彻底终结**。
- **保留**：A3 的失败精确刻画了边界——单环不够，需 E/I 双环（→ C 线）。

### 与 A2 的对比

- A2：数学等式错（嵌入性），无救。
- A3：线性版 + 单调非线性版均失败，但失败**指向了 C 线的 E/I 双环机制**——这是有方向性的负面结果，不是死路。

### 结论 / 决定

- **不采用** A3（彻底证伪）。A 线剩 A1（连续热核作近似算子，绕过嵌入性，未测）。
- **转向 C 线**：A3 的失败为 C 线的"正反馈+负反馈双环"提供了动机——单 tanh 环冻结到不动点，E/I 双环（Wilson-Cowan 式）才可能产生极限环。C2 判官实验（正/负反馈是否可分离）成为下一优先。
- **待补**：若 C 线 E/I 双环能产生极限环 + 拟合时序，则"环→涌现时序"命题在 C 线框架下复活；否则整个"环→时序"方向证伪。

### 可复现信息

- 原型：`lab/attn_diffusion/a3_nonlinear.py`（`python lab/attn_diffusion/a3_nonlinear.py`），含离散 `simulate` + ODE `simulate_ode` 两版。
- 诊断：线性化 `eig(-I+W)`、长时能量 `late_std`。
- 产物：`lab/attn_diffusion/results/a3_nonlinear.{json,png}`。

---

## 2026-07-17：A3 线性版被证伪 — 环不产生涌现时序（负面结果）

### 背景 / 动机

紧接 A3 形式化条目。线性半群原型验证 4 个可证伪预测，全部 FAIL。本条记录证伪证据与一个理论错误的修正。

### 假设与成功标准

- **H_A3（线性版）**：含环图上 $x(t)=e^{t(P-I)}x(0)$ 产生承载时序结构的轨迹，不显式给时间索引即可拟合时序任务；DAG 不能。
- 成功：P1（环复特征值/DAG 实）、P2（环轨迹拟合时序显著优于 DAG）、P4（持续模非塌缩）。

### 实验设计

原型 `lab/attn_diffusion/a3_verify.py`（CPU，N=16–24）。环图（有向 directed cycle，修正后）vs DAG，行随机 P，线性半群 $e^{t(P-I)}$，ridge 线性读出拟合 sin 周期信号（无时间索引）。

### 实现与踩坑（关键）

- **第一版 bug**：`cycle_P` 用了**无向对称环**（`A[i,i±1]=1`），对称矩阵特征值必实 → P1 测出环 |Im|=0.0000、DAG |Im|=0.5476，与命题完全相反。
- **修正**：改为**有向环**（`A[i,(i+1)%N]=1`，非对称），环 |Im| 升至 0.21。但**修正后命题仍错**——DAG |Im|=0.55 仍大于环。
- **理论错误（实锤）**：复特征值来自**有向性/非对称性**，不来自环性。原命题"环→复特征值→振荡"的谱论证不成立。对称环全实，有向 DAG 也有复。

### 结果（修正后有向环，原始运行）

```
P1  cycle: |Im|=0.2102 gap=0.7898   DAG: |Im|=0.5476 gap=0.3013
    autocorr half-life: cycle=2.83  DAG=1.82   → FAIL（环应更长但差距小）
P2  cycle: test R^2=-4.449  DAG: R^2=-23.752  const-baseline MSE=0.6555
    cycle MSE=1.053  DAG MSE=4.783   → FAIL（都极差，cycle 略好但 R^2<0）
P3  smooth 0.5/1/2/4/8 → gap 0.96/0.90/0.71/0.23/0.005 → horizon(R^2>0.5) 全 0
    corr(gap,horizon)=0.000   → FAIL
P4  cycle 持续模(|λ|>0.9 且 |Im|>0.05)=0
    轨迹能量 t=0→31.014, t=30→0.359 (ratio 0.012)   → FAIL（指数塌缩，无持续模）
```

### 观察与解释

- **观察**：环轨迹能量 30 单位时间后衰减到 1.2%；无 $|\lambda|>0.9$ 的复模；ridge 读出 R²<0。
- **解释**：线性半群 $e^{t(P-I)}$ **总收敛到平稳分布 π**（A2 的 P3 已验证），环只改变收敛速率（谱间隙），**不产生不终止轨迹**。A3 的"时序"在线性设定下被指数塌缩压缩掉。
- **解释（理论错误）**：复特征值来自非对称性（有向性），不来自环性。"环→复模→振荡"是错的谱论证。
- **观察（微弱信号）**：P2 中 cycle MSE=1.05 确实优于 DAG 4.78，说明环轨迹比 DAG 轨迹**略多**携带时序信息，但远不足以拟合（R²<0）。

### 什么改变了（相对 A3 形式化条目）

- A3 形式化的**谱论证部分被推翻**：复特征值≠环性，是有向性。
- A3 **线性版证伪回退**：线性半群下环不产生涌现时序（P4 持续模=0、能量塌缩）。
- **保留**：A3 形式化里自己标注的诚实边界——"真正不收敛需非线性（tanh 饱和）或谱间隙=0"。线性证伪的是**线性版 A3**，非线性版未测。

### 与 A2 的区别（重要）

- **A2**：数学等式错误（嵌入性），**无救**——softmax 无法无损嵌入连续半群。
- **A3 线性版**：线性设定失败，**非线性可能救**——VPSC 的 tanh 均场环节点 + 有向环可能产生真极限环/持续轨迹（饱和非线性可维持振荡，线性不行）。
- A3 的命运取决于非线性实验，尚未定论。

### 结论 / 决定

- **不采用** A3 线性版（证伪）。
- **下一步**：测非线性版 A3——用 tanh 均场环节点 + 有向环，检验能否产生持续振荡轨迹（能量不塌缩）+ 拟合时序任务。这是 A3 的最后机会。
- **待补**：若非线性版仍塌缩/提不出时序，A3 彻底证伪，A 线剩 A1（近似热核）。

### 可复现信息

- 原型：`lab/attn_diffusion/a3_verify.py`（`python lab/attn_diffusion/a3_verify.py`）。
- 产物：`lab/attn_diffusion/results/a3_verify.{json,png}`。
- bug 修正记录：`cycle_P` 无向→有向（见代码注释）。

---

## 2026-07-17：A3 形式化 — 环→涌现时序（进行中，理论就绪待验证）

### 背景 / 动机

A2 严格等式被嵌入性问题证伪（见下条），但 A2 的 P5（环谱间隙小→τ→∞ 收敛慢）成立，且 A2 失败不波及 A3——A3 不依赖"softmax=半群"等式，只依赖"环让传播慢/有持续模"。A3 是 A 线最独特、最推测的方向：**环→传播不终止→时序内生于态演化**。本条记录 A3 的理论形式化，代码原型随后。

### 假设与成功标准

- **H_A3**：在含环图上，激活传播的连续态演化 $x(t)=e^{t(P-I)}x(0)$ 产生承载时序结构的轨迹；不显式给时间索引，仅靠轨迹投影能拟合时序任务。DAG/树控制组不能。
- 成功标准：(1) 环产生复特征值/振荡轨迹，DAG 无；(2) 环轨迹拟合时序任务精度显著高于 DAG；(3) 谱间隙越小→可表达时序越长。

### 命题形式化

图 G，行随机转移矩阵 P，连续时间态演化：
$$\dot{x}=(P-I)x,\quad x(t)=e^{t(P-I)}x(0)$$
P 特征值 $\lambda_k=r_k e^{i\theta_k}$（Perron-Frobenius $|\lambda_k|\leq 1$，$\lambda_0=1$）。谱分解：
$$x(t)=\sum_k c_k e^{(\lambda_k-1)t}v_k=\sum_k c_k e^{(r_k-1)t}e^{i\theta_k t}v_k$$

**拓扑谱差异（核心）**：
- **DAG/树**：特征值全实、$r_k<1$ → 每模 $e^{(r_k-1)t}$ 单调衰减，无振荡，态 O(1/谱间隙) 内塌缩到 $\lambda_0$。时序信息快速消失。
- **环（cycle）**：特征值**复共轭对** $e^{\pm i\theta_k}$，$r_k\to 1$（谱间隙小）→ 模 $e^{i\theta_k t}$ **振荡不衰减**，态沿环持续旋转。多 $\theta_k$ 叠加 → 永不重复的轨迹，携带时序。

### 可证伪预测（原型需验证）

1. **P1**：环图 $e^{t(P-I)}x(0)$ 轨迹有持续周期分量（轨迹自相关不快速衰减）；DAG 无。
2. **P2（判官）**：不显式给时间索引，仅用 $x(t)$ 轨迹投影拟合时序任务（复制/周期预测），环显著优于 DAG。
3. **P3**：谱间隙↔可拟合时序长度单调——环谱间隙越小，可表达时序越长。
4. **P4（失败判据）**：若环轨迹指数收敛到稳态（无持续模）、时序被压缩到单一指数速率，命题被证伪。

### 观察与解释（理论层，待数值验证）

- **解释**：环的复特征值来自其循环对称性（环的邻接矩阵是循环矩阵，特征值 $2\cos(2\pi k/N)$；行随机化后仍复）。DAG 的特征值全实因其上三角结构。
- **推测**：脉冲/均场环节点的非线性（tanh 饱和）可能把"长瞬态"变成"真极限环"——线性半群终收敛到 π，非线性可维持振荡。这部分接 VPSC 的均场层，待 A3 验证后再接。

### 诚实边界 / 风险

1. **与 RNN 重叠**：环上 $e^{t(P-I)}$ 本质是线性 RNN。A3 增量必须是非线性脉冲节点 + 图结构环，否则就是重发明 RNN。原型先用线性半群验证"环 vs DAG"的谱/轨迹差异（最干净的判别），非线性扩展留后。
2. **"不收敛"的精确性**：严格说 $e^{t(P-I)}$ 总收敛到 π（A2 的 P3 已验证）。A3 的"时序"是**收敛前的长瞬态**（谱间隙小→瞬态长），不是永恒。真正不收敛需非线性或谱间隙=0。必须诚实标注。
3. **谱间隙=0 的退化**：若环被确定性化（P 近置换矩阵），谱间隙→0，收敛无穷慢，但表达能力退化（几乎不混合）。存在精度-时序长度的权衡。

### 结论 / 决定

- **采用** A3 形式化作为 A 线主线（A2 已证伪回退）。
- **下一步**：写原型 `lab/attn_diffusion/a3_verify.py`，线性半群验证 P1–P4，判官实验是 P2（环轨迹拟合时序任务 vs DAG 控制）。
- **待补**：若 P2 通过，接非线性均场环节点验证"真极限环"；若 P2 失败（环无优于 DAG），A3 证伪，A 线剩 A1（近似热核）。

### 可复现信息

- 命题推导：见本条目"命题形式化"。
- 待写原型：`lab/attn_diffusion/a3_verify.py`。
- 理论参照：循环矩阵谱理论、Perron-Frobenius、Markov 链谱间隙与混合时间。

---

## 2026-07-17：A2 核心等式被证伪 — 嵌入性问题（负面结果）

### 背景 / 动机

紧接上一条（A2 修正命题推导 + 文献核实，判为正面结果）。当时推导给出"softmax 注意力 = 连续时间随机游走半群 `exp(τ(P−I))` 在 τ=1 的取值"，并计划用最小代码原型验证 5 个可证伪预测。本条记录原型运行结果：**核心等式被证伪，A2 正面判断回退**。

### 假设与成功标准

- **待验证等式（P2）**：`exp(1·(P−I)) == P`（softmax 注意力矩阵），偏差应 < 1e-4。
- 成功标准：P2 通过；τ 扫描（P4）在 τ≈1 达峰且 τ→∞ 下降。

### 实验设计

原型 `lab/attn_diffusion/a2_verify.py`（CPU，N=32/d=8 合成 + N=8/d=16 bag 分类任务）：
- P1：`exp(τ(P−I))` → I as τ→0。
- P2：`exp(1·(P−I))` == P。
- P3：`exp(τ(P−I))` → 平稳分布 π as τ→∞。
- P4（判官）：τ 扫描精度曲线，应 τ≈1 达峰、τ→∞ 下降。
- P5（speculative）：环谱间隙小 → τ→∞ 收敛慢。
- 诊断脚本：scipy `logm(P)` 检验可嵌入性（torch 无 logm）。

### 结果（原始运行）

```
P1  tau=1e-6: ||S-I||_max = 1.01e-06            PASS
P2  tau=1.0:  ||exp(P-I) - softmax||_max = 3.75e-01   FAIL
P3  tau=50:   max|row - pi| = 5.59e-08          PASS
P4  tau-sweep: 0.90/0.98/0.995/0.9975/0.995/0.9975/0.9975/0.9975/0.9975/0.9975
    peak tau*=0.75, rises-before=True, falls-after=False  → interior=False → FAIL/inconclusive
P5  gap: dense=0.8213  cycle=0.0979  → cycle 收敛慢   PASS (speculative)
```

诊断（scipy，N=8/d=4 与 N=6 对称/非对称）：
```
exp(P-I) vs P dev = 0.45
logm(P) imag max = 2.3–8.7   （对称 P 也有 2.3）
exp(logm(P).real) vs P dev = 0.04–0.34
logm(P) vs (P-I) dev = 29.2
```

### 观察与解释

- **观察**：`exp(P−I) ≠ P`（偏差 0.375–0.45）；`logm(P)` 有大虚部（2.3–8.7）；取实部后 `exp(logm(P).real) ≠ P`（偏差 0.04–0.34）。连对称行随机矩阵都不可嵌入。
- **解释**：这是 Markov 链的**嵌入性问题**（embedding problem，Kingman 1962 / Speakman 1967）。一个行随机矩阵 P 能写成 `exp(τQ)`（Q 为 Markov 生成元，非对角元非负）当且仅当 P "可嵌入"——这是严格子集，绝大多数随机矩阵不满足。softmax 注意力 P 是**离散一步转移矩阵**，一般没有连续时间生成元。`P−I` 不是 P 的生成元；`logm(P)` 不是合法 Markov 生成元（有虚部）。
- **解释（对文献）**：这很可能正是 Candanedo、Lin、Roffo 三篇都停在"单步 Markov 算子 P"而**不取指数成连续半群**的原因——他们可能知道嵌入性障碍。A2 的"连续 τ 半群"路线被这个经典理论否决。
- **推测**：`exp(τ(P−I))` 仍是 P 的某种连续松弛（P1/P3 成立），只是不经过 P。它可作为"注意力插值器"（τ→0 恒等、τ→∞ 平稳分布），但与 softmax 是**近似关系，非等式**。

### 什么改变了（相对上一条）

- 上一条判 A2"正面结果，新颖性未被吃掉"——**该判断基于纯文献核实，未跑数值验证**。文献新颖性结论仍成立（无人占据 `exp(τ(P−I))` at τ=1 的半群陈述），但**这个陈述本身数学上错误**。新颖但错，无价值。
- **回退**：A2 修正命题的核心等式（softmax = `exp(τ(P−I))` at τ=1）**证伪回退**。原 H0（小 t 极限）此前已证伪；修正版也被证伪。A2 作为"严格等式形式化"**不可用**。

### 保留的正面部分

- P1、P3、P5 成立：`exp(τ(P−I))` 作为**近似插值器**（τ→0→I、τ→∞→π、环慢收敛）行为正确。
- P4：精度在 τ≈0.75–1 达峰，但任务太简单（bag 分类近饱和），τ>1 未下降，**判官实验未真正执行**——需更难任务才能判定 τ→∞ 过平滑是否伤精度。
- 嵌入性障碍本身是**有价值的负面结果**：它解释了为何现有工作走单步路线，为后续 A 线工作划清边界。

### 诚实风险 / 边界

- 若改用"可嵌入的近似 P"（如对称化、或限制 P 为 `exp(Q)` 族），可绕过嵌入性障碍，但牺牲 softmax 的精确形式——退化为 A1（连续热核注意力作为近似算子），不再是严格等式。
- P4 判官实验在简单任务上未真正执行，"τ→∞ 伤精度"未验证。
- `exp(τ(P−I))` 对大 τ 数值稳定，但稠密 O(N³)，需 Chebyshev/Lanczos 才可扩展。

### 结论 / 决定

- **不采用** A2 作为严格等式形式化（核心等式被证伪）。
- **保留** `exp(τ(P−I))` 作为**近似注意力插值器**的可用性（P1/P3/P5 成立），标记为"近似而非等式"。
- 负面结果存档：原型 `lab/attn_diffusion/a2_verify.py` + `results/a2_verify.{json,png}` 保留，代码注释已标注证伪。
- **下一步（待用户定）**：A 线剩余方向——A3（环→涌现时序，speculative，P5 给了部分支撑）、A1（连续热核注意力作为近似算子，绕过嵌入性）；或转向 C 线（环形反馈，C2 判官实验）。

### 可复现信息

- 原型：`lab/attn_diffusion/a2_verify.py`（`python lab/attn_diffusion/a2_verify.py --epochs 40`）。
- 诊断：scipy `logm(P)` / `expm`，torch `torch.linalg.matrix_exp`（torch 无 logm）。
- 产物：`lab/attn_diffusion/results/a2_verify.{json,png}`。
- 理论参照：Markov 链嵌入性问题（Kingman 1962, Speakman 1967）。

---

## 2026-07-17：A2 命题推导与文献核实（正面结果，含修正）

> **更新（见上条）**：本条的"正面结果"判断仅基于文献核实，核心等式 `exp(P−I)==P` 经数值验证**被证伪**。文献新颖性结论仍有效，但命题数学上不成立。以下保留原始记录。

### 背景 / 动机

源自一个关于脉冲网络注意力形式化的创造性命题：**注意力 = 激活在连通图上的传播，时间是传播过程自带的隐维度，不应作为显式张量轴**。平行展开两条调研线：
- A 线：图扩散注意力——注意力权重 = 图拉普拉斯热核 `exp(-tL)`，softmax 可能是小 t 极限。
- C 线：环形反馈注意力——正反馈=记忆、负反馈=门控，注意力=二者增益调谐。

用户选定：先推 A2（"softmax 是热核小 t 极限"的严格证明），再核实文献，理论形式化在先、代码原型在后。

### 假设与成功标准

- **原假设 H0**：softmax 注意力是图拉普拉斯热核 `exp(-tL)` 在 t→0 的极限。
- 成功标准：给出严格推导，并通过文献核实确认未被现有工作占据。

### 推导结果（关键：H0 被推翻，修正为更强命题）

推导过程中发现 **H0 字面不成立**：

- `exp(-tL)` 一般对称、行和不为 1；softmax 注意力是行随机矩阵。**两者类型不匹配**，不能直接相等。
- 改用行随机转移矩阵 `P_ij = exp(q_i·k_j/√d) / Σ_m exp(...)`（即 softmax 注意力本身就是一步随机游走转移矩阵），取生成元 `Q = P − I`，连续时间半群 `exp(τ(P−I))`。
- **小 τ 极限是恒等 I**（`exp(τ(P−I)) → I`），不是 softmax。原命题"softmax 是小 t 极限"**被证伪**。
- softmax 注意力出现在 **τ=1**，不是极限。

**修正命题 A2**：softmax 注意力 = 连续时间随机游走半群 `exp(τ(P−I))` 在 τ=1 的取值；τ 是连续传播时长参数：
- τ→0 → 恒等 I（无混合）
- τ=1 → 标准 softmax 注意力
- τ→∞ → 平稳分布 π（过平滑）
- 经相似变换 `D^{-1/2} exp(-τL_sym) D^{1/2}` 等价于对称拉普拉斯热核（热核语言成立，但仍是 τ=1 取值而非极限）。

这比原命题更强：把注意力从离散算子提升为单参数半群，τ 即"隐传播时长"。

### 可证伪预测（原型需验证）

1. τ→0：`exp(τ(P−I))` 数值上 → I，Dirichlet 能量 → 初始值。
2. τ=1：`exp(1·(P−I))` 严格等于 softmax 注意力矩阵（在 P 由同 QK 构造时）。
3. τ→∞：→ 平稳分布 π（P 的左主特征向量），与 token 内容无关。
4. **判官实验**：固定 QK，扫 τ∈(0,∞) 做传播，精度应在 τ≈1 附近达峰、τ→∞ 因过平滑下降。**若单调则命题被证伪**。
5. 环（谱间隙小）→ τ→∞ 收敛变慢 → "时序"被拉长（接 A3 涌现时序，**此环无文献支撑，纯推测**）。

### 文献核实结果（三 subagent 并行：A 线 / C 线 / A2 核实）

**核实方法**：WebFetch 直接抓 arXiv 摘要/正文 pdftotext、Crossref/OpenAlex 元数据、GitHub README。SSRN 对自动化封锁，Curry 论文经 Crossref DOI + GitHub README 间接证实。区分"源证明 / 仅暗示 / 未测试 / 未证实存在"。

**已占据（非增量）**：
- "softmax = 一步随机游走 P"（平凡陈述）——Zhao 2022 (arXiv:2211.06605) 等大量 GNN 过平滑文献。
- 堆叠层 = P^n → π = 过平滑（离散幂）。
- 注意力↔Laplacian 联系——Candanedo (arXiv:2604.09560)、Lin (arXiv:2607.10677)、Roffo (arXiv:2603.00175)，**全停留在单步 P 或生成元-Laplacian，无人取指数成连续 τ 半群**（pdftotext 全文确认 Candanedo 不含 "semigroup/heat kernel/exp(τ"）。
- 热核 + PPR 作 GNN 传播——Gasteiger 2019 (arXiv:1911.05485, 1810.05997)，GNN 语境非注意力语境。
- Curry 2025 SSRN "Heat Kernel Attention"（DOI 10.2139/ssrn.5959898，**确实存在**，Crossref+OpenAlex+GitHub 三源证实）——但用的是**连续空间高斯热核打分** `qk/2t − α·d²/4t` 追求稀疏，数学对象（标量高斯核）和目标（O(n²)→O(n·r) 稀疏）与 A2 不同，**不冲突**。

**真空白（A2 增量）**：
- softmax = `exp(τ(P−I))` at τ=1、τ 连续可调——**未发现任何已发表工作占据**。
- τ 作 I↔softmax↔π 连续插值参数——**空白**。
- PPR `(1−α)(I−αP)⁻¹` 与 `exp(τ(P−I))` 在注意力语境统一——**空白**。
- 环→谱间隙小→τ→∞ 收敛慢→涌现时序：谱间隙↔收敛有理论支撑；**"收敛慢→涌现时序"无支撑，纯推测**。

**Google Scholar 返回的候选（"Graph Diffusion Transformer" Liang&Chen、"Coherence-Diffusion Dynamics" Shin、"From Attention to Diffusion" Wang&Wang OpenReview）经 Crossref/OpenAlex/arXiv 交叉核实均无法证实为真实文献**，疑似 Scholar 片段幻觉，不计入。

### 观察与解释

- **观察**：A2 数学内核（连续时间 Markov 半群 `exp(τQ)`）是教科书内容，非新颖。
- **解释**：新颖性完全系于"显式施加到 softmax 注意力 + 连续可调 τ"这一具体形式化，以及随之的应用（可学习 τ、PPR 统一、谱间隙-时序推论）。
- **推测**：环让 P 不可约且谱间隙小，使 τ→∞ 收敛慢，于是"时序"被拉长——这是 A3（涌现时序）的理论支点，但无现有工作支撑，需自建实验。

### 诚实风险

1. 数学内核是教科书内容，发表时必须清晰切割与 Candanedo（单步 Markov）、Curry（高斯热核稀疏）的边界。
2. "收敛慢→涌现时序"是最脆弱一环，可能被证伪（指数收敛压缩时序、与 RNN 重叠）。
3. 掩码/因果性使 P 不对称、平稳分布不存在，命题需限定为无掩码正则情形。

### 结论 / 决定

- **采用** A2 修正命题：softmax = `exp(τ(P−I))` at τ=1，τ 连续可调。原 H0（小 t 极限）**证伪回退**。
- **下一步**：写最小代码原型 `lab/attn_diffusion/a2_verify.py`，验证上述 5 个可证伪预测，重点是判官实验（τ 扫描精度曲线）。
- **待补**：若 τ 扫描精度单调（非 τ≈1 达峰），A2 的实用价值需重新评估；"涌现时序"推测需独立实验。

### 可复现信息

- 命题推导：见本条目"推导结果"与"可证伪预测"。
- 文献核实：三 subagent 报告（A 线 agentId a8f1f206880d51607、C 线 a15cb9bb824f43fec、A2 核实 aa9607fa413b06b89），输出存于会话 task 目录。
- 关键文献链接：Candanedo arXiv:2604.09560、Lin arXiv:2607.10677、Roffo arXiv:2603.00175、Zhao arXiv:2211.06605、Gasteiger arXiv:1911.05485 / 1810.05997、Curry DOI 10.2139/ssrn.5959898 + GitHub JDCurry/Heat-Kernel-Attention。
- 待写原型路径：`lab/attn_diffusion/a2_verify.py`。

---

## 2026-07-16：VPSC 理论与原型完成（三定理 + 深网络 + MNIST）

### 背景

提出 VPSC（Variational Predictive Spiking Coding）SNN 训练方法：自由能原理（目标）+ 均场退火（无替代梯度前向）+ 势博弈（多层信用分配）。理论优先、精度次要。

### 三定理状态

- **定理 1（STDP = 自由能零温极限）**：定稿于 `docs/theorem1.md`，吸收三处修订（平方能量阈值、双 τ 窗、误差塌缩归因重写为 LIF 动力学）。`deep_stdp.py` 验证窗形态 R²=0.82、τ≈4；符号 anti-Hebbian（开放问题）。
- **定理 2（势博弈收敛 → F 单调）**：纯生成目标下 F 单调非增。`toy_verify.py` P1、`shd_train.py`、`deep_critical.py` D1 均 PASS。
- **定理 3（临界 β_c=1/ρ(W) 处磁化率发散）**：`toy_verify.py` P2（孤立递归层）、`deep_critical.py` D2（深网络，β*=0.80 vs β_c=0.81）均 PASS。

### 深网络扩展

`vpsc/recurrent.py`：递归均场层（W_rec 反馈 + 谱投影）。诚实边界：纯 F 需硬谱投影防 ρ→∞（Ising 交互无下界）。

### MNIST 实验与基准

- 三模型对比（8 epoch）：CNN 98.99% / MLP 97.95% / VPSC 96.95%。VPSC 在静态任务落后 ~2pp，符合预期。
- 持续注视实验（`fixation.py`）：精度随注视时长 59%→96.4%（poisson），证实 LIF 积分收益；static vs poisson 峰值几乎相同（噪声被积分平均掉）。
- 基准（`benchmark.py`）：参数 CNN<MLP<VPSC；训练 MLP<VPSC<CNN；推理 MLP<CNN<VPSC。VPSC 训练比 CNN 快一倍（无卷积+无BPTT）。

### 结论

理论站住，精度符合"理论优先"定位。MNIST 非主场，收益在时序积分。

### 可复现信息

- 仓库路径：`/Users/united_pooh/PyProjects/vpsc`
- 全实验 CPU 可跑：`python experiments/{toy_verify,deep_critical,deep_stdp}.py`、`python lab/mnist/{run_all,fixation,benchmark}.py`。
