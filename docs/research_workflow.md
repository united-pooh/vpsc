# VPSC 分支与研究晋级规范

本仓库将“稳定工程基线”和“研究验证过程”分离管理。目标不是删除负面结果，
而是避免未验证机制直接进入 `main`，同时保留可复现的研究证据。

## 分支职责

### `main`：稳定分支

`main` 只接收已经验证有用、且不会破坏当前稳定基线的原子更改，包括：

- 已通过对应单元测试、数值等价测试和任务门的模型机制；
- 已确认的正确性、数值稳定性或后端修复；
- 可复现的基线、正式结果摘要和必要文档；
- 不改变研究结论的维护性修改。

以下内容不得直接在 `main` 开发或提交：

- 尚未完成预注册和机制验证的新架构；
- smoke-only、部分训练、单次临时调参或未审计结果；
- 为探索方便加入的旁路、silent fallback 或放宽判据；
- 没有对应测试、artifact 和结论边界的性能声明。

### `codex/research-<topic>`：研究分支

每个独立研究问题使用单独分支，例如：

```text
codex/research-fe2h-routing
codex/research-e3-local-learning
codex/research-sparse-kernel
```

研究分支可以包含预注册、实验驱动、诊断代码、失败尝试和负面结果。不同研究问题
不要长期堆积在同一分支；方向变化时从最新 `main` 新建分支。

## 标准研究流程

1. 从最新 `main` 创建 `codex/research-<topic>`。
2. 在 `dev/LOG.md` 先记录问题、假设、冻结配置、成功门和反证条件。
3. 在研究分支实现机制、测试、实验驱动和诊断。
4. 保存原始 artifact、运行命令、环境、随机种子和结果边界。
5. 根据预注册门给出 `PASS`、`NEGATIVE`、`BLOCKED` 或 `INCONCLUSIVE` 判定。
6. 只有 `PASS` 的最小必要代码可以整理成原子提交，晋级到 `main`。
7. 负面结果保留在研究分支和研究日志中，不为进入 `main` 而事后修改判据。

## 晋级 `main` 的最低门槛

准备把研究技术合入 `main` 时，至少检查：

- **机制门**：数学语义、serial/reference 等价或明确的误差容限通过；
- **正确性门**：相关单元测试、回归测试和有限值检查通过；
- **任务门**：在预注册数据、预算和指标上达到冻结标准；
- **非退化门**：负载、路由、状态、梯度或输出没有依赖未声明的塌缩；
- **系统门**：若声明加速或省显存，必须测量真实执行路径，不能用理论 active
  fraction 或 dense-mask 结果代替；
- **证据门**：`dev/LOG.md`、结果 JSON、命令、环境和 claim boundary 完整；
- **维护门**：提交范围最小，`git diff --check` 和目标测试通过，不夹带大型缓存、
  checkpoint 或无关实验产物。

某项与改动无关时可以标记 `N/A`，但必须写明原因。未通过的研究不能以“可导入”、
“smoke 通过”或“局部指标更好”代替正式晋级。

## 晋级方式

研究通过后，不直接把整个研究分支合入 `main`。优先采用以下方式：

1. 在研究分支把可晋级内容整理为一个或少量原子提交；
2. 排除临时脚本、失败配置、大型运行产物和与结论无关的修改；
3. 将原子提交 rebase 或 cherry-pick 到最新 `main`；
4. 在 `main` 上重新执行目标测试和最小正式验证；
5. 在 `dev/LOG.md` 记录晋级 commit、验证命令和最终边界。

## 当前架构基线

- `vpsc/network.py` 与 `vpsc/recurrent.py`：VPSC 理论与 mean-field 验证基线；
- `vpsc/world_model/cores.py::E3GatedTraceScanCore`：稳定时序工程基座；
- `vpsc/world_model/scaling_variants.py::TemporalMoEGatedTraceCore`：已验证的 d4
  dense temporal-MoE 基线；
- FE-1、FE-2H 及后续稀疏路由、负载均衡和局部学习机制：在通过各自预注册门前，
  均属于研究分支内容。
