# VPSC 研究与实验日志

本日志按 NoA 规范维护：证据层（命令/运行/产物）+ 决策层（动机/假设/证据/决定）。倒序排列。

---

## 2026-07-19：E3-SG19 结果 — objective plan tape + visited-edge spikes（数学/速度PASS，状态/二步FAIL）

### 基准语义修复与证据链

- 复核SG17二步树发现：`inventory`/`examine coin`后的文本不含room，但TextWorld物理房间并未改变；旧构造却把下一次`look`标成`room_previous`。采集器现持久化current room，并为既有canonical tree提供只改该语义的显式迁移；共修正`48/616` pairs，source tree SHA-256=`EAF5F15797C1E0BB4C28E1FDA55828C11801B1D15D7FCBF7409E78C2A1416217`，repaired tree SHA-256=`1A8C6C1C61BDCCF06D66F75A547C4FA55BAEBC8A8A4B9F5E5391AB1F352BB373`。
- 因此SG18日志中“80个错误全是模型的previous-room错误”不能继续作为因果结论；原始artifact保留，SG19 artifact内记录完整repair rule、pair ids与新旧hash，不静默覆盖历史证据。
- 回归命令：`.venv-wsl/bin/python -m pytest -q tests/test_e3_sg17_two_step_rollout.py tests/test_e3_sg19_plan_edge_spikes.py`，结果`9 passed`。正式命令：`.venv-wsl/bin/python experiments/e3_sg19_plan_edge_spikes.py --output results/e3_scan/e3_sg19_plan_edge_spikes.json`。
- 推送前全回归：`.venv-wsl/bin/python -m pytest -q`，结果`196 passed, 74 subtests passed in 53.30s`。
- canonical artifact=`results/e3_scan/e3_sg19_plan_edge_spikes.json`，SHA-256=`EA2C855CDC9ECA0D41B8345B3CF3918F4BD2BADCCCA0B35F17DFE1F104505C22`。

### 正式结果

- 公开objective compiler审计仍为48/48等于official walkthrough（walkthrough只作审计、不进入模型）。640 train records压成443 unique prototypes，compression ratio=`.6921875`，目标歧义key=`0`；weighted/expanded train score最大差=`2.22e-15`且prediction等价，部署闭式训练=`.1017 s`，数学与训练门PASS。
- fresh fifth test的一阶11维delta exact=`1.0`；next-affordance bit=`.98594`，但exact mask=`.93125 < .95`，所以one-step state门FAIL。计划tape改善了方向状态，却未完全生成新房间的出口mask。
- 修复错误标签后，teacher/self二步exact均=`.94643`、drop=`0`、routing=`1`、premature=`0`；reward/done=`1.0`、exit=`.99675`、room=`.94805`。仍有33个真实`room_previous`错误，其中31个是`return_edge=1`的立即回边，另2个需要更长visited-edge记忆。
- SNN two-score p50/p95=`.2453/.3493 ms`，相对SG17三seed的LSTM/Transformer六组均PASS；logical storage=`40,756 bytes`，storage PASS。速度、数学与一阶delta成功不能覆盖状态和二步质量失败。

**决定：SG19 overall FAIL。** `(1+δreturn)`只提供相似度加成，未隔离31个回边prototype；下一轮预注册SG20 strict return-edge isolation（`δreturn`分块核）验证这一可证伪机制，再用visited-edge set处理残余2例。next-mask仍保持`.95`硬门，回边机制收口后再测试确定性状态不变量或objective token reservoir，不降低门槛、不宣称已替代ANN。

---

## 2026-07-19：E3-SG19 预注册 — objective plan tape + visited-edge spikes（进行中）

### SG18已经把状态误差压缩到两个可命名变量

SG18 exhaustive affordance state令fresh root delta四通道全部1.0，二步reward/done/exit也全部1.0；剩余80/616错误全是`room_previous`被判novel，来自rollback时丢弃“刚走过的物理边”。next-mask 26/160 exact错误则全部只涉及四个方向bits，说明novel room的下一出口方向不在current mask中。TextWorld公开objective包含完整自然语言路线；对48 train/valid/test games做冻结审计，按文本顺序抽取`north|south|east|west`并追加`take coin`，48/48与stored walkthrough一致。因此最小新增状态是**公开语言计划spike tape + last physical edge**，不是高维ANN hidden。

### 纯spike状态与PSD核

- objective compiler只读episode/environment公开`objective`字符串，用固定word-boundary方向词抽取生成5-slot action spikes；runner把parsed plan与walkthrough比较仅作数据审计，模型输入和runtime绝不读取walkthrough。新games只要模板审计失败就fail-closed，不手调parser。
- active path仍由predicted/actual room relation push/pop/hold；另保留`last_move`，任何move candidate执行后更新为该方向，meta action保持。关系spike `return_edge=1[candidate == inverse(last_move)]`，表达“沿刚离开的边返回已访问room”，rollback不再擦除。
- phase `p=len(active_path)`索引plan tape的`plan[p]`与`plan[p+1]`（越界专用PAD）。key扩展为`K3 suffix + current affordance mask + plan_current + plan_next + return_edge + candidate`。
- primary kernel预注册为SG18 `k_phase_suffix×(1+mask-dot/8)`再乘 `(1+δreturn)` 与 `(1+δplan_current+δplan_next)`；各项均为categorical/linear PSD kernel的非负和与乘积。target/`λ=1e-6`/355-style weighted unique sufficient-statistics流程不变，不从fifth test选权。
- train仍用SG18 exhaustive cache SHA-256=`2E6F7462C2620163CC1F89C3F52D7EE6851C2EAE4C52BADBA5DB08287561A9AB`的640 records；只重新tensorize新增spikes，不再调用环境。expanded-vs-weighted score<=`1e-6`且prediction一致仍为硬门。

### 冻结门与公平边界

- **LANGUAGE/STATE AUDIT**：三split所有objective parse恰等于5-action official plan；runtime plan source标记objective-only；inverse映射只允许四个compass moves。新增unique key内部目标歧义必须报告。
- **ONE STEP**：test delta exact/all channels=1.0；next-mask bit>=.98、exact>=.95（比SG18 `.8375`至少+.10）。
- **TWO STEP**：teacher/self exact>=.98、各channel>=.98、drop<=.01、routing=1、premature=0；80个previous-room错误降至<=5，且next-mask self state由模型输出而非oracle。
- **MATH/SPEED/STORAGE**：weighted equivalence硬门；deployment train仍快于SG17所有ANN；two-score p50/p95不慢于ANN，logical bytes<=最小ANN。
- **What if：**语言目标可以先被编译成稀疏未来事件spike，拓扑记忆则只需一位“是否沿刚才的边返回”；这两个离散变量是否足以让闭式SNN从实时反射跃迁为可组合两步想象，而无需连续hidden和反向传播？
- 若机制PASS，下一步必须构建同plan/mask输入的LSTM/Transformer并用全新games做独立公平确认；若mask PASS但room仍FAIL，升级为visited-edge set/episodic graph spikes；若room PASS但mask FAIL，加入objective原始token reservoir而非walkthrough；若速度FAIL，继续unique dictionary增量Cholesky。

---

## 2026-07-19：E3-SG18 结果 — exhaustive affordance weighted unique KRR（数学PASS / 质量FAIL）

### 真实数据、精确压缩与数值tie

- official exhaustive cache=`results/e3_scan/e3_sg18_exhaustive_affordance_cache.json`，SHA-256=`2E6F7462C2620163CC1F89C3F52D7EE6851C2EAE4C52BADBA5DB08287561A9AB`；train/valid/test=`640/160/160`，三split全部factual replay won、clone不污染live。
- 640 train records压为355 unique keys，compression ratio=`.5546875`，17个key含不同targets。deployment tensorize+aggregate+kernel+weighted Cholesky=`.09552 s`，expanded audit另计`.01891 s`。
- 首次formal在证据落盘前因prediction-equivalence断言fail-closed；改为先写诊断后发现weighted与expanded train score max diff仅`7.77e-9`，小于预注册`1e-6`，但next-mask恰为0的浮点符号翻转导致`>0` prediction不同。原zero-threshold artifact保留，SHA-256=`408B64101F7B639BB493F8723C24DC715405E65FCC79237F871E5704A3598ED5`。
- 使用同一预注册等价容差`1e-6`作为“数值零默认不可用”的mask threshold；不改kernel/data/target/quality门。复用同一cache重跑后score diff不变、完整delta+mask prediction equivalence=true，weighted math PASS。canonical artifact SHA-256=`86D069D6AEAC497ABBCAD64CD6E327DB8A4CB7A85C081EB3A68DC92A1CF08FE3`。

### 质量与因果定位

- exhaustive fifth test delta exact/macro/四channel=`1.0`，相对SG17 all-admissible first `.75`提升`.25`；当前affordance spike与全候选覆盖已完全修复一阶world delta。
- next-affordance bit accuracy=`.975`，exact-mask=`.8375`，未过`.90`门。26个mask错误全部在四个direction bits，按phase分布`6/6/8/5/1`；meta/object bits无错，指向novel room的未来路线信息缺失。
- teacher/self second exact均=`.87013`、drop=0、routing=1、premature=0；reward/done/exit=1，room=`.87013`。80个错误目标全部为room_previous：第一步backtrack后第二步沿反向边回到刚访问room，但active path rollback已删除该edge。预测mask错误没有进一步拉低本轮delta，故两个缺失变量可独立处理。
- two-score p50=`.1983 ms`、p95=`.2856 ms`，相对SG17 LSTM/Transformer六组全部PASS；training/storage PASS。速度与一阶成功不能覆盖next-mask/two-step硬门失败。

**决定：SG18 overall FAIL，但数学压缩与一阶状态机制成立。** 不采用自动route的泛化“raw reservoir”作为第一反应；按上方SG19先加入证据直接指出的objective plan与visited-edge spikes，继续闭式/unique训练。SG18仍不与ANN做输入不匹配的胜利声明。

---

## 2026-07-19：E3-SG18 预注册 — exhaustive affordance spikes + weighted unique KRR（进行中）

### SG17缺的是可生成的当前世界状态，不是更长action memory

SG17 strict SNN的first routing=1、self=teacher、reward/done=1，但room/exit只有约`.73/.74`；同时first/second train-key coverage仅`.619/.604`。因此本轮不加K4、不回BPTT，而把真实观测中的`admissible_commands`编成8-bit sparse affordance state，并把训练从“factual+2 copies”扩为每个expert root的**全部**合法candidate。模型同时预测原11-logit delta与next-affordance 8 bits，使第二步self rollout可以消费自己生成的世界状态。

### 数学加速：重复prototype压成带权唯一spike dictionary

- exhaustive root collection仍只用official `Environment.copy()`，预期train/valid/test=`640/160/160`（每level-5 game 5 roots、每局共20 admissible actions）；live factual replay必须全部won且clone不污染state。
- key固定为`phase + K3 action suffix + current 8-bit affordance mask + candidate action`。primary PSD kernel预注册为`k_phase_suffix × (1 + <mask_q,mask_p>/8)`：SG15 strict nested suffix负责事件组合，线性bit inner-product提供当前世界内容相似度；不看valid/test选权。
- target=`11-dim +/-1 multichannel code + 8-dim +/-1 next-affordance code`；terminal next mask严格为0。self rollout由预测room/done路由context，并把阈值化预测mask作为第二步state；teacher control用真实first next mask。
- 对完全相同key的`n_g`个样本只保留`count n_g`与target mean `ȳ_g`。令`W=diag(n_g)`，唯一prototype kernel为`K_u`，解对称系统 `(sqrt(W) K_u sqrt(W)+λI)b=sqrt(W)ȳ`，部署系数`c=sqrt(W)b`。这与展开所有重复样本的kernel ridge函数解等价；runner必须在train scores上验证max diff<=`1e-6`，expanded solve只作审计、不计部署训练wall。
- primary仍`λ=1e-6`、float64 fit/float32 stream；报告640→unique count压缩率、单遍聚合、weighted solve wall、expanded audit wall、模型bytes与two-score p50/p95。

### 对照、冻结门与分支

- SG18是**纯SNN机制实验**：引用SG17 corrected FAIL artifact SHA-256=`E46E5F24C3D57A40A3405D8BCFBF737A5223C151371FC235BC527DC2096CC7EF`，复用同一fifth official tree及其LSTM/Transformer作为action-only task controls；本轮若PASS，下一轮必须给ANN同一affordance输入后再做公平独立比较，不能直接宣称替代。
- **DATA/MATH**：三split exhaustive counts、全won/non-mutating/零OOV；unique weighted与expanded train scores<=`1e-6`、prediction完全一致，压缩率<1。
- **ONE-STEP STATE**：fresh exhaustive test delta exact>=.95、各channel>=.95；next-mask bit accuracy>=.95、exact-mask>=.90。相对SG17 first exact `.75`至少提高`.15`或过`.95`绝对门。
- **TWO-STEP**：teacher second exact>=.95；self second exact>=.90、teacher-self drop<=.05、各self channel>=.90、premature=0，并至少不低于SG17 strict self `.6834`+`.15`。
- **SPEED/STORAGE**：unique deployment training wall小于SG16R每个LSTM/Transformer 50-epoch wall；two-score p50/p95不慢于SG17 ANN controls，logical bytes不大于最小ANN。
- **What if：**世界状态不必由高维连续hidden慢慢学出来；把可行动性直接编码成稀疏观测spikes，再用带权唯一联想原型闭式更新，能否同时消除数据覆盖洞、生成下一latent state，并让两步思考仍保持亚毫秒？
- 若teacher和self均PASS，进入SG18R：同affordance prompt的LSTM/Transformer + 新games独立比较；若teacher PASS/self FAIL，说明随机novel-room future mask不可由当前state决定，加入objective-language plan spike tape/不确定性；若teacher也FAIL，加入raw observation reservoir content；若math/速度FAIL，做增量Cholesky unique-key update或prototype pruning。

---

## 2026-07-19：E3-SG17 结果 — two-step official branch rollout（机制FAIL）

### 先修正terminal target契约，再接受真实否证

- 首次formal完整结束但审计发现SG17直接读取TextWorld在`done=True`后残留的`admissible_commands`，把terminal `take coin`的exit-count标成1；SG10训练契约则把terminal next action set定义为空、exit=0。无效artifact保留为`e3_sg17_two_step_rollout_invalid_terminal_label.json`，SHA-256=`2359A14BED0DF06E538C7BF7B2FFF57C1BE0D25046E7CEB9A3F372A7A80AA698`。
- 只修`done -> after_actions=()`并增加回归测试；模型/kernel/训练/tree/门不变。修正后formal wall=`135.70 s`，canonical artifact=`results/e3_scan/e3_sg17_two_step_rollout.json`，SHA-256=`E46E5F24C3D57A40A3405D8BCFBF737A5223C151371FC235BC527DC2096CC7EF`。

### 真实双层树与质量分解

- 8 official games、40 factual roots、160 first branches、616 nonterminal second pairs；canonical tree SHA-256=`3BEDCF33CF66C6BB26100583C32788746A80978A4844838FE50062DC0227A2C8`。全部live factual replay最终won、两层`Environment.copy()`均不污染live；clone p50约`106.1 ms`且与model latency分列。
- strict phase SNN：all-admissible first exact=`.75`；teacher-forced second=`.6834`；self second=`.6834`，macro=`.8673`，room/reward/done/exit约=`.7273/1/1/.7419`。first routing=1.0、premature stop=0，self与teacher完全相同，说明失败不是递归latent漂移，而是action-only state本身缺少分支内容。
- LSTM teacher/self约`.62-.65/.58-.63`；Transformer约`.67-.71/.56-.66`，best ANN teacher也未到预注册`.85` task gate。当前typed-action输入对全分支任务整体信息不足，不能把SNN单独判作优化问题。
- SNN two-score p50=`.111-.126 ms`、p95=`.169-.195 ms`；相对LSTM p50=`.396-.430 ms`和Transformer=`.641-.664 ms`的6组均PASS。training/storage也PASS，但quality/task门失败，速度不能覆盖否证。

### 覆盖率控制

- 原480-example train只有201个唯一`(phase,last3,candidate)` keys且train内部零歧义；SG17 first/second真实分支key覆盖仅`.6188/.6039`，missing集中`look`与`inventory`，证明`counterfactual_limit=2`造成明显coverage hole。
- 但已覆盖key在fresh first/second上的majority exact也只有`.9394/.8522`；修正terminal标签后仍存在跨world room/exit差异。因此单纯多采候选可能提高但不能保证两步生成，当前affordance/目标计划/观测状态确实不充分。

**决定：SG17 overall FAIL，不生成sixth confirmation，不宣称多步世界模型。** 保留SG16R one-step闭环独立成功；下一阶段先做exhaustive candidate coverage + observation/affordance spike state，并用唯一原型加权闭式解控制训练增长。若teacher能过而self不过，再引入可生成的objective-plan/observation latent；不回到BPTT。

---

## 2026-07-19：E3-SG17 预注册 — two-step official branch rollout composition（进行中）

### 先拆分“动力学想象”与“未来action proposal”，避免一个失败掩盖另一个

SG16R已证明one-step delta足以在真实receding-horizon闭环中超过ANN，但它每步都从真实环境取得当前`admissible_commands`；这还不是模型内两步思考。两步扩展有五条可检验路线：

1. **global action imagination**：第二步在8个全局动作中选，完全无oracle，但会把未知future affordance与transition composition混为一谈；新房间的随机出口方向原则上不可由action-only state精确预知。
2. **official clone-tree candidate proposal**：`Environment.copy()`只向评估器提供第一步真实分支后的合法第二动作集/target；模型仍独立递归预测。这是纯动力学组件测试，**本轮选择**。
3. **learned affordance-mask head**：把8动作可用性加入world-state target，再做model-only two-step MPC；若路线2通过，这是下一步移除oracle的工程路线。
4. **observation reservoir/generative content**：编码或生成room description/exit content后预测未来动作集，信息更完整但会同时改变状态表示，暂不用于最小归因。
5. **uncertainty-aware frontier stop**：known room可rollout，预测novel room时输出不确定性并回到一步replan；更符合随机procedural topology，但需先量化路线2的真实composition误差。

### 冻结数据树、递归规则与对照

- 引用SG16R独立artifact SHA-256=`CFD3E2FF3F3F384EE1D6EAB445D468432DF6AB5A0BC534B86EC63114B250598E`；训练仍是相同480 expert-history examples、strict phase kernel/`λ=1e-6`与50-epoch三seed LSTM/Transformer，不新增label或训练数据。
- 在fifth test 8个官方`.z8`中沿存档factual path定位5个root states；对每个root的全部真实第一候选用`env.copy().step(a1)`，若非terminal，再对该branch的全部真实admissible `a2`用第二次copy执行。live factual env只在枚举结束后沿存档action前进一步；所有target均来自官方core transition。
- **teacher-forced second**：模型预测a1 delta，但用a1的真实room relation选择push/pop/hold后预测a2；衡量transition head在正确latent上的二阶质量。**self-rollout second**：只用模型预测的room/done选择imagined push/pop/hold后预测a2；衡量误差复合。若错误预测terminal，所有实际存在的second branches记为premature-stop错误，不偷偷继续teacher forcing。
- SNN用K3 bool suffix+phase递归；LSTM/Transformer从同一expert prefix cached state分支，novel commit candidate state、previous恢复上一depth、same/no-room保持。第二候选集是明确记录的**evaluator oracle proposal**，不进入质量外推，也不称完整planner。
- 先在已用于SG16R的fifth games做mechanism；若PASS，冻结runner并生成sixth valid=`20270101..08`、test=`20270109..16`独立确认。branch tree本身生成规范、game SHA、root/action counts与canonical fingerprint必须落artifact。

### 冻结门与后续数学路线

- **REAL TREE/DATA**：全部root来自official TextWorld 1.7 core、双层分支均由`Environment.copy()`且不改变live factual trajectory；8 games×5 roots、全部stored factual path最终won；无action OOV。
- **QUALITY**：SNN all-admissible first exact>=.98、teacher-forced second exact>=.98、self-rollout second exact>=.95；self相对teacher下降<=.03，各second channel>=.95，premature-stop=0，且self exact/macro不低于最佳LSTM/Transformer-.02。
- **TRAIN/ROLLOUT RESPONSE**：沿SG16R training/storage门；每个真实two-action pair的model-only score p50/p95在三replication均不慢于LSTM与Transformer。环境clone耗时单列，不混入模型响应。
- **What if：**strict phase spike memory不仅能对当前候选作反射，还能把自己的第一步delta变成下一latent state，在不展开BPTT或attention history的情况下稳定组合第二次转移；这种局部可组合性是否就是实时世界模型“思考”的最小数学单元？
- 若独立PASS，进入SG18 learned sparse affordance head，移除future candidate oracle并做真正2-step MPC；若teacher-forced PASS/self FAIL，修正latent transition/不确定性而非扩大读出；若两者均FAIL，转observation reservoir×strict phase kernel；若只有速度FAIL，向量化unique prototype dictionary。

---

## 2026-07-19：E3-SG16R 结果 — fifth-fresh real closed-loop confirmation（独立PASS）

### 冻结协议与第五批数据

- SG16 mechanism artifact在生成前锁定，SHA-256=`89569254C863DDD8C496911DCB04C40B11F516F9BFD3C831EBA3D897559124FC`；confirmation runner逐项断言planner语义顺序/破平、topological rollback、strict kernel/`λ=1e-6`、K3 state、ANN结构与50-epoch三seed预算完全相同。
- fifth corpus valid=`20261201..08`、test=`20261209..16`，生成/采集wall=`240.69 s`；train manifest/episodes/token-events SHA与SG2/SG15R逐字节一致。48 games全部won、每局5 steps、每step两个copy counterfactual，counts/groups=`480/120/120`,`160/40/40`，零OOV/歧义。
- SG16R运行前的`protocol/train artifacts/vocabulary/action alphabet`四项reproduction均为true；planner仍不请求walkthrough、不调用counterfactual clone，只评分当前真实`admissible_commands`并执行一个动作。

### 独立闭环结果

- formal wall=`55.56 s`，artifact=`results/e3_scan/e3_sg16r_fifth_fresh_closed_loop_confirmation.json`，SHA-256=`CFD3E2FF3F3F384EE1D6EAB445D468432DF6AB5A0BC534B86EC63114B250598E`。
- strict phase SNN三次均`8/8`，合计24/24新游戏；mean actions=5、optimal-five win rate=1、无超预算。LSTM三seed wins=`8/8,8/8,1/8`，mean=`.7083`；Transformer=`7/8,1/8,6/8`，mean=`.5833`。SNN在独立真实闭环quality与path efficiency上均严格不低于最佳ANN。
- offline fifth test：SNN exact=1.0；LSTM=`.90/.97/.91`；Transformer=`1.0/.98/1.0`。Transformer离线接近满分但闭环明显不稳定，证明固定三candidate expert-history accuracy不能替代“全部admissible action + 自己诱导状态”的闭环评价。
- SNN online fit=`.12316 s`，仍严格快于LSTM `.78-1.03 s`和Transformer `1.02-1.21 s`。SNN candidate p50=`.06478-.07413 ms`、decision p50=`.5439-.6521 ms`；相对两ANN的6组candidate/decision p50/p95全部PASS，storage PASS。

**决定：SG16R independent overall PASS。** 当前获得的是level-5真实语言环境中“typed action spike state + observation-corrected topology + one-step delta planner”的工程基底：闭式在线训练、更快实时响应、闭环质量超过同预算LSTM/Transformer。它仍不是通用世界模型：没有生成观测、没有未来affordance分布、没有多模态融合，也尚未证明模型内多步想象。下一步按冻结route进入SG17两步official `Environment.copy()` rollout；成功与否都不能把SG16R边界外推。

---

## 2026-07-19：E3-SG16 预注册 — real TextWorld closed-loop candidate planner（进行中）

### 从120/120状态预测转为真实环境行动，而非继续刷离线accuracy

SG15R在第四批未见games上120/120，但它仍只回答“给定expert history与candidate，下一状态是什么”。世界模型技术基底必须把预测用于行动，并承受自身选择改变后续状态分布。SG16因此打开官方`.z8`解释器，让模型在每一步读取真实`admissible_commands`、为全部候选预测四通道delta、选择并执行一个动作，再由真实观测更新状态，直到won或预算耗尽；不向planner暴露walkthrough、counterfactual clone或离线factual标签。

### 冻结planner、状态更新与公平对照

- primary仍是SG15 `strict_phase_suffix = kp·(k0+k1+k2+k3)`、K3 bool spike delay、unit-root progress phase、480 train prototypes与`λ=1e-6`；SG15R artifact SHA-256=`A0599E48C13E3FFC1171DD5FCF08B175F4110FE884CD33EF0ED6B916A1698ACE`必须校验。
- planner只使用模型输出，固定语义优先级为`reward_positive > done > room_novel > room_same > room_previous > no_observation`；完全同类时用各channel margin，再以action字典序稳定破平，不使用action名称特判或真实下一状态挑candidate。
- 执行动作后才允许用真实room observation维护共同的topological stack：novel room push，返回已见room则rollback到对应depth，same/no-room不改变progress。SNN截断/恢复spike suffix；LSTM/Transformer恢复各自同depth cached state。该规则对三者完全相同，且不读取reward/done之外的oracle plan。
- 对照为同一480 train examples、同一D32/state31/共享Bilinear 11-logit任务头、50 epochs/500 updates/24k exposures、同一schedule的LSTM与Transformer；默认3个训练seed。SNN使用一次10×48 block Cholesky–Schur在线拟合，无BPTT。各模型分别从reset后的同一真实test game运行，最大15 actions。
- 先在现已看过离线结果的fourth test `20261109..16`做mechanism run；即使PASS也不算独立。runner和planner冻结后，生成fifth valid=`20261201..08`、test=`20261209..16`做SG16R；train仍逐字节复用`20260801..32`。

### 冻结门与分支

- **CLOSED LOOP QUALITY**：SNN win rate=1.0、mean actions<=5.0、无超预算；win rate与path efficiency均不低于LSTM/Transformer最佳值。ANN不因质量差而从速度对照中删除。
- **MODEL VALIDITY**：三模型先报告同一离线test四通道指标；实时所有candidate action必须在train vocabulary/alphabet内，真实game SHA与manifest一致，全部run来自官方TextWorld 1.7 core API。
- **TRAIN/RESPONSE**：SNN在线fit wall严格小于LSTM与Transformer各自训练wall；三次/三seed比较中，SNN每candidate p50/p95与整步decision p50/p95均不慢于两种ANN，模型逻辑bytes也不大于最小ANN参数bytes。
- **What if：**可恢复的spike suffix不是一个被动分类器，而是一张由真实观测校正的微型认知地图；闭式联想delta能否像model-predictive reflex一样，在每个真实环境step选择通向novel state或terminal reward的动作，以更低训练/响应成本达到Transformer闭环质量？
- 若SG16R独立PASS，进入SG17多步counterfactual rollout（至少2-step beam、累计reward、模型想象与真实分支一致性）；若状态预测正确但闭环失败，优先修planner horizon而不改SNN核；若某candidate OOV/观测状态不足，进入observation reservoir×strict phase kernel，不降低闭环门。

### 第四批闭环机制结果（非独立）

- 首次2-game smoke已完成真实run，但SNN initialization恰为`0 ms`，复用的LM throughput汇总器计算`tokens/p50`时除零，结果在写artifact前fail-closed。只把SG16本地计时汇总改为普通count/mean/p50/p95/p99，不改模型、planner、游戏、门或任何预测；同配置重跑smoke后SNN `2/2`且均5步。该实现负结果保留，防止把汇总故障误写成模型失败。
- formal wall=`52.79 s`，artifact=`results/e3_scan/e3_sg16_closed_loop_planner.json`，SHA-256=`89569254C863DDD8C496911DCB04C40B11F516F9BFD3C831EBA3D897559124FC`。三次SNN均`8/8`、mean actions=5、optimal-five win rate=1；总计24/24真实`.z8`闭环均走最短winning path。
- LSTM三seed closed-loop wins=`7/8,7/8,1/8`，mean win=`.625`、mean actions=`8.75`；Transformer离线test三seed均exact=1.0，但闭环wins=`5/8,1/8,8/8`，mean win=`.5833`、mean actions=`9.1667`。差距来自runner必须评分**全部**admissible actions与自身状态分布，而离线test每step只有factual+两个固定counterfactual；这正是闭环门要揭示的泛化缺口，不据此修改planner。
- SNN在线deployment fit=`.08130 s`，三seed LSTM train=`.70-.86 s`、Transformer=`1.01-1.11 s`。SNN candidate p50=`.06225-.06694 ms`、decision p50=`.5215-.5307 ms`；对LSTM/Transformer的6组candidate p50/p95与decision p50/p95逐项全PASS，storage也PASS。

**决定：SG16 mechanism overall PASS，且在这批真实闭环任务上质量、训练、响应形成Pareto优势。** 但fourth worlds已用于SG15R离线观察，不能称独立取代ANN；planner/artifact现冻结，按预注册生成第五批`202612xx`做SG16R一次性确认。

---

## 2026-07-19：E3-SG15 结果 — strict phase-isolated spike associative memory（独立PASS）

### SG14R只差3例，但差错精确来自跨phase负迁移

SG14R primary在第三test为`.975`，3个错误均是step3 factual `go south`的exit1→2；冻结门不能放宽。预注册control `phase_product_only=kp·Σki`在同一train CV、third valid、third test均为1.0，而`base+product`失败。这说明跨phase普通suffix项`Σki`把step2的exit2证据泄漏进step3；不需要先引入reservoir，最小修复是令不同phase的联想能量严格为0。

### 机制固化与第四批独立确认

- SG15 primary固定为`strict_phase_suffix = kp·(k0+k1+k2+k3)`，不含additive phase、不含cross-phase base；它等价于按unit-root event phase分块的四级suffix spike memory。kernel仍PSD，因为是两个PSD delta kernel的乘积。
- lambda仍只由32 train games的4-fold CV选择，预计/冻结候选grid不变；在线block Cholesky–Schur、K3 bool delay、phase state、480 prototypes、float32 stream全不变。
- 先在已看第三数据上运行mechanism artifact，确认它与预注册control结果一致；此结果不算独立。随后在生成前冻结artifact，并生成fourth valid=`20261101..20261108`、test=`20261109..20261116`，train仍为`20260801..32`，counts/groups仍`480/120/120`,`160/40/40`。
- **QUALITY**：mechanism/fourth test均沿`.98`门；fourth独立最多2/120错。**MECHANISM**：third step3 factual错误从3降至<=1，且strict exact至少比base+product `.975`高`.01`或过绝对门。
- **TRAIN/ONLINE/STREAM**：完整CV、online score<=1e-6、三个计时replication与storage全部沿fresh Transformer门，不因删去cross-phase项而放松。
- **What if：**实时SNN世界状态应像分区的海马情境记忆：只有“同一世界阶段”的spike pattern可以产生联想，跨阶段相似动作必须完全正交；严格phase gating能否在第四批新拓扑上稳定复现Transformer的位置条件推理？
- 若第四批独立PASS，进入closed-loop candidate planner；若FAIL，才说明action+phase不足，转观测内容reservoir×strict phase kernel。

### 机制与第四批独立结果

- mechanism artifact在已看第三批数据复现预注册control：test exact/macro/all channels=`1.0`，train-game CV exact/macro=`.9979167/.9994792`；cached p50=`.082-.096 ms`、p95=`.139-.175 ms`。artifact SHA-256=`3E9F05EF88703A01D334D91A2C39A36127FD8E18E73B179699449F3ECD270F73`，只算机制PASS。
- fourth corpus生成wall=`222.89 s`；train manifest/episodes/token-events SHA与SG2/SG13R/SG14R逐字节一致。valid/test=`20261101..08/20261109..16`各8 games，全部won；counts/groups=`480/120/120`,`160/40/40`，每step两个真实counterfactual，零OOV，审计PASS。
- 冻结SG15R formal wall=`21.39 s`，artifact=`results/e3_scan/e3_sg15r_fourth_fresh_confirmation.json`，SHA-256=`A0599E48C13E3FFC1171DD5FCF08B175F4110FE884CD33EF0ED6B916A1698ACE`。fourth test严格`120/120`，exact/macro/四channel/step consistency全为1.0，无错误记录；train CV数值与mechanism一致。
- online/training/stream全PASS：selection+fit=`.14456 s`；三replication SNN p50=`.08147-.08307 ms`、p95=`.13849-.14269 ms`，均快于Transformer p50=`.28998-.29639 ms`、p95=`.41114-.52308 ms`。

**决定：SG15获得第四批独立PASS，strict phase isolation是目前首个同时跨新world复现ANN级状态质量、闭式在线训练和更快cached response的纯spike组件。** 证据仍限于level-5 typed action event delta，不宣称已是通用/多模态世界模型；按上方SG16接真实闭环行动。

---

## 2026-07-19：E3-SG14R 结果 — third-seed frozen phase-bound confirmation（独立FAIL）

### 第二个机制集满分后，仍用第三批worlds做一次真正冻结确认

SG14在已用于错误定位的fresh test达到120/120，但该数据已经影响`phase×suffix`假设，不能作为独立证据。SG14R在生成任何新episode前冻结kernel、lambda、状态、训练、门与参考artifact，再使用从未出现的procedural seeds。

- train继续字节级复用`20260801..20260832`；third valid=`20261001..20261008`、third test=`20261009..20261016`。输出`results/e3_scan/textworld_sg14r_l5`，不覆盖前两批语料；预期counts/groups=`480/120/120`与`160/40/40`。
- primary严格冻结为`base_plus_phase_product: Σki + kp·Σki`、K3 bool delay、unit-root phase、480 prototypes、`λ=1e-6`、block Cholesky–Schur、float32 deploy。原SG14 artifact SHA-256=`15D345DE44B73BBA4E39BD2D3199E616AE88F70CCEF9B4D55757CD183B460B2B`必须校验。
- 仍可重算train 4-fold CV以证明train路径一致，但third valid/test不用于kernel/lambda/门选择。runner必须断言train SHA等于前序、三split seeds严格隔离、全部won、零OOV。
- **CONFIRM QUALITY**：third test exact/macro>=.98、各channel>=.95、rare positive>=.90，且相对fresh Transformer-.02；120例最多2错。**TRAIN/ONLINE/STREAM** 原样，三个replication逐一过门。
- **What if：**phase×suffix乘法不是对第二批8个test worlds的事后修补，而是稳定表达“当前阶段下的事件组合”的可迁移SNN归纳偏置；在第三批全新拓扑上仍能否保持ANN质量、无反传训练和更快实时响应？
- 若独立PASS，锁定为事件世界模型技术基底的首个跨seed确认组件，并进入真实candidate planner闭环；若FAIL，撤回跨world泛化，只保留机制/速度证据，转观测内容reservoir融合。

### 独立结果

- third corpus生成wall=`222.93 s`；train三核心artifact SHA与原train相同，valid/test各8新games、全部won、零OOV、counts/groups/seed隔离PASS。third summary SHA train/valid/test=`2b24019c.../8b736c13.../82ee79ee...`。
- frozen SG14 primary/`λ=1e-6`/train-CV复现通过；formal wall=`20.76 s`，artifact SHA-256=`B765964EAF8845A65B71F98304A560EC935BE898188856E97EC9AB0C6A48013B`。
- third test exact/macro=`.975/.99375`，room/reward/done=1.0、exit=`.975`；3/120错误全部为step3 factual `go south`的`exit1→exit2`。冻结`.98`门与“step2/3错误<=2”门均FAIL，只差1例也不能宣布成功。
- training/online继续PASS；cached p50=`.091-.092 ms`、p95=`.140-.153 ms`继续明显快于Transformer，但随quality门FAIL。
- 预注册controls：old additive `.9333`；base+product/depth-product `.975`；**phase_product_only third valid/test均1.0，train CV也1.0**。这把失败定位为cross-phase base项负迁移，而非suffix深度或状态内容不足；看过third test后只能形成SG15假设，不能把control事后改称primary成功。

**决定：SG14R独立FAIL，撤回base+product跨world确认。** 按上方SG15严格移除cross-phase联想，并要求第四批全新games确认；reservoir内容融合延后到strict phase也失败时。

---

## 2026-07-19：E3-SG14 结果 — phase-bound hierarchical spike kernel（机制PASS / 待第三批确认）

### fresh Transformer近满分，SG13R缺的是phase×history绑定

SG10R确认fresh Transformer exact=`.9972`，所以任务在新worlds上仍可学；suffix kernel的`.9333`不是ANN ceiling下降。逐例审计SG13R 8个错误全部是**factual candidate**的exit count，6个在step3把真实exit1判为2，2个在step2把真实exit2判为1；step4没有错误。K3在这些step已包含完整历史，单纯加K4无因果依据。

SG13 kernel写成`k_suffix + k_phase`，phase只提供独立加性相似度，无法表达“同一action suffix在step2与step3应有不同后果”。最小数学修复是加入Schur/product kernel `k_phase×k_suffix`；两个PSD kernel的Hadamard/product仍PSD，相当于只有phase相同的prototype才提供对应阶数的suffix联想。

### 冻结核族与primary

- 保留四级nested suffix matches `k0..k3`与phase equality `kp`。新增非负product项 `kp·ki`，runner显式记录base/additive/product三组权重。
- **primary运行前固定**为 `base_plus_phase_product = Σki + kp·Σki`：跨phase仍有普通backoff，同phase匹配权重加倍；不使用fresh test选择。controls=`old_additive_phase`、`phase_product_only`、`candidate_phase_product`、`depth_weighted_product`均预注册只解释机制。
- data仍为SG13R fresh corpus，train game-CV只选lambda；fresh test现已被SG13R/SG10R查看，因此SG14即使PASS也只是机制修复，后续必须再生成`202610xx`新games做SG14R。
- online仍用block Cholesky–Schur，stream仍用24-bit K3 delay+phase与480 prototypes；只增加逐prototype的phase-gated suffix conjunction，不引入ANN/reservoir或可学习循环参数。

### 冻结门

- 引用fresh Transformer artifact SHA-256=`1E5E4A49E2B0000D91D2E2EB71CFECEE270FC4913F8E35E93C213C08DD8927A6`和SG13R SHA=`A2EDBE97273FA1AEDCE8A34393C618B48CF613A41D6C55BA234A3D22061A8F96`，运行时必须校验。
- **QUALITY**：primary fresh test exact>=.98、macro>=.98、各channel>=.95、rare positive>=.90，且在exact/macro不低于fresh Transformer-.02。**MECHANISM**：step2/3 exit错误总数<=2，且exact至少比old additive `.9333`高`.04`或过绝对门。
- **TRAIN/ONLINE**：完整4-fold train-CV+fit<=fresh Transformer mean wall `.9758 s`；online score差<=1e-6/prediction一致，每次full pass<=对应Transformer。
- **STREAM**：质量过门后，三个replication p50/p95<=fresh Transformer，model bytes<=Transformer，persistent/full差<=1e-6。
- **What if：**世界模型的关键不是再多存一个事件，而是把“事件组合”与“当前世界阶段”做乘法绑定；一个严格PSD、可闭式与在线求解的phase-gated spike kernel能否复现Transformer位置×attention交互，同时维持SNN稀疏状态和实时优势？
- 若机制PASS，生成完全新的SG14R seeds确认后再接closed-loop；若仍FAIL，说明action/phase不足，进入`frozen gated SNN reservoir content × exact delay/phase kernel`融合，并要求新观测内容通道。

### 正式结果

- formal wall=`23.05 s`，artifact=`results/e3_scan/e3_sg14_phase_bound_kernel.json`，SHA-256=`15D345DE44B73BBA4E39BD2D3199E616AE88F70CCEF9B4D55757CD183B460B2B`；primary CV选择`λ=1e-6`。
- train-game 4-fold OOF exact/macro=`.99375/.99844`；fresh test exact/macro/channel/step全部`1.0`，SG13R的8个step2/3 factual exit错误降为0。old additive control保持`.9333`，定位出的乘法绑定变量产生`.0667`提升。
- 完整CV+fit wall=`48.16 ms` vs fresh Transformer=`975.82 ms`，约20.3x；online三order prediction/score等价过门。
- cached p50=`.09697/.09832/.10039 ms`、p95=`.16079/.16696/.16571 ms`，均快于Transformer`.28998-.29639/.41114-.52308 ms`；model/state bytes门继续通过。

**决定：所有机制门PASS。** 但第二批fresh test已用于SG13R错误审计，结果不能冒充独立确认。按上方SG14R生成第三批games；在此之前不接closed-loop、不宣称已跨world稳定替代Transformer。

---

## 2026-07-19：E3-SG10R 结果 — fresh-game ANN/SNN task-control rerun（Transformer task PASS）

### SG13R失败后先验证任务门，不立刻改SNN

SG13R fresh test exact仅`.9333`，但当前相对参考仍是原4-game test上的Transformer=1.0；如果fresh procedural worlds连冻结ANN协议也明显下降，就不能把全部差距归因于suffix kernel。该控制不提出新模型，只在同一fresh corpus重跑SG10五模型，区分“task distribution shift”与“SNN associative state不足”。

- corpus、train/fresh valid/fresh test、counts=`480/120/120`、groups=`160/40/40`与SG13R完全相同；训练集字节级等于原SG10 train，vocab fingerprint不变。
- 三种SNN/LSTM/Transformer仍D32/state31、identical Bilinear(D,D,11)、50 epochs、500 updates/model、24k exposures、inverse-frequency channel CE、seeds `{0,1,2}`、CPU4 threads；不因SG13R结果改seed/epoch/门。
- frozen SNN ridge与cached timing原样运行；主要目的不是让旧SNN过门，而是得到fresh ANN task ceiling与逐channel错误。formal output固定`results/e3_scan/e3_sg10r_fresh_game_baselines.json`。
- **TASK** 沿SG10：best ANN exact>=.90、每channel>=.95、rare reward/done recall>=.90。**相对判别**：若Transformer仍接近1.0而suffix kernel=.9333，SG14必须加入世界内容状态；若Transformer也明显低于原结果，仍不降低SNN绝对`.98`最终门，但后续需扩大train或采用更强跨game状态表征，不能宣称ANN ceiling已被追平。
- **What if：**fresh确认失败不是suffix联想核单独的问题，而是原32-game train对新world topology覆盖不足；同一Transformer全局attention能否仍恢复近满分，从而证明差距确实来自SNN状态归纳偏置？

### 正式结果

- 首次启动在模型构建前因CLI仍期待旧heldout seeds而fail-closed；显式传入预注册fresh seed列表后重跑，无部分训练产物。formal wall=`148.71 s`，artifact SHA-256=`1E5E4A49E2B0000D91D2E2EB71CFECEE270FC4913F8E35E93C213C08DD8927A6`。

| model/path | fresh exact | macro | room/reward/done/exit |
|---|---:|---:|---:|
| SNN-BPTT | .9111 | .9750 | .9833/1/1/.9167 |
| SNN-AT1 | .9111 | .9750 | .9833/1/1/.9167 |
| SNN-RA0 | .9111 | .9750 | .9833/1/1/.9167 |
| LSTM | .9056 | .9757 | .9972/1/1/.9056 |
| **Transformer** | **.9972** | **.9993** | **.9972/1/1/1** |
| frozen SNN ridge | .9417 | .9826 | .9722/1/1/.9583 |

- **TASK PASS**：Transformer三seed合计仅1/360 example错误，fresh task ceiling没有下降；LSTM仍主要错exit，说明recurrent衰减状态共同受限，而attention+absolute position解决了组合。
- SNN与ridge继续FAIL，不因扩大test而改变；RA0 cached约`.147-.168/.221-.251 ms`仍快于Transformer`.290-.296/.411-.523 ms`，但质量不过门。
- 这确认SG13R `.9333`的8个exit错误属于phase-conditioned action-history表示不足，而非fresh worlds不可学。逐例错误全部是factual candidate、集中step2/3，支持上方phase×suffix乘法核而不支持盲目K4。

**决定：fresh TASK有效，SG13R独立否证成立。** 不扩大epoch、不降质量门；按上方SG14只加入PSD phase-bound suffix interaction，保留闭式/online与纯稀疏spike state。

---

## 2026-07-19：E3-SG13R 结果 — fresh procedural games independent confirmation（独立否证）

### 为什么原test PASS后仍不能直接进闭环

SG13 primary在原test达到`.9833`，但该test的K3覆盖率和各variant结果已在SG11/SG12设计阶段查看；即使SG13没有用它选择kernel/lambda，架构仍受同一数据的研究反馈影响。独立确认必须在冻结所有数学选择后生成新games，不能用新test继续调权重、kernel或阈值。

### 冻结确认协议

- train仍为`20260801..20260832`，确保训练信息与SG13完全相同；fresh valid=`20260901..20260908`、fresh test=`20260909..20260916`，与SG2/SG10原valid/test `20260833..20260840`及train全部按game seed隔离。
- 官方TextWorld 1.7.0 `tw-coin_collector --level 5`、每step最多2个`Environment.copy()` counterfactual；新输出目录`results/e3_scan/textworld_sg13r_l5`，不覆盖旧语料。预期counts train/valid/test=`480/120/120`、step groups=`160/40/40`、每game恰5 steps。
- primary严格冻结为`suffix3_phase: k0+k1+k2+k3+kp`、K3 bool delay、phase、train prototypes480、target code、`λ=1e-6`、float64 train/float32 deploy；不从fresh valid/test选kernel/lambda，不改`.98`质量门。
- 生成后先验证manifest/game/episode/event SHA、全部won/return、action alphabet无OOV、fresh seeds不相交，再一次性运行。原SG13 artifact SHA-256=`1DF3593277FED31B2624DAC27AD486E368203B3D9C76079A36BA91F5FFEC8C6E`必须作为冻结架构证据。
- **CONFIRM QUALITY**：fresh test exact>=.98（120例即最多2错）、macro>=.98、各channel>=.95、rare positive recall>=.90；fresh valid只报告、不选择。train CV与原SG13结果必须数值复现。
- **TRAIN/ONLINE/STREAM**：完整train-CV+fit、block Cholesky–Schur等价、三个fresh-test计时replication继续沿SG13门；模型bytes<=Transformer，persistent/full scores差<=1e-6。
- **What if：**分层spike suffix kernel的优势来自可迁移的事件组合归纳偏置，而不是刚好贴合原40个procedural games；冻结后面对16个全新世界，它能否仍达到Transformer级多通道质量并保留无反传训练与实时响应？
- 若独立PASS，下一步把kernel score接真实TextWorld candidate planner，比较闭环选择成功率/累计reward/rollout horizon，并同时保留Transformer对照；若FAIL，撤回泛化胜利，仅保留原任务机制结果，进入reservoir+delay multimodal content fusion。

### 数据生成与独立结果

- 官方runner生成/采集wall=`228.84 s`；新valid/test各8 games、40 steps、80真实counterfactuals，全部won。train manifest/episodes/token-events SHA与原train逐字节相同；fresh summary SHA train/valid/test=`1ab61124.../99db1105.../02647dae...`。
- SG10审计器原先把heldout每event length固定为4 groups；在任何模型运行前泛化为`expected_groups/5 lengths`，旧4-game审计结果不变，fresh每length=8。修正后counts/groups/OOV/vocab/seed隔离/provenance全部PASS。
- frozen primary/kernel/`λ=1e-6`/train-CV与原SG13数值复现断言全部通过。正式命令为SG13 runner加`--fresh-confirmation --corpus-dir ... --expected-counts 480 120 120 --expected-groups 160 40 40`；wall=`21.51 s`，artifact SHA-256=`A2EDBE97273FA1AEDCE8A34393C618B48CF613A41D6C55BA234A3D22061A8F96`。
- fresh test exact/macro=`.9333/.9833`，room/reward/done=1.0、exit=`.9333`；`<exit_count_1>` recall=`.85`、`exit_count_2=.9722`，共8/120错误。冻结`.98`门明确FAIL，不能用样本数扩大解释掉。
- train-CV仍精确`.9896`，training wall=`56.30 ms`，online等价PASS；fresh cached p50=`.0824-.0831 ms`、p95=`.1457-.1509 ms`，模型bytes仍`23,520`，速度/内存结论复现但随质量门失败。

**决定：独立 overall FAIL，撤回“suffix kernel跨procedural worlds已确认”的扩张性结论。** 保留原SG13为机制证据、闭式/在线训练与实时工程证据。先按上方SG10R重跑fresh ANN task controls；然后才决定是reservoir内容融合还是更大真实train覆盖。

---

## 2026-07-19：E3-SG13 结果 — hierarchical suffix spike kernel associative memory（机制PASS / 待独立确认）

### SG12 证明精确状态与RLS都足够快，但线性二阶读出无法组合多步历史

SG12 的K3 delay state在train key层面无歧义，K1→K2→K3→K4 test exact也单调`.8833→.9167→.9333→.9667`，说明精确保留历史确实有因果价值；然而feature只含“每个lag×candidate”的二阶项，不能表达三个历史动作的联合conjunction。下一步不加ANN recurrence，而把固定脉冲延迟线升级为**分层后缀核联想记忆**：深层精确匹配负责已见组合，浅层后缀匹配为未见组合提供平滑回退。

### PSD事件核、训练与在线状态

- 每例状态写成四个categorical spike slots `[a_{t-2},a_{t-1},a_t,candidate]`，不足三步用专用`<history_pad>`；另维护不读label的event phase。状态仍由K3 bool shift register递推，外加一个整数/单位根phase。
- 对两个事件状态定义嵌套delta kernels：`k0=1[candidate相同]`，`k1=k0·1[last1相同]`，`k2=k1·1[last2相同]`，`k3=k2·1[last3相同]`，`kp=1[phase相同]`。每项都是categorical one-hot内积，非负和保持PSD。
- **primary在运行前冻结**为`k=k0+k1+k2+k3+kp`；它同时编码SG12确认的三阶内容与SG11确认的相位。controls=`candidate_only/suffix1/suffix2/suffix3_no_phase/depth_weighted_phase`全部预注册，但不据原test选择primary。
- kernel ridge用train 480 prototypes直接解`α=(K+λI)⁻¹Y`；lambda不再依赖4-game valid，而由train 4-fold game-seed CV选择（每fold按8个procedural games隔离），规则仍为CV exact/macro/MSE/lambda。官方valid/test均不参与选择。
- online训练实现10×48 block Schur/Woodbury kernel inverse append：每个block只需旧inverse、cross-kernel与新block kernel，最终alpha/scores必须与batch KRR差<=`1e-6`且prediction完全一致。
- cached inference从24-bit delay state与phase生成query key，向量化比较480个prototype的candidate/1/2/3阶suffix与phase，再做`480×11`alpha readout；报告prototype/alpha/state bytes、真实p50/p95与batch full-key等价。

### smoke fail-closed后的数值实现修正（未移动模型门）

首次smoke中primary、lambda=`1e-6`和test quality已冻结，但“显式维护kernel inverse”的block Schur在大量重复prototype下病态发散，score与batch相差`~7e9`，因此online equivalence fail-closed。未改数据/kernel/权重/lambda/quality/测试集；只把同一Schur递推写成block Cholesky factor update：`V=L⁻¹B`、`S=D-VᵀV`、追加`chol(S)`，最终用`cholesky_solve`求alpha。修正后factor重建误差`5.3e-15`、score差`1.2e-14`、prediction完全一致；这是数值稳定实现修复，不是事后调模型。

### 冻结门

- **PRIMARY QUALITY**：official test exact>=.98、macro>=.98、各channel>=.95、rare positive>=.90，且不低于SG10 Transformer exact/macro-.02。原test已被前序实验查看，故即使PASS仍只算机制证据，必须SG13R fresh games确认。
- **MECHANISM**：primary CV out-of-fold exact>=suffix2 control，且official test至少比SG12 K3 `.9333`高`.04`或直接过`.98`绝对门；suffix3_no_phase用于隔离phase贡献但不要求必须更差。
- **TRAIN**：完整4-fold lambda选择+full fit wall<=Transformer三seedmean training wall；block-online最终等价且单次full-data pass wall<=Transformer。若CV研究成本失败，不能只报部署fit快。
- **STREAM**：质量通过后，三个计时replication的p50/p95均<=对应Transformer，full/persistent score差<=`1e-6`；模型存储bytes<=Transformer参数bytes。
- **What if：**spike delay line不应把所有多步组合展开成巨型高阶张量，而应通过后缀核把“完全相同的因果片段”和“部分相同的最近事件”统一成可闭式求解的联想能量；这能否在稀疏事件流上获得Transformer级组合质量，同时保持无反传、在线可更新与亚毫秒响应？
- 若PASS，立即生成fresh procedural valid/test做SG13R，并把kernel scores接candidate planner闭环；若quality仍失败，说明仅靠action suffix无法泛化，转`frozen gated reservoir + exact delay/phase kernel`的内容融合；若速度失败，压缩prototype为unique keys/Nyström dictionary。

### 正式结果

- 正式命令：`.venv-wsl/bin/python experiments/e3_sg13_suffix_spike_kernel.py --device cpu --threads 4 --output results/e3_scan/e3_sg13_suffix_spike_kernel.json`；wall=`17.59 s`，artifact SHA-256=`1DF3593277FED31B2624DAC27AD486E368203B3D9C76079A36BA91F5FFEC8C6E`。
- primary=`suffix3_phase`、lambda=`1e-6`均由预注册/4-fold train-game CV冻结；official valid/test从未参与选择。train OOF exact/macro=`.9896/.9974`，exit=`.9896`，其余channel=1.0。
- 原official test exact/macro=`.9833/.9958`，room/reward/done=1.0、exit=`.9833`，仅1/60 exit错误；相对SG12 K3 exact提高`.0500`，**QUALITY/MECHANISM PASS**。
- primary完整train feature record + kernel + 4-fold lambda selection + full fit wall=`44.34 ms`，vs Transformer 50-epoch mean=`1.0893 s`，约快24.6x；不是只比较最后一次solve。
- block Cholesky–Schur三种48-example order均prediction完全等价，score差<=`1e-6`，不维护病态显式inverse；smoke修正未改变primary/lambda/quality。
- cached三replication p50=`.0738-.0754 ms`、p95=`.1251-.1328 ms`，约比Transformer快4x；persistent/full差0。480 prototypes + uint8 keys/phase + float32 alpha逻辑存储=`23,520 bytes`，低于Transformer参数`81,504 bytes`；persistent state=`32 bytes`。

**决定：DATA/QUALITY/MECHANISM/ONLINE/TRAIN SPEED/CACHED STREAM/overall全部PASS。** 但这只是已被研究反馈接触过的原test机制结果；不宣称通用世界模型或最终替代ANN。按上方SG13R冻结架构，在全新procedural game seeds上独立确认后才进入闭环。

---

## 2026-07-19：E3-SG12 结果 — sparse spike delay-line + Block-Woodbury/RLS（训练与速度正面 / 质量负面）

### SG11 说明相位有用，但真正的最小充分状态是短期动作历史

SG11 的 one-hot phase oracle 仅把 mean exact 从`.9667`提高到`.9833`，而valid-only选中的unit-root仍只有`.9722`；单独时钟不是稳定解。随后做只读条件歧义审计：在train上，`last1 context action + candidate` majority accuracy=`.8708`，`last2+candidate=.9646`且仍有冲突，`last3+candidate=1.0`且无任何同key异label；因此下一变量应是**精确保留最近3个typed action spikes**，不是继续堆衰减常数。

同一审计已不可逆地查看原valid/test：last3 key覆盖valid/test=`42/60,55/60`，覆盖部分均100%正确。故SG12在原test上的结论只算机制验证，不能冒充独立确认；若通过，必须另生成未见game seeds做SG12R。

### 纯SNN状态与两种数学训练解

- train action alphabet固定为8个真实typed events：四方向move、inventory、look、examine coin、take coin；不读取game id、label、observation future或TextWorld规则。
- 三阶delay-line state为`d_t=[onehot(a_t),onehot(a_{t-1}),onehot(a_{t-2})]∈{0,1}^24`，每个event只做固定block shift并注入8-bit spike；这是无可学习循环参数的稀疏脉冲移位寄存器。`<event_start>`不写action block，空历史由全零slot表示。
- 世界读出feature固定为`[1,d_context,candidate,d_context⊗candidate]`，维度=`1+24+8+192=225`；11通道突触权重仅`2,475`，显著少于SG10约20k参数ANN。primary K=3在运行前冻结；K=1/2只作不足记忆负对照，K=4只作容量上界，不能据test改primary。
- **batch primal ridge**：对225维feature直接解`(XᵀX+λI)W=XᵀY`；lambda grid仍仅由valid exact/macro/MSE选择。
- **online block RLS/Woodbury**：从`P0=λ⁻¹I,W0=0`出发，每个48-example真实batch更新 `K=P Xᵀ(I+X P Xᵀ)⁻¹`、`W←W+K(Y-XW)`、`P←P-KXP`；10个block一次遍历，不做反传/epochs。必须与同lambda batch ridge train/valid/test scores及weights数值等价，报告每block与总wall。
- cached stream维护24-bit delay state；候选路径只做一次shift、225维稀疏feature和11-logit线性突触读出。计时包含全部三项，逐seed与SG10达标Transformer cached p50/p95比较；state bytes与multiply-add上界必须报告。

### 冻结门与否证条件

- 数据、四channel、三model seeds、SG10 Transformer正式参考及其SHA完全不变。**PRIMARY QUALITY**：K3 batch ridge mean exact>=.98、macro>=.98、各channel>=.95、rare reward/done recall>=.90，且不低于Transformer exact/macro-.02；K1/2不参与选择。
- **RLS EQUIVALENCE**：每seed K3 RLS与batch ridge prediction完全一致，max score difference<=`1e-6`（若数值条件只能达到更宽阈值，正式门FAIL而不事后放宽）；一次遍历training wall<=Transformer 50-epoch wall。
- **STREAM**：K3逐seed cached p50/p95<=Transformer，full-history重建与persistent delay state scores差<=`1e-6`；质量失败时速度门随之失败，不能用快但错误的模型过门。
- **MECHANISM**：K3 mean exact至少比K1与K2各高`.01`或两者未过绝对`.98`门；否则“三阶最小充分状态”解释不成立。
- **What if：**实时世界模型并不需要用衰减模拟所有历史，而可以把最近的稀疏因果事件通过固定脉冲轴突延迟精确保留，再用一次闭式/递归突触求解完成多通道预测；这种有限阶精确状态能否同时消除Transformer质量差距与SNN迭代训练成本？
- 若原test机制门PASS，生成新valid/test procedural game seeds做独立SG12R并接真实closed-loop rollout；若K3失败而K4过，说明Markov order审计不足，转自适应稀疏 associative memory；若K4也失败，恢复随机reservoir内容分支并学习临界/unit-circle basis，而不假称delay line已替代通用世界模型。

### 正式结果

- 正式runner：`experiments/e3_sg12_spike_delay_rls.py`；formal artifact=`results/e3_scan/e3_sg12_spike_delay_rls.json`，SHA-256=`EB776BC2884CA6E23EEA319A81312098EF8AC93111FC62919BE8E7EEBE5EA76C`。
- 8-event alphabet、K3 train零歧义、K2仍12个冲突keys以及SG10真实数据/provenance门全部通过；K3 state=`24 bool bytes`，feature/readout=`225/2,475`，每例实际仅激活2..8个features。

| exact test | K1 | K2 | K3 primary | K4 upper |
|---|---:|---:|---:|---:|
| sparse delay ridge | .8833 | .9167 | **.9333** | .9667 |

- K3 room/reward/done均1.0，exit count=`.9333`，exact/macro=`.9333/.9833`；相对K1/K2分别+`.0500/.0167`，故 **MECHANISM PASS**，但远低于`.98`绝对质量门。K4虽继续提高，也只有`.9667`，按预注册不能把“多存一步”当成功。
- **RLS EQUIVALENCE PASS**：三种block order均与batch ridge prediction完全一致，max score difference<=约`2.3e-13`；10个48-example block一次遍历，lambda选择+RLS端到端wall=`.042-.049 s` vs Transformer=`1.069-1.130 s`，约快22–27x。
- **实时路径强正面但随质量门失败**：p50=`.0313-.0320 ms`、p95=`.0620-.0662 ms`，比Transformer `.294-.300/.524-.538 ms`约快9x/8x；full/persistent score差0，state 24 bytes，dense readout 2,475 MACs。

**决定：DATA/MECHANISM/RLS/TRAIN SPEED PASS，PRIMARY QUALITY/CACHED STREAM/overall FAIL。** 数学训练加速路线有效，失败点是二阶readout对多步categorical conjunction表达不足。K4也未过门，因此不走单纯order扩张，按上方SG13改为分层suffix spike kernel联想记忆。

---

## 2026-07-19：E3-SG11 结果 — recursive temporal basis for persistent SNN state（诊断正面 / 部署门失败）

### SG10 的剩余缺口是状态相位，不再是热路径速度

SG10 中 Transformer 三 seed 的 exact/macro/channel 全为 1.0，而三种 SNN 的 reward/done 已全对，错误几乎全部集中于 history length 3..6 的 exit count，少量 room relation 错误也发生在长 history。RA0 cached candidate 已比唯一达标 ANN Transformer 快两倍以上；继续优化 kernel 没有因果依据。Transformer core 独有绝对 sinusoidal position，SNN/LSTM 只依赖衰减状态，因此先用不读取 label 的递归时间基检验“缺少稳定事件相位”假设。

### 五条数学路线与最小决定性对照

| 路线 | 递推状态 | 作用 | 地位 |
|---|---|---|---|
| baseline | 原 SG10 gated trace | 复现 frozen ridge `.9667` exact，防止 runner 漂移 | 必须复现 |
| unit-root | `c_t=c_{t-1}+1` | 无衰减累计事件年龄，最小长期计数器 | deployable 诊断 |
| multiscale leaky | `z_t=Λz_{t-1}+(I-Λ)1` | 多时间常数稳定逼近事件年龄 | 纯 affine-scan 候选 |
| oscillator | 二维 rotation blocks | 用周期相位避免单位根无界增长 | 振荡 SNN 候选 |
| binary spike | 三个翻转 bit | 以离散 spike 状态精确编码短期事件计数 | speculative 候选 |
| one-hot oracle | 六状态 ring shift | 给出“事件相位足够时”的有限任务上界 | 只诊断，不可选部署 |

### 冻结设计、选择规则与门

- 数据、四个真实 channel、train/valid/test=`480/60/60`、frozen RA0 SNN reservoir 初始化 seeds、D32/state31、class code、lambda grid与SG10完全一致；引用的SG10正式产物 SHA-256=`56BD001A17AD7093F4B3A37329B9B2083AD127F848072F5881148C941C02A77F`，运行时必须校验。
- 各时间基只由“收到一个event”递推，不读取action、game、target或future；在两个query位置生成状态，替换hidden末尾同等数量坐标后再构造 `[1,h_prev,h_candidate,h_prev⊗h_candidate]`，故所有 variant feature dimension/readout parameter count固定为`1,089/11,979`，不靠扩参数获胜。
- deployable candidates=`unit_root1/leaky4/oscillator4/binary3`；按 valid exact、valid macro、valid MSE、状态维度、固定名称顺序选择，test只在选择后用于主结论。one-hot6明确排除选择，仅判断相位信息上界。
- 所有 closed-form wall 包含 train+valid feature extraction与全lambda solve；stream计时必须包含 cached-decay SNN candidate step、递推时间基更新、feature标准化与11-logit readout，并逐seed与SG10唯一达标ANN Transformer的cached p50/p95比较。
- **DIAGNOSTIC PASS**：baseline test与SG10 frozen ridge每seed exact误差<=`1/60`，且one-hot oracle mean exact>=.98。**DEPLOYABLE QUALITY**：valid-only winner mean test exact>=.98、macro>=.98、各channel>=.95、reward/done positive recall>=.90，并在exact/macro不低于SG10 best ANN-.02。**SPEED**：closed-form wall<=Transformer iterative wall，逐seedstream p50/p95<=Transformer；full/cached feature logits差<=1e-5。
- **What if：**当前 gated SNN 并非缺少真实世界事件内容，而是所有trace都严格衰减、没有能跨事件保存“我处于第几阶段”的自主相位载波；若加入极小的递归脉冲时钟且保持读出维度不变，能否补齐 Transformer 的绝对位置优势，同时保留闭式训练和更快单事件响应？
- 若deployable basis过门，下一步把赢家写进持久SNN state并做online RLS/Block-Woodbury closed-loop rollout；若oracle过而deployable失败，增加可学习临界/unit-circle basis；若oracle也失败，否证单纯相位缺失，转action-history associative state而非继续调clock。

### 正式结果

- 正式命令：`.venv-wsl/bin/python experiments/e3_sg11_temporal_basis.py --device cpu --threads 4 --output results/e3_scan/e3_sg11_temporal_basis.json`；wall=`20.63 s`，结果 SHA-256=`C1877A3741A0012C5456631E06145F4488F48F0432E91E05E69F3B9EC6ACEE86`。
- 三seed baseline分别精确复现SG10 frozen ridge exact=`.9833/.9667/.9500`，差均为0；所有variant feature/readout固定`1,089/11,979`，reference SHA运行时校验通过。
- valid-only全局选择为`unit_root1`；其mean test exact/macro=`.9722/.9917`，room/reward/done/exit=`.9889/1/1/.9778`，未达到`.98` exact与best ANN-.02门。
- one-hot oracle mean test exact=`.9833`，故 **DIAGNOSTIC PASS**：稳定相位确实能解释至少一部分SG10误差；但它只是有限六状态上界，不能作为部署解。
- 预注册的各variant exploratory test显示oscillator4与binary3均为`.9889` exact，但它们没有赢得valid-only选择；看过test后不得反选并宣布成功，只能为新独立数据提出假设。
- selected closed-form training wall均值=`30.31 ms` vs Transformer=`1.0893 s`，约快35.9x。含 cached-decay SNN step、unit-root更新、1089维标准化与readout后的p50/p95三seed=`.1244/.2009,.1248/.2122,.1306/.2254 ms`，全部快于Transformer约`.294-.300/.524-.538 ms`；full/cached差<=1e-5。

**决定：DIAGNOSTIC与closed-form/stream SPEED PASS，DEPLOYABLE QUALITY及overall FAIL。** 不降低`.98`门，也不根据已看test改选oscillator/binary。结合train-only最小Markov阶审计，按上方SG12用精确三阶spike delay line验证“内容保真”而非继续只调clock；同一原test结果必须明确标为机制证据。

---

## 2026-07-19：E3-SG10 结果 — multichannel action-conditioned event delta（质量负面 / 速度正面）

### 从单relation扩到真实多通道结果

SG9R 已确认atomic cached SNN在单通道上稳定快于ANN，但这可能是四方向逆关系的特例。SG10 使用每个官方episode的全部5个factual steps及每步2个真实counterfactuals，不再只选hard move；输入为 `<start> + prior factual action events + candidate action event`，输出同时覆盖语言状态关系、reward、done与可行动出口数。

### 只读数据审计与冻结设计

- train/valid/test=`480/60/60` examples，step groups=`160/20/20`，每group恰含1 factual+2 counterfactual；event sequence length=`2..6`，每个长度train恰96例/32 groups，可按长度无padding组成`16 groups ×3=48 examples` batch。
- 四个真实channel：`room_relation={no_room,novel,previous,same}` 分布train=`192/128/128/32`；`reward={zero,positive}`=`448/32`；`done={continue,done}`=`448/32`；`move_exit_count_after={0,1,2}`=`32/160/288`。room由normalized next_obs的真实room feature与current/prior observations比较；其余直接来自reward/done/admissible_actions_after或下一factual step。
- `won`与本语料中的reward/done完全共线，不单列以免虚增通道；episode没有可验证的post-step inventory state，inventory留到支持该字段的数据集，绝不从action字符串伪造label。
- test 60条中52条完整action histories在train出现、8条未见；所有split input→label vector无歧义。exact-vector majority仅`.20`；reward/done majority虽`.933`，因此必须报告rare positive recall，不能靠全预测continue过门。
- 五模型使用同初始化 `Bilinear(D,D,11)` multi-head logits，按train class frequency固定inverse-frequency CE weights；50 epochs、每epoch10 length-stratified batches、500 updates/model、24,000 example exposures，seeds/CPU4不变。参数spread<=3%。
- frozen SNN outer features用valid-only lambda grid做multi-output dual ridge；选择顺序固定为valid exact accuracy、macro channel accuracy、MSE、lambda。batch训练通过后，cached candidate每例重复128次+16 warmup，并要求逐seed与最快达标ANN比较。
- **TASK**：best ANN exact-vector>=.90、每channel accuracy>=.95、reward-positive与done-positive recall>=.90。**SNN QUALITY**：RA0与ridge分别满足同绝对门，并在exact/macro上不低于best ANN-.02；BPTT/AT1作为梯度对照。
- **TRAIN/STREAM SPEED**：closed-form fit wall或RA0 batch<=最快达标ANN；RA0 cached-decay逐seedp50/p95<=最快达标ANN，full/cached logits差<=1e-5。
- **What if：**bilinear atomic SNN是否不仅能识别方向反转，还能把同一持久event state一次映射成空间、价值、终止与可行动性多个世界通道，同时保留闭式训练和单事件响应优势？
- 若PASS，下一步把batch dual ridge改成可在线更新的recursive/block-Woodbury least squares并接closed-loop rollout；若ANN过而SNN失败，增加多尺度状态而不回退文本surface；若速度失败，进入compiled fused event step。

### 正式结果

- 正式命令：`.venv-wsl/bin/python experiments/e3_sg10_multichannel_delta.py --device cpu --threads 4 --output results/e3_scan/e3_sg10_multichannel_delta.json`；wall=`101.45 s`，结果 SHA-256=`56BD001A17AD7093F4B3A37329B9B2083AD127F848072F5881148C941C02A77F`。
- 数据门完整通过：vocabulary size=`13`、fingerprint=`9d31551deae8300719634f1fc584b6ec508348c74b94efd3c456eb2096f1f749`；test中52/60完整inputs被train覆盖、8条未见，且跨split已覆盖input没有label冲突。

| model/path | exact ↑ | macro ↑ | room/reward/done/exit ↑ | train example p50 ms ↓ |
|---|---:|---:|---:|---:|
| SNN-BPTT | .9111 | .9764 | .9833/1/1/.9222 | .05077 |
| SNN-AT1 | .9111 | .9764 | .9833/1/1/.9222 | .06304 |
| **SNN-RA0** | **.9111** | **.9764** | **.9833/1/1/.9222** | **.03988** |
| LSTM | .9333 | .9833 | 1/1/1/.9333 | **.02881** |
| Transformer | **1.0000** | **1.0000** | **1/1/1/1** | .04089 |
| frozen SNN ridge | .9667 | .9903 | .9889/1/1/.9722 | closed-form wall=.03567 s |

- **DATA/TASK/TRAIN SPEED PASS**：Transformer三seed全部完全正确；frozen ridge平均训练wall=`35.67 ms`，快于达标Transformer iterative training。RA0 batch也略快于Transformer，但仍慢于未达质量门的LSTM，故不把后者当合格速度基线。
- **SNN/RIDGE QUALITY FAIL**：RA0 exact=`.9111`，ridge=`.9667<1.0-.02`。reward/done及rare positive全部正确；错误主要是 exit count，RA0 seed0/1/2分别有`3/7/4`个exit错误，seed2另有3个room `novel→previous`，集中于step/history length 3..6。LSTM错误也主要是exit，而Transformer全对。
- **CACHED速度本身为强正面证据，但门随质量失败**：RA0三seed cached p50约`.141/.142/.147 ms`、p95约`.220/.235/.246 ms`；唯一达标ANN Transformer约`.300/.53 ms`。即热路径已超过ANN两倍，当前应修状态质量而非继续只做kernel优化。

**决定：overall FAIL，不降低冻结质量门。** 保留SG10真实多通道任务作为后续共同基准；按上方SG11先验证稳定时间相位/长期计数状态，再把有效数学基写入纯SNN持久状态。速度结论只表述为“在质量未达标的当前SNN上已有余量”，不宣称完整胜利。

---

## 2026-07-19：E3-SG9R 结果 — cached latency high-repeat replication（确认正面结果）

### 为什么SG9均值PASS后仍不直接扩任务

SG9正式每seed只有24个candidate latency samples；均值门PASS，但seed0 RA0 cached-decay p95约`.24 ms`，慢于同seed LSTM约`.21 ms`。p50三seed都接近，少量调度噪声足以改变p95结论。工程化实时门需要更强证据，不能用72个总样本宣布稳定超越。

- 代码、数据、模型seeds、训练、atomic events、quality gates全部不变；只把每个test candidate在prefix state已缓存后重复`256`次，另做`32`次不计时warmup，即每mode/seed记录`6,144`个candidate samples。
- 所有模型仍4 CPU threads；SNN使用cached-decay一步，LSTM/Transformer使用原生cached state/KV。计时包含candidate token tensor、embedding、一次core update与bilinear head，不含已完成的prefix。
- **严格确认门**：三个seed分别都要求RA0 cached-decay p50与p95<=同seed最快达标ANN；accuracy/step=1.0，cached/full max logit差<=1e-5。不能再用跨seed均值抵消单seed失败。
- 若严格门PASS，SG9实时结论升级为confirmed并进入多通道；若FAIL，保留quality/closed-form训练结论，转1/2/4/8/16线程与compiled fused event-step，不扩张实时胜利。

### 复现结果

- 命令：`.venv-wsl/bin/python experiments/e3_sg9_atomic_event_stream.py --device cpu --threads 4 --timing-repeats 256 --timing-warmup-repeats 32 --strict-per-seed-stream --output results/e3_scan/e3_sg9r_atomic_event_stream_replication.json`；wall=`70.65 s`，SHA-256=`BD8B610E6FE9D0CE2EE6F7D4883669EAC0BAAB3BBEFF9782ECE08ED448912DEF`。
- 每mode/seed正式candidate samples=`6,144`；所有模型/ridge accuracy/step=1.0，cached/full equivalence PASS。

| seed | RA0 cached p50/p95 ms ↓ | fastest ANN p50/p95 ms ↓ | seed gate |
|---:|---:|---:|---:|
| 0 | .1007/.1804 | .1382/.2295 | PASS |
| 1 | .0991/.1830 | .1469/.2487 | PASS |
| 2 | .0991/.1724 | .1347/.2330 | PASS |
| mean | **.0997/.1786** | .1399/.2371 | **PASS** |

- RA0 p50/p95相对最快ANN分别快约28.8%/24.7%；三个seed分别过门，消除了SG9 seed0 24-sample p95反转。
- prefix state bytes SNN/LSTM/Transformer=`248/256/256`；候选热路径只处理1 atomic event。闭式ridge平均训练wall=`11.61 ms`，质量继续全过。

**决定：confirmed PASS。** 在当前真实room-relation事件任务上，`bilinear binding + frozen closed-form readout + persistent atomic SNN state + cached-decay step`已同时达到ANN质量、明显更快训练方案和逐seed更快实时响应；但结论严格限于该单通道任务。按上方SG10验证多通道而非直接宣称世界模型完成。

---

## 2026-07-19：E3-SG9 结果 — atomic event streaming + cached bilinear SNN（正面结果 / 延迟待高重复确认）

### SG8 已解决质量与训练时间，剩余瓶颈是重复prefill

SG8 中 trainable bilinear SNN 与 frozen-reservoir ridge 都达到三seed accuracy/step=1.0；closed-form训练比达标LSTM快约19.9x，RA0 batch也快于Transformer。当前response门仍失败，是因为每个候选都从零重算17-token格式prompt；真实实时世界模型应长期持有previous-event state，只对新candidate event做增量响应。

### 冻结设计

- 从同一真实TextWorld SG6 examples生成两个atomic event tokens：`previous_action`与`candidate_action`各映射为一个train-vocabulary move token；label仍由真实`next_obs` history membership生成，counts/groups/seeds完全不变，不使用方向规则赋label。
- 五模型继续D32/state31 + identical bilinear head，paired 32-example batch、50 epochs/300 updates/9,600 exposures；query从文本位置`(5,11)`缩为event位置`(0,1)`。closed-form ridge仍使用冻结SNN hidden outer features与valid-only lambda选择。
- 推理报告两种口径：`full_event_pair`从零处理2 events；`cached_candidate`预先处理previous event并缓存core state/h_prev，只计1个candidate event + bilinear head。每step两个候选复用同一prefix cache，必须与full输出逐例一致。
- SNN额外比较 generic one-token core 与 cached-decay tensor step；decay只在参数更新后预计算一次。LSTM/Transformer使用各自原生state/KV cache，不能只给SNN缓存。
- **QUALITY**：所有候选SNN mean accuracy>=.98、step>=.95且不低于best ANN-.02；ridge同门。**TRAIN SPEED**：RA0/closed-form各自与最快达标ANN比较。**STREAM SPEED**：RA0 cached candidate p50/p95均<=最快达标ANN，accuracy完全一致；同时记录prefix amortization与state bytes。
- **What if：**多模态世界模型的内部语言不是带格式的token句子，而是持续到达的稀疏typed events；若SNN只为新事件更新一次状态，是否能把其常数扫描优势转成真实闭环响应优势？
- 若质量与cached speed全过，下一步把room relation扩成reward/done/inventory/exit多通道，并把batch ridge改为recursive least squares；若仍只差推理，进入`torch.compile`/C++ fused event-step kernel，不再改变任务。

### 正式结果

- atomic vocabulary size=`10`、fingerprint=`57231174773d2471e6cd666c69e651aad7d384757b387e4f0e449161d182562a`；4个move action tokens一一映射，无collision/OOV，labels继续来自真实next_obs membership。
- 正式命令：`.venv-wsl/bin/python experiments/e3_sg9_atomic_event_stream.py --device cpu --threads 4 --output results/e3_scan/e3_sg9_atomic_event_stream.json`；wall=`40.97 s`，结果 SHA-256=`244AAEDBF33DD6E04076765A3DF7C607425CD490C973AADD8B5D77A238975B69`。

| model/path | accuracy/step | train example p50 ms ↓ | train wall s ↓ | cached candidate p50/p95 ms ↓ |
|---|---:|---:|---:|---:|
| SNN-BPTT generic | 1.0/1.0 | .04583 | .5076 | .229/.328 |
| SNN-AT1 generic | 1.0/1.0 | .06655 | .6836 | .218/.301 |
| **SNN-RA0 cached-decay** | **1.0/1.0** | **.03894** | **.4030** | **.1386/.1947** |
| LSTM cached state | 1.0/1.0 | **.02776** | **.3218** | .1474/.2127 |
| Transformer KV | 1.0/1.0 | .04032 | .4367 | .2914/.3572 |
| frozen SNN ridge | **1.0/1.0** | — | **.0131** | [待SG9R独立缓存读出计时] |

- **DATA/QUALITY/RIDGE QUALITY PASS**：五trainable模型与closed-form ridge三seed accuracy/step全部1.0；atomic encoding没有改变真实标签或game split。
- **TRAIN SPEED PASS由closed-form驱动**：ridge平均`13.1 ms`，比LSTM`.3218 s`快约24.5x。RA0 iterative batch仍比LSTM慢约25%，不能混为同一结论。
- **CACHED STREAM均值门PASS**：RA0 cached-decay mean p50/p95=`.1386/.1947 ms`，比LSTM`.1474/.2127 ms`约快6.0%/8.4%；所有cached logits与full pair误差<=1e-5。
- **稳定性限制**：seed0 p50仍略快，但p95约`.24>.21 ms`；seed1/2均过。每seed仅24 samples，故先执行上方SG9R，不把mean PASS描述成已稳定超过ANN。

**决定：atomic typed events + persistent SNN state 是正确的实时系统边界；当前质量与closed-form训练加速已确认，推理领先仅为初步正面证据。** SG9R严格复现通过后才扩多通道。

---

## 2026-07-19：E3-SG8 结果 — bilinear spike binding + closed-form ridge（正面机制 / 工程门未过）

### SG7 对优化与结构的判别

SG7 的 paired binary batch 使 Transformer 三seed全为1.0，也把 RA0 每例训练时间压低约23倍；但三种SNN binary NLL都停在`.673±.001`、accuracy仅`.56–.61`。由于 BPTT/AT1/RA0 同时失败且彼此NLL几乎一致，继续调整 reverse-adjoint 没有因果依据；缺失的是 `previous action × candidate action` 的显式二阶绑定。

### 两条互补数学路线

- **SG8-A trainable bilinear binding：**五种core都从action末端的两个hidden取 `(h_prev,h_candidate)`，通过同构 `nn.Bilinear(D,D,2)` relation head；SNN-BPTT/AT1/RA0分别保持其时间梯度。仍用SG7 paired batch、50 epochs、300 updates、9,600 examples exposures，唯一结构变量是二阶读出。
- **SG8-B frozen SNN reservoir + closed-form ridge：**冻结与SG8-A同初始化的SNN trace，构造 `[1,h_prev,h_candidate,vec(h_prev⊗h_candidate)]`；用dual ridge一次求解标量relation score。lambda网格固定为`1e-6,1e-4,1e-2,1,100`，仅按valid accuracy最高、valid MSE最低、lambda最小的字典序选择，test只评一次。
- SG8-A五模型共享完全相同的bilinear head初值；参数spread必须<=3%。SG8-B报告feature extraction、五lambda solve总wall、feature维度、readout参数和完整prompt inference；不把直接action lookup当SNN。
- **TASK**：bilinear LSTM/Transformer中至少一个 mean accuracy>=.98、step>=.95。**TRAINABLE-SNN QUALITY**：RA0 accuracy>=best ANN-.02且>=.98、step>=.95、binary NLL<=best ANN+.05，且与BPTT/AT1 accuracy gap<=.02/NLL gap<=.05。
- **RIDGE QUALITY**：frozen SNN ridge 三seed mean accuracy>=.98、step>=.95，且每seedvalid选择不读取test。**SPEED**：候选SNN的训练/fit wall<=最佳达标ANN且完整prompt response p50/p95<=最佳达标ANN；同时保留RA0相对AT1/BPTT>=1.25x门。
- **What if：**真实世界模型中的稀疏事件关系不应让一阶trace自己“涌现”乘法，而应把二阶突触绑定作为原生SNN算子；进一步，若reservoir已保留足够事件身份，读出甚至可以通过一次闭式求解完成？
- 若SG8-A过而ridge失败，说明需要端到端塑形二阶动力学；若ridge过，优先沿闭式/递推最小二乘扩展在线世界状态；若两者都失败，转显式group-equivariant direction code，再测试未见action-pair split。

### 正式结果

- 五模型bilinear head均`2,050`参数，总参数=`11,110/11,110/11,110/11,156/11,316`，spread<3%；frozen SNN ridge feature/readout维度=`1,089`。
- 正式命令：`.venv-wsl/bin/python experiments/e3_sg8_bilinear_closed_form.py --device cpu --threads 4 --output results/e3_scan/e3_sg8_bilinear_closed_form.json`；wall=`43.27 s`，结果 SHA-256=`57AAF0E89BBB94B7C5562A43A6A4BD5CD529BF12573C0D8109EEDC526B9D96DD`。

| trainable model | binary NLL ↓ | accuracy/step ↑ | example p50 ms ↓ | train wall s ↓ | response p50/p95 ms ↓ |
|---|---:|---:|---:|---:|---:|
| SNN-BPTT | 1.11e-5 | 1.0/1.0 | .09306 | .9939 | .407/.539 |
| SNN-AT1 | 1.21e-5 | 1.0/1.0 | .09376 | .9955 | .392/.527 |
| **SNN-RA0** | 1.11e-5 | **1.0/1.0** | **.05593** | **.5930** | .406/.652 |
| LSTM | 3.17e-5 | 1.0/1.0 | **.03310** | **.3754** | **.175/.262** |
| Transformer | **1.02e-5** | 1.0/1.0 | .06238 | .6554 | .349/.448 |

- **TASK与TRAINABLE-SNN QUALITY PASS**：三种SNN、LSTM、Transformer全部三seed accuracy/step=1.0；RA0 NLL与BPTT/AT1/Transformer差<`2e-6`。显式二阶绑定消除了SG7的一阶trace结构瓶颈。
- **RIDGE QUALITY PASS**：三seed frozen SNN ridge均由valid选择lambda=`1e-6`，valid/test accuracy/step=1.0；test从未参与lambda选择。平均train+valid feature extraction与五次dual solve=`18.90 ms`，约为达标LSTM训练wall的1/19.9。
- **训练加速为正面证据**：RA0相对AT1/BPTT=`1.677x/1.664x`，并比Transformer每例快约10.3%；closed-form进一步把迭代反传移除。
- **TRAINABLE/RIDGE SPEED均因stream response FAIL**：RA0与ridge完整prompt p50约`.406/.380 ms`，LSTM `.175 ms`；RA0 batch训练也仍比LSTM慢约69%。因此 overall FAIL，不能把质量与训练wall胜利扩张为完整训推胜利。

**决定：采用bilinear event binding与closed-form readout作为SNN主线组件，不采用一阶trace单独承担关系推理。** 按上方SG9去除重复文本prefill并公平比较各core缓存状态；只有cached真实响应过门后才扩多通道。

---

## 2026-07-19：E3-SG7 结果 — paired binary batched move-delta（混合结果）

### SG6 暴露的是关系绑定与优化稳定性，不是数据不可识别

SG6 把 surface 移除后，Transformer 两个 seed 完全正确、一个 seed 只错 2/24；所有 test action triples 都在 train 出现且 label 无歧义，因此 compact state-delta 是可学任务。失败集中于当前 B1/full-vocabulary CE：五模型每条样本都对18词全词表更新，而真正的状态通道只有两个 label；SNN 尤其在 `<novel_room>` 的非反向 action pair 上欠拟合，RA0 seed0 还落入相反类别的坏盆地。

### 数学路线比较

| 路线 | 类型 | 最小决定性实验 | 价值 | 主要风险 |
|---|---|---|---|---|
| R1 paired two-logit batch | established direction | 每个真实step的正负候选同batch，仅对两个relation logits做CE | 同时消除无关词梯度、把9,600次B1更新压成300次并行更新 | dedicated channel可能不改善通用LM head |
| R2 bilinear/tensor-product spike binding | speculative new idea | 在previous/candidate event之间加入二阶脉冲突触项 | 直接表达“候选是否为逆动作”的交互，可扩展对象×动作关系 | 二阶状态/参数成本可能抵消速度收益 |
| R3 closed-form ridge readout | established direction | 固定SNN reservoir，以Kronecker/event feature一次求解relation readout | 训练从迭代反传降为单次线性求解 | 固定特征可能只记住已见action pair |
| R4 adaptive multiscale spike | cross-domain analogy | learnable decay/threshold与快慢trace共同训练 | 改善短prompt绑定并保留长时世界状态 | 增加优化不稳定性，不能单独解决二元交互 |
| R5 pair-frequency balanced continuation | established direction | 按16种train triples均衡采样，再回到真实分布评估 | 针对反复出错的低频novel pair，改动最小 | 可能掩盖结构性表达不足 |
| R6 native fused constant scan | established engineering direction | fused short-sequence dispatch与1/2/4/8/16线程扫描 | 直接处理RA0相对LSTM仍慢59%的工程门 | 只提速，不会修复当前质量差距 |

**推荐与顺序：**先执行 R1，因为它既是最小优化变量，又直接使用CPU/GPU batch并行；若ANN/SNN质量仍失败，执行 R2 与 R3 区分“需要可学习二阶动力学”还是“闭式读出已足够”；质量通过但速度失败才执行 R6。R4/R5保留为对照，不能替代真实closed-loop最终门。

### 冻结设计与门

- 数据、32/4/4官方games、真实 `next_obs` membership labels、D32/state31、model seeds `{0,1,2}`、50 epochs和4 CPU线程完全沿用SG6；不重选seed、不改test。
- 每个step的 `<previous_room>/<novel_room>` 两例不可拆分，`batch_groups=16` 即32 examples/batch；96 train groups每epoch 6 batch updates，50 epochs共300 optimizer updates/model，但仍恰好暴露9,600 examples/model。
- loss只取LM head中两个relation logits做二分类CE；五模型完全同目标、同AdamW/clip/shuffle。报告batch p50、折算example p50、examples/s与总wall；推理仍用完整17-token prompt。
- **TASK/QUALITY** 沿用SG6：ANN每seed NLL/二分类loss改善>=.10，best ANN accuracy>=.98/step>=.95；RA0 accuracy>=best ANN-.02且>=.98、step>=.95、loss<=best ANN+.05，并与BPTT/AT1 gap过门。
- **SPEED**：RA0 example-equivalent p50对AT1/BPTT>=1.25x且<=LSTM，并要求RA0训练总wall<=LSTM；**RESPONSE**：RA0 p50/p95<=LSTM。
- **What if：**SNN并不需要更深反传，而需要让一个batch直接呈现世界状态delta的正负对，使eligibility/reverse-adjoint在稀疏关系通道上获得低方差梯度？

### 实现与正式结果

- 每batch含16个完整step groups/32 examples，标签严格16:16；每epoch 6 updates，50 epochs=`300` optimizer updates/model，仍精确暴露`9,600` examples。调度、batch tensor和batched reverse-adjoint回归测试均通过。
- 正式命令：`.venv-wsl/bin/python experiments/e3_sg7_paired_binary_batch.py --device cpu --threads 4 --output results/e3_scan/e3_sg7_paired_binary_batch.json`；wall=`39.88 s`，结果 SHA-256=`BB558AE7363F0D859BFB88094F07AED2F830CDB901FAC7420EE3E6A2FFB132BC`。

| model | binary NLL ↓ | accuracy ↑ | step ↑ | example p50 ms ↓ | train wall s ↓ | examples/s ↑ |
|---|---:|---:|---:|---:|---:|---:|
| SNN-BPTT | .6738 | .5972 | .3333 | .08540 | .8860 | 10,851 |
| SNN-AT1 | .6736 | .5556 | .3056 | .07148 | .7378 | 13,309 |
| **SNN-RA0** | .6727 | **.6111** | **.3611** | **.04892** | **.5084** | **18,895** |
| LSTM | .3696 | .8333 | .6667 | **.02820** | **.3249** | **29,942** |
| Transformer | **.0007** | **1.0000** | **1.0000** | .05657 | .5866 | 16,403 |

- **TASK PASS**：Transformer三seed accuracy/step均1.0，两ANN每seed binary NLL改善>.10；SG6中差2例的优化不稳定已由paired objective消除。
- **QUALITY FAIL**：RA0/BPTT/AT1 binary NLL差仅`.0009/.0010`，却均接近二分类随机熵；RA0 accuracy `.6111`、step `.3611`。这反证“只需低方差梯度”，支持现有一阶trace缺少二阶关系绑定。
- **并行训练是正面结果但 SPEED仍FAIL**：RA0 example p50较SG6 `1.1288→.04892 ms`（约23.1x），相对AT1/BPTT=`1.461x/1.745x`且比Transformer快；但仍比LSTM慢约73.5%，总wall `.5084>.3249 s`。
- **RESPONSE FAIL**：RA0完整17-token p50/p95约`.34/.49 ms`，LSTM约`.15/.23 ms`。batching不改变单流推理，这与预期一致。

**决定：**采用paired binary batching作为后续状态通道的默认训练方式，因为它同时稳定ANN task并带来数量级吞吐收益；不把它当作SNN质量解决方案。按上方SG8加入显式bilinear spike binding，并用closed-form ridge隔离“特征不足”与“迭代训练不足”。

---

## 2026-07-19：E3-SG6 结果 — compact TextWorld move state-delta（负面结果）

### surface energy 失败后的表征分解

SG5 把双option降为单candidate后，五模型train/test仍精确`.5`，说明小模型无法从随机房间surface学出 action×outcome 交互。下一步按既定路线显式分离**可预测动力学状态**与**不可稳定建模的surface realization**：对真实hard move step，从 episode 事实审计 target是否等于上一已观察房间，构造 `previous move + candidate move → <previous_room>/<novel_room>`。

- 标签由真实 `next_obs` 对 prior normalized observations 的成员关系生成，不由方向规则直接赋值；必须审计 counterfactual target恰为lag-1 previous room、factual target未在history出现。每step两个move action各一例，预期 counts=`192/24/24`、labels 1:1、step groups=`96/12/12`。
- prompt只保留 `<bos> previous move:<action><eos> candidate move:<action><eos> next room relation:`，去掉随机surface；这不是最终LLM生成，而是可组合世界状态delta。输出仍是token CE，模型仍为纯SNN/LSTM/Transformer统一LM wrapper。
- D32/state31五模型、50 epochs（9,600 updates/model）、3 seeds、B1、4线程、同AdamW/clip；参数spread目标<=3%（小词表使固定core差异占比上升，精确值在模型运行前审计）。
- **TASK**：LSTM/Transformer每seed NLL改善>=.10，至少一个ANN mean accuracy>=.98、step consistency>=.95。**QUALITY**：RA0 accuracy>=bestANN-.02且>=.98、step>=.95，NLL<=bestANN+.05，与BPTT/AT1 accuracy gap<=.02/NLL gap<=.05。
- **SPEED**：RA0对AT1/BPTT>=1.25x且<=LSTM；**RESPONSE**：完整compact prompt p50/p95<=LSTM。
- **What if：**真正适合实时SNN世界模型的基底不是背诵surface，而是事件化delta；若RA0在真实TextWorld方向反转上达到ANN质量且训练接近ANN速度，就可把 room relation、reward/done、inventory、exit delta逐通道组合，再用条件surface模块生成语言。
- 若质量过而速度失败，立即进入 native fused constant-scan/short-sequence dispatch；若ANN过而SNN失败，改 adaptive/multiscale spike；若全门通过，扩展多通道delta并接closed-loop candidate scoring。

### 正式前审计

- vocabulary size=`18`、fingerprint=`4c660544808a1553f1d54383c3be69a28c2377d3602f09bc9698a091fd5b5b0e`，五模型参数总量=`9,060/9,060/9,060/9,106/9,266`，spread=`2.261%<3%`。
- train/valid/test examples=`192/24/24`、step groups=`96/12/12`；每split标签严格1:1，prompt恒为17 tokens、held-out OOV=0。
- 真实关系审计零违规：96/12/12个 factual outcomes 均未在prior history出现，96/12/12个counterfactual outcomes均精确命中lag-1上一房间。2-epoch smoke DATA PASS，只验证runner，不移动正式门。

### 正式结果

- 命令：`.venv-wsl/bin/python experiments/e3_sg6_move_delta.py --device cpu --threads 4 --output results/e3_scan/e3_sg6_move_delta.json`；3 seeds × 5 models × 9,600 updates，wall=`234.23 s`，结果 SHA-256=`31FBF1305F77228F14E5E8EA2EAB1F745B8105C60156C35DBCF2E8CDD1FEBD24`。

| model | test NLL ↓ | accuracy ↑ | step consistency ↑ | update p50 ms ↓ | response p50/p95 ms ↓ |
|---|---:|---:|---:|---:|---:|
| SNN-BPTT | .7412 | .8194 | .6389 | 1.8276 | .41/.55 |
| SNN-AT1 | .8101 | .8056 | .6111 | 1.4806 | .36/.49 |
| **SNN-RA0** | 1.5550 | .7222 | .4444 | **1.1288** | .39/.53 |
| LSTM | .2805 | .8333 | .6667 | **.7093** | **.17/.23** |
| Transformer | **.0987** | **.9722** | **.9444** | 1.1547 | .31/.43 |

- **DATA PASS；TASK FAIL**：两种ANN每seed NLL都改善>.10，但best ANN accuracy `.9722<.98`、step `.9444<.95`，均只差2/72 examples；冻结门不下调。
- **QUALITY FAIL**：RA0 seed accuracy=`.5833/.75/.8333`，mean `.7222`；seed0把大多数reverse pair判成novel，导致NLL `2.9003`。BPTT/AT1也只到`.8194/.8056`，说明不只是reverse-adjoint近似误差。
- **SPEED/RESPONSE FAIL**：RA0对AT1/BPTT=`1.312x/1.619x`，但 `1.1288 ms` 比LSTM `.7093 ms`慢约59%；平均训练wall `12.02 s` vs `7.50 s`，17-token response也慢约2.3x。
- 所有12种test `(previous,candidate,label)` triples在train均有5–29次支持且无label歧义。Transformer错误仅seed0的 `north→west, novel` 两例；SNN错误主要集中低频novel pairs，而非未见组合。

**决定：overall FAIL，不把compact化本身当作SNN成功。** 保留“状态delta优于surface复制”的方向，但下一步先按上方SG7用paired two-logit batch同时测试优化信号与并行训练；若仍失败，再进入bilinear spike binding与closed-form readout。

---

## 2026-07-19：E3-SG5 结果 — hard move outcome compatibility energy（负面结果）

### SG4 对称塌缩后的最小改写

SG4 的 train/test accuracy都精确`.5`，target margin约`1e-7`，说明把同一candidate的两个长option交换后联合输出A/B，在当前小模型/全词表CE下形成了强对称塌缩；ANN同样失败，不能触发SNN associative-memory结论。SG5 保留同一hard factual-move vs reverse-move后果，但每次只输入**一个** candidate outcome，预测 `<compatible>` / `<incompatible>`：这是标准 energy-style action-outcome compatibility，可直接作为规划器候选打分。

- 每个hard step产生4例：factual action×{factual正例, reverse负例}，reverse action×{reverse正例, factual负例}；两类action与两个完整房间surface相同频次，label严格1:1，不存在action type、outcome identity或位置捷径。
- counts仍为`384/48/48`，candidate groups=`192/24/24`，每group一正一负；prompt=`full trajectory + current observation + candidate action + candidate next observation + compatibility:`，target单label，max<=448、held-out OOV<10%。
- D32/state31五模型、20 epochs、seeds/优化器/4线程与SG4相同。指标为forced/open accuracy、target margin、candidate-pair consistency（同action正负均判对）、step consistency（四项全对）、NLL、update与response p50/p95。
- **TASK**：两个ANN每seedNLL改善>=.10；至少一个ANN mean forced>=.90、candidate-pair consistency>=.80。**QUALITY**：RA0 forced>=bestANN-.03且>=.90、pair>=.80，NLL<=bestANN+.10，并与BPTT/AT1 accuracy gap<=.03、NLL gap<=.10。
- **SPEED/RESPONSE** 原样：RA0对AT1/BPTT>=1.25x且<=LSTM；full-prompt p50/p95<=LSTM。DATA要求exact counts、label balance、每group正负完整、不同outcome、SHA链与OOV/长度全过。
- **What if：**trace SNN并非缺少动作因果状态，而是SG4要求在一个hidden中同时保留并比较两段长surface；把世界模型写成可组合能量 `E(history, action, candidate_next)` 后，RA0能否在稀疏单label监督下达到ANN精度和实时速度？
- 若ANN过门而SNN失败，进入spiking associative memory；若五模型仍随机，说明文本surface不是合适的最小因果表征，转显式state-delta；若全门通过，下一步用energy排序驱动真实closed-loop候选选择。

### 正式前审计

- vocabulary size=`300`、fingerprint=`43dbe84bfb295e0168bf166102e9fd1f035d280d2c6b6fe202887732a311025c`；prompt max train/valid/test=`306/301/303`，valid/test OOV=`.68%/1.67%`。
- counts=`384/48/48`、candidate groups=`192/24/24`、step groups=`96/12/12`；test compatible/incompatible=`24/24`，每candidate一正一负、每step四组合完整，DATA PASS。
- 2-epoch smoke五模型仍forced/open `.5`、pair/step consistency=0；只证明无显式泄漏和runner贯通，不改变20-epoch门。

### 正式结果

- 正式20 epochs wall time `253.2 s`；结果 SHA-256 `F081E8CEDFFD79E82504B4500ED415E709D40DA0100A5649E39E131702898300`。
- 五模型 train/test forced accuracy均`.5`、candidate-pair/step consistency均0；mean NLL BPTT/AT1/RA0/LSTM/Transformer=`.7049/.7042/.7062/.6962/.7014`，margin约0。ANN TASK FAIL，surface energy没有学出交互。
- RA0 update `1.3352 ms`，对AT1/BPTT=`1.353x/1.871x`，仍慢于LSTM `1.1447 ms`；response同样失败。DATA PASS，其余门与overall FAIL。

**决定：**停止对原始房间surface做label包装，不进入SNN结构归因；按上方SG6抽取由真实历史验证的 compact move delta，先建立有效的动作动力学与工程速度基线。

---

## 2026-07-19：E3-SG4 结果 — hard move-pair counterfactual ranking（负面结果）

### 从不可学surface复制到最小动作因果门

SG0–SG3 已依次排除 target缺失、4-game样本不足和D32容量不足；即使D64 Transformer的edit升到`.6298`，ANN/SNN move room仍只有`.02–.06`。继续堆数据/容量没有新的因果信息。下一任务不退回 inventory shortcut，而在每个 factual move 与同一步真实 counterfactual reverse move 之间构造**同类型双房间候选**：输入完整history、candidate move action和两个真实 next observations，预测 `<option_a>` 或 `<option_b>`。

- 只选择 factual action与一个counterfactual action都为move的步骤；两个option都是完整真实房间文本，候选action分别取两条move，每个候选再交换A/B顺序。因此每个world step产生4例，label与action/option位置严格平衡；模型不能靠“move选房间、inventory选carrying”过门。
- train/valid/test预期 examples=`384/48/48`（32/4/4 games × 每game 3个hard steps ×4），semantic candidate groups=`192/24/24`；每group两种option order。target是一个label token，prompt保留full factual trajectory及两段完整surface，max<=512；train-only vocabulary，held-out prompt OOV<10%。
- 非神经 position/random baseline固定`.5`；inverse-history oracle可达1.0，仅证明可识别性。神经模型必须在交换顺序后选择同一个semantic outcome，报告 forced-choice accuracy、open-vocab label accuracy、margin、swap consistency、NLL、update与整prompt response p50/p95。
- 模型回到工程基线 `D32/state31`：BPTT/AT1/RA0/LSTM/1-layer Transformer，参数spread<=2%、seeds `{0,1,2}`、20 epochs（384×20=7,680 updates/model）、B1、4 CPU threads、同shuffle/AdamW/clip。唯一任务改变是生成→真实候选排序。
- **H-SG4-DATA**：完整官方SHA链、exact counts/groups、label与每action A/B平衡、option不同、每group swap完整、prompt<=512、OOV<10%。
- **H-SG4-TASK**：LSTM/Transformer每seed NLL改善>=.10；至少一个ANN mean forced accuracy>=.90且swap consistency>=.95。
- **H-SG4-QUALITY**：RA0每seed NLL改善>=.10；mean NLL<=最佳ANN+.10、与BPTT/AT1 gap<=.10；forced accuracy>=最佳ANN-.03且>=.90、与BPTT/AT1 gap<=.03；swap consistency>=.95。
- **H-SG4-SPEED**：RA0 update p50对AT1/BPTT>=1.25x且<=LSTM。**H-SG4-RESPONSE**：RA0 full-prompt p50/p95均<=LSTM。
- **What if：**gated-trace SNN无法逐token复制300-token surface，却能否把history压缩成“上一动作方向/上一房间”状态，并以单个稀疏label监督达到ANN级因果选择？若能，下一步再把ranking score作为纯SNN生成/规划的训练信号；若ANN过门而SNN失败，进入spiking associative memory。
- runner/产物固定为 `experiments/e3_sg4_move_pair_ranking.py` 与 `results/e3_scan/e3_sg4_move_pair_ranking.json`。

### 正式前审计与 smoke

- train-only vocabulary size=`301`、fingerprint=`dcf0d7070a97b93e33d39d6c792a97117f73b60e4fce9c790ee422def041e846`；prompt max train/valid/test=`347/356/358`，valid/test prompt OOV=`.64%/1.64%`。
- counts=`384/48/48`、semantic groups=`192/24/24`；每group恰有A/B两种顺序，test label=`24/24`，无相同option，DATA PASS。
- 2-epoch smoke 中五模型 forced/open accuracy均恰为`.5`、swap consistency=`0`：模型尚未学习且固定位置预测在swap后被正确惩罚，证明无position shortcut；只用于验证 runner，不改变20-epoch门。

### 正式结果

- 正式20 epochs wall time `245.9 s`；结果 SHA-256 `DACEE27E506CDFD8E58C35C61E9766EAC6965B7EEAF54C98AD6A6ECEAF0A5546`。
- 五模型 train与test forced/open accuracy均精确`.5`、swap/group consistency均`0`；mean test NLL为 BPTT/AT1/RA0/LSTM/Transformer=`.7086/.7062/.7050/.7000/.7034`，target margin绝对值约`1e-7`。这不是过拟合，而是未打破A/B比较对称。
- RA0 update `1.3463 ms`，对AT1/BPTT=`1.275x/1.858x`刚过相对门，但仍慢于LSTM `1.1515 ms`；response p50/p95也慢于LSTM。DATA PASS，TASK/QUALITY/SPEED/RESPONSE及overall FAIL。

**决定：**SG4 不作为候选因果能力证据；ANN task gate失败，因此不归因SNN。按上方SG5把双option比较改写为单candidate energy compatibility，保持同样hard move数据和单label稀疏监督。

---

## 2026-07-19：E3-SG3 结果 — D64/state63 history-retrieval capacity gate（混合/负面结果）

### SG2 后的最小结构判别

SG2 把 move-copy train examples 从16增至128后，teacher NLL大幅改善，但 ANN move room仍只有`.0625`、RA0 `.0833`。Transformer代码审计已确认 sinusoidal position、global causal attention、512 cache均覆盖最长373-token input；下一项最小变量是**全模型共同扩容**，而不是只给ANN加层或给SNN外挂copy head。

- 五模型统一从 `D32/state31` 扩到 `D64/state63`；Transformer仍1 layer/4 heads/MLP ratio2，LSTM hidden64，三种SNN core/embedding/output均64。SNN wrapper与core初值继续逐tensor共享；参数spread必须<=2%，否则实验INVALID。
- 数据、32/4/4 game seeds、full trajectory prompt、319词train-only vocabulary、25 epochs、B1、model seeds `{0,1,2}`、4 CPU threads、优化器与所有 DATA/TASK/QUALITY/SPEED/STREAM门全部沿用SG2。唯一自变量是公平容量。
- **What if：**history retrieval 的失败只是31维trace/32维hidden无法同时保留房间surface与动作链；翻倍到63/64后，RA0是否出现稳定的move room复制，并因更大矩阵更充分利用多核而缩小对fused LSTM的绝对速度差？
- 若ANN move room>=.75而RA0<.50，直接支持 spiking associative memory；若全部仍失败，则完整自由生成不是当前最小可学因果门，转 paired candidate ranking 后再逐步恢复生成；若质量过而速度失败，进入 native fused/batched constant scan。
- runner仍为泛化后的 `experiments/e3_sg1_history_generation.py`，正式产物固定 `results/e3_scan/e3_sg3_d64_history_generation.json`。

### 正式结果

- D64参数：三种SNN `54,065`、LSTM `54,143`、Transformer `54,463`，spread `.735%`。正式25 epochs wall time `342.6 s`；结果 SHA-256 `AF7D18B506A7CB033D883B7FEE2946DEB29A9A1662D937AD212F47A07B7D6F90`。

| model | test NLL ↓ | edit ↑ | move room acc ↑ | update p50 ms ↓ |
|---|---:|---:|---:|---:|
| SNN-BPTT | 1.1269 | .6128 | .0417 | 2.7818 |
| SNN-AT1 | 1.1440 | .5965 | .0417 | 3.9822 |
| **SNN-RA0** | 1.1309 | .6013 | **.0625** | **1.5005** |
| LSTM | **1.1059** | .5990 | .0208 | **1.3600** |
| Transformer | 1.7088 | **.6298** | .0417 | 1.5817 |

- **TASK/QUALITY FAIL**：Transformer edit已过 action-majority+.05，但最佳ANN move room仅`.0417<.75`；RA0 `.0625<.50`。D64没有形成稳定检索，不能靠edit过门。
- **SPEED FAIL**：RA0对AT1/BPTT=`2.654x/1.854x`且快于Transformer；对LSTM差距由D32的21.1%缩至约10.3%，但仍未达到绝对门。
- **STREAM FAIL**：RA0 token p50/p95继续快于LSTM；prefill `.629–.664 ms` vs LSTM `.413–.434 ms` 全seed失败。

**决定：overall FAIL；停止继续扩games、epochs或hidden。** 容量扩大改善了Transformer surface edit与RA0相对速度，却没有解决动作条件room identity。按上方SG4把同一真实后果改成双候选因果排序，先建立ANN可学、无action-type捷径的最小门。

---

## 2026-07-19：E3-SG2 结果 — scaled official history generation（混合/负面结果）

### 为什么先扩真实数据而不是立刻换动力学

SG1 已证明 target 完整存在于 history，却仍只有 RA0 seed2 的1/4 move room偶然正确；LSTM/Transformer也为0。控制代码复核确认 Transformer 使用 unbounded sinusoidal position、global causal attention 与512-token cache，prompt不超过301，不存在截断/无位置编码缺陷。更直接的实验变量是：SG1 train只有4个game、16个 move-copy examples，全部模型 train NLL已接近0，跨game检索规律却没有足够重复。

候选包括 D64容量、pointer/copy head、spiking associative memory、paired ranking、增加官方games。先选择**只增加真实官方games**，因为它不改变模型/目标/指标，能最干净地区分“样本不足”与“结构不能检索”。pointer head会提前引入ANN attention捷径；associative SNN应在ANN task gate有效后再比较。

**What if：**把相同的 lag-1 known-edge 规律从16个扩到128个独立 procedural games 后，RA0 trace state是否会像Transformer/LSTM一样学出跨世界的 action-reversal retrieval，同时保留长上下文的并行训练优势？

### 冻结数据、预算与门

- 官方 TextWorld 1.7.0 `tw-coin_collector --level 5`，新目录不覆盖旧语料；train seeds=`20260801..20260832`（32 games），valid=`20260833..20260836`（4），test=`20260837..20260840`（4），严格按game seed隔离。每步仍取最多2个 `Environment.copy()` 真实 counterfactual，预期 examples=`320/40/40`、pairs=`160/20/20`、move=`128/16/16`；生成后必须由 manifest/game/episode/event SHA 与 exact counts自证，否则不训练。
- prompt/target/normalization/history-rule与SG1完全相同；vocabulary只由新train history prompt+target构建。held-out move surface prior-history ratio必须100%，prompt<=384、target<=80、valid/test OOV<10%、test完整target overlap<=20%。
- 模型仍为 `D=32,state=31` 的 BPTT/AT1/RA0/LSTM/1-layer Transformer，参数spread<=2%；seeds `{0,1,2}`、CPU4 threads、B1、同shuffle、AdamW/clip不变。训练改为25 epochs，即每模型8,000个真实update，是SG1 4,000的2倍但不是按数据量机械维持100 epochs。
- TASK/QUALITY/SPEED/STREAM门原样沿用SG1，尤其 ANN mean move room>=.75、RA0>=.50；不因数据扩大降低门。若ANN过TASK而SNN失败，进入 spiking associative memory；若ANN仍失败，才预注册 D64全模型容量对照；若质量过而速度失败，转 native fused/batched scan。
- 数据 runner 使用 `experiments/e2_textworld_dataset.py`；训练 runner复用泛化后的 `experiments/e3_sg1_history_generation.py`，正式结果固定为 `results/e3_scan/e3_sg2_scaled_history_generation.json`。

### 正式前 DATA gate 修正（未发生模型更新）

40个冻结seed生成后，首次 smoke 在 provenance通过、模型构建前 fail-closed：新train有一个 target（含EOS）长度`71`，仅违反从旧4-game语料外推的`<=70`经验上界；prompt max=`307`、valid/test OOV=`1.00%/1.19%`、held-out move history=`32/32`及其余门均通过。为避免删除长样本或重选seed造成挑数据偏差，正式前把 SG2 target 上界改为`80`，仍不截断；旧SG1的70门与结果不回写。新 task vocabulary size=`319`、fingerprint=`31c085a5d5cb207adb1eec87076cf5356371876c419dad51ca04c82beba55c08`。

### 正式结果

- 官方数据生成 wall time `387.4 s`；32/4/4 episodes 全部 `won=True/return=1.0`，steps=`160/20/20`、counterfactuals=`320/40/40`。summary SHA-256 train/valid/test=`2cdec296.../7ad27867.../13083785...`，runner随后再次验证完整 game/manifest/episode/event SHA链。
- 正式命令为同一 runner 加冻结 seeds/counts、`--epochs 25 --threads 4 --seeds 0 1 2`；wall time `289.3 s`；结果 SHA-256 `0065E47EBF80861AA6CC918BDD645961998D9DA7DC22B722ED7DC5CC89D78354`。

| model | test NLL ↓ | edit ↑ | move room acc ↑ | update p50 ms ↓ |
|---|---:|---:|---:|---:|
| SNN-BPTT | 1.1222 | .6114 | .0625 | 2.2564 |
| SNN-AT1 | 1.1452 | **.6155** | .0208 | 3.4814 |
| **SNN-RA0** | 1.1177 | .5998 | **.0833** | **1.2552** |
| LSTM | **1.0661** | .6012 | .0625 | **1.0368** |
| Transformer | 1.6055 | .6058 | 0 | 1.4372 |

- **DATA PASS**：vocab319、prompt<=307、target<=71/修正门80、valid/test OOV=`1.00%/1.19%`、held-out move history=`32/32`。
- **TASK FAIL**：最佳ANN edit `.6058` 刚低于 action-majority `.5580+.05=.6080`，更关键的 move room `.0625<.75`。
- **QUALITY FAIL**：RA0 NLL与BPTT/AT1 gap仅`.0045/.0275`，edit gap也过门，paired=`1.0`；但 edit `.5998<.6080`、move room `.0833<.50`。room hits只在少数特定test examples/seeds出现，不稳定。
- **SPEED FAIL**：RA0对AT1/BPTT=`2.774x/1.798x`且快于Transformer，但 `1.2552 ms` 比 LSTM `1.0368 ms` 慢约21.1%。
- **STREAM FAIL**：RA0 token p50/p95持续快于LSTM；prefill三seed `.541–.585 ms` 均慢于LSTM `.384–.412 ms`。

**决定：overall FAIL。** 数据扩大证明 RA0/BPTT/AT1 的 teacher质量与LSTM接近，并把 OOV降到约1%，但没有形成动作条件的历史surface复制；因此不继续加games或epochs。按上方SG3只做一次公平D64容量门，之后必须改变检索机制或任务形式。

---

## 2026-07-19：E3-SG1 结果 — history-conditioned known-edge generation（混合/负面结果）

### SG0 失败后的可识别性审计

SG0 的全部模型 room accuracy=0，但这可能来自 prompt 缺失世界历史，而非模型无法学习动作动力学。新增只读 runner `experiments/e3_sg1_history_identifiability.py`，逐条检查 counterfactual target room/surface 是否存在于 current observation 或同一真实 episode 的先前 factual observations；不读取游戏源码、未来 factual transition 或 counterfactual target 以外的信息。

- 命令：`.venv-wsl/bin/python experiments/e3_sg1_history_identifiability.py --output results/e3_scan/e3_sg1_history_identifiability.json`；产物 SHA-256 `F2AE1D07817D1828D5DC4C5700CCDA28E81AAEEB2AD05AE71A5E8FDEB8478A14`。
- train/valid/test counterfactual move 数为 `16/4/4`。target room 在 current observation 的可见数为 `0/0/0`；target room 与**完整规范化 surface**在 prior history 的出现数均为 `16/4/4`，且全部 history lag=`1`。held-out 合计即 current `0/8`、prior history `8/8`。
- look target 也全部等于 current observation（`4/1/1`）。因此 SG0 single-observation identifiability FAIL，而 history-conditioned route PASS；这不是扩大训练集能单独修复的随机方差。

### 路线选择与 What-if

SG0 已比较 history generation、paired ranking、state delta、data scaling、byte representation、adaptive spike 与 native scan。审计把首选收敛为 **full factual trajectory generation**：它仍生成完整自然语言，不把任务降成分类；目标 move surface 已由历史给出，但模型必须根据 factual action chain 与 candidate reverse action检索正确 observation。

**What if：**RA0 的并行正/反 scan 在 284–301 token prompt 上不仅扩大相对 AT1/BPTT 的训练优势，还能让 trace SNN 学会“动作反转→检索上一世界状态”；如果 Transformer 能复制而 SNN 不能，失败将直接支持 spiking associative memory / event-addressed retrieval，而不是继续调 optimizer。

### 冻结数据与任务

- prompt：`<bos> trajectory:`，依次加入每个 prior factual `observation:<text><eos> action:<actual><eos>`，再加入 `current observation:<text><eos> candidate action:<cf><eos> next observation:`；target仍为完整规范化 counterfactual `next_obs+<eos>`。只包含 candidate 时刻之前的 factual history，无未来或 target 泄漏。
- normalized train-only vocabulary size=`186`、fingerprint=`2480444d34a970d03f8e0b1c59643b432012769f9362d9eda402943ec827c314`；valid/test target OOV仍为`7.69%/6.59%`。prompt max train/valid/test=`301/283/284`，input max test=`350`，target max<=67；held-out move target surface in history=`100%`。
- 非神经诊断：action-majority edit `.6116`；确定性 history-rule（move取上一 observation、look取current、其余取train action-majority）edit `.9667`、exact `.9`、move room accuracy `1.0`。它是可识别性上界/机制诊断，不要求神经模型击败手写 oracle。
- 模型与公平条件延续 SG0：SNN-BPTT、SNN-AT1、SNN-RA0、LSTM、1-layer Transformer；`D=32,state=31`、参数spread<=2%、100 epochs、B1、同seed shuffle、AdamW `1e-3/wd=.01`、clip1、seeds `{0,1,2}`、CPU4 threads、完整 target query、无截断。

### 冻结门

- **H-SG1-DATA**：原 manifest/game/event SHA通过；counts/pairs=`40/10/10`与`20/5/5`；prompt<=384、target<=70、valid/test target OOV<10%、format-only=0、test target overlap<=20%；held-out move>=8且 target surface prior-history ratio=`100%`。否则 INVALID。
- **H-SG1-TASK**：LSTM/Transformer每seed teacher NLL改善>=`.10`；至少一个ANN的 mean edit>=action-majority+.05，并且 mean move room accuracy>=`.75`。它必须证明模型实际使用历史，而不是再次靠 inventory 过门。
- **H-SG1-QUALITY**：RA0每seed NLL改善>=`.10`；mean NLL<=最佳ANN+.25，与BPTT/AT1 gap各<=.10；edit>=最佳ANN-.10、与BPTT/AT1 gap各<=.05且>=action-majority+.05；move room accuracy>=最佳ANN-.25且>=.50；paired sensitivity>=.50。
- **H-SG1-SPEED**：RA0 update p50 对AT1/BPTT各>=1.25x且绝对<=LSTM。
- **H-SG1-STREAM**：RA0 greedy token p50/p95与full-trajectory prefill p50每seed均<=LSTM；state bytes单列。
- 2-epoch smoke 仅验证全链路：DATA PASS；RA0 update `1.20 ms` vs BPTT `2.14`、AT1 `4.89`、LSTM `1.05`，prefill `.46` vs LSTM `.62`；所有模型 move room仍为0，不能提前判质量。正式 runner/产物固定为 `experiments/e3_sg1_history_generation.py` 与 `results/e3_scan/e3_sg1_history_generation.json`。

### 正式结果与判门

- 正式命令：`.venv-wsl/bin/python experiments/e3_sg1_history_generation.py --device cpu --threads 4 --seeds 0 1 2 --epochs 100 --output results/e3_scan/e3_sg1_history_generation.json`；wall time `155.8 s`；SHA-256 `E17E6BB2FBB7AECE276025C4D0D3DD0D4294AA1327470F3258292CBB9A6E6FD6`。

| model | test NLL ↓ | edit ↑ | move room acc ↑ | update p50 ms ↓ |
|---|---:|---:|---:|---:|
| SNN-BPTT | **2.7600** | **.5956** | 0 | 2.2115 |
| SNN-AT1 | 2.7772 | .5700 | 0 | 4.6073 |
| **SNN-RA0** | 2.8198 | .5663 | **.0833** | **1.2239** |
| LSTM | 2.8427 | .5975 | 0 | **.9846** |
| Transformer | 4.1869 | .4488 | 0 | 1.4304 |

- **DATA PASS**：history identifiability与长度/OOV全过。
- **TASK FAIL**：最佳ANN edit `.5975` 低于 `.6116+.05=.6616`，best ANN move room=`0<.75`；history存在不等于14K参数、16个move样本能学会检索。
- **QUALITY FAIL**：RA0 NLL对BPTT/AT1 gap `.0597/.0425` 均过门、paired=`1.0`，但 edit `.5663` 未过非神经下限，move room `.0833<.50`。唯一正确room来自 seed2 对 `Cookhouse` 的1例，且其余surface仍错误，不能视为稳定机制。
- **SPEED FAIL**：RA0对AT1/BPTT=`3.764x/1.807x`，也快于Transformer，但 `1.2239 ms` 比 LSTM `.9846 ms` 慢约24.3%。
- **STREAM FAIL**：RA0 token p50/p95三seed均快于LSTM；full-history prefill seed0 `.492<.626 ms`，seed1/2 `.532/.553 > .341/.397 ms`。

**决定：overall FAIL，不进入闭环。** SG1 排除了“目标不在输入”的SG0混淆，却暴露了第二个独立问题：极小数据下五模型都记忆训练文本而不学习历史检索。保留 RA0 的相对SNN加速和单例 room hit，但不把它解释成 associative memory 成功；按上方SG2预注册先扩官方真实games。

---

## 2026-07-19：E3-SG0 结果 — action-conditioned counterfactual sequence generation（混合/负面结果）

### 从 sparse-token 非劣到连续世界输出

RA0 已在 TW0 的 K16 teacher-forced token 上同时超过 AT1/BPTT/LSTM 训练速度，但这仍可能只说明“稀疏监督适合 reverse adjoint”。下一任务不立即跳到更大数据，而是用同一批真实 TextWorld transition 构造连续反事实生成：给定当前 observation 与**未实际执行的 action**，模型必须生成该 action 的 `next_obs` 完整 token 序列。它直接检验 action-conditioned world response、连续 K、free-running exposure error 与 paired action sensitivity。

本地原始 `episodes.jsonl` 已审计：train/valid/test 按 game seed 隔离，分别有 `40/10/10` 个 counterfactual transition；action type 为 train `{move:16, inventory:16, look:4, examine:4}`，valid/test 各 `{move:4, inventory:4, look:1, examine:1}`。规范化规则只删除空行、`>` HUD/status 行，并在 observation 存在 `-= Room =-` 时删除其之前的启动 logo/goal；房间、出口、物体、inventory 与终局文本保留。

按最终 compact prompt 序列化重新审计后，word-token context train/valid/test 最大 `76/76/78`、mean `56.2/58.5/58.7`，target（不含 EOS）最大 `64/64/66`、mean `24.3/27.3/25.8`；无空 target、无 format-only target。train target 仅21种，valid/test各6种，但 valid/test 各只有1个完整 target 与 train 重复；规范化 train-only task vocabulary 的 target OOV ratio为 `0%/7.69%/6.59%`，低于10%但必须单列，free generation 按编码后的 `<unk>` target 公平比较。此前 `73/73/75` 是未加入 BOS/EOS 的自然文本预审计值，不用于正式门。

### 任务路线比较

| 路线 / epistemic label | 核心监督 | 价值 | 最小决定实验 | 主要风险 |
|---|---|---|---|---|
| **规范化 word-token counterfactual generation** / New task composition | prompt=`observation+candidate action`，连续生成完整 `next_obs+EOS` | 直接测语言世界响应；T<=142 可复用现有模型 | teacher NLL + greedy edit/LCS/feature/paired sensitivity | 40 train examples小；6% OOV |
| normalized UTF-8 byte generation / Established representation | 无 OOV 的 byte autoregression | 最严格 exact text | 同样60 example、target最长318 bytes | 序列4–5倍长，先混淆表示与动力学 |
| raw JSON counterfactual-line generation / Baseline shortcut | 生成 action/keys/HUD/next_obs 全行 | 最接近现有 event stream | raw token NLL/exact | JSON与HUD格式主导，不接受为首选 |
| paired counterfactual discrimination / Established contrastive task | 在两个 candidate next states 中选正确者 | 数据效率高、可测 action sensitivity | pair accuracy | 不是生成，不能证明实时响应 |
| latent next-state prediction + decoder / Established world-model route | 预测 observation latent 再解码 | 可能提高长文本质量 | latent retrieval + reconstruction | 引入 ANN decoder/额外目标，混淆纯 SNN 主线 |
| actual+counterfactual multitask generation / Established augmentation | 同时生成实际与反事实 next observation | 训练样本约增50%，结构更广 | channel-conditioned generation | 改变当前单变量任务，失败难归因 |
| online TextWorld closed-loop rollout / Ultimate evaluation | 模型生成状态并驱动下一 action | 与最终世界模型最接近 | success/consistency over episodes | 当前先需证明单步 free generation 有效 |
| HomeGrid multimodal dynamics / Cross-domain transfer | image/symbol+action→next frame/state | 多模态关键门 | visual latent rollout | 应在语言生成门之后独立隔离 encoder |

选择第一条；byte generation 是若 OOV 成为主要误差时的下一表示对照，paired discrimination只作诊断，不替代生成。**What if：**RA0 的 reverse scan 成本几乎与 K 无关，而 AT1 eligibility 随 K×parameter 增长；当 K 从16变成每个 target 的5–67个连续位置时，RA0 是否会获得更大的训练优势，同时因 exact gradient 保持 free-running generation 与 BPTT 非劣？

### 冻结数据、训练与指标

- 每例 prompt tokens：`<bos>` + `observation:` + 规范化 observation + `<eos>` + `action:` + candidate action + `<eos>` + `next observation:`；target 为规范化 next_obs 的 word tokens + `<eos>`。compact semantic markers 避免 JSON/channel 标点主导，且全部来自 train vocabulary。输入为 `prompt+target[:-1]`，query 是预测全部 target 的连续 causal positions；不截断 target，不跨样本传 state。
- tokenizer沿用 manifest/SHA 已验证的 TextWorld event corpus；vocabulary 则由**规范化后的 train prompt+target**重新确定性构建，valid/test 不贡献 token identity 或 frequency。split仍由原 game seed决定。审计必须记录 raw source hash、task-vocabulary fingerprint、example/pair/action分布、context/target长度、OOV、完整 target overlap、copy-observation 与 train action-majority baseline。
- 模型固定 `D=32,state=31`：BPTT gated trace、AT1 forward eligibility、RA0 parallel reverse adjoint、LSTM、1-layer Transformer；wrapper/三种 SNN初值共享，参数spread<=2%。训练 `100 epochs`、每例B1、每 epoch 使用按 seed 预生成的相同 shuffle schedule；AdamW `lr=1e-3,wd=.01`、clip1、foreach clip+fused optimizer、seeds `{0,1,2}`、CPU4 threads。
- teacher-forced 报 target NLL/PPL/top1、action-type macro；greedy 从完整 prompt prefill 后生成到 EOS或80 tokens，报 exact、token edit similarity、LCS-F1、world-feature F1（room/direction/coin/inventory/end）、paired action diversity，并保存每例 target/prediction token。另报 prompt prefill和逐 token p50/p95。

### 冻结门

- **H-SG0-DATA**：manifest/game/event SHA通过；counts=`40/10/10`、pair counts=`20/5/5`；context<=80、target<=70；valid/test target OOV<10%、format-only=0、test完整 target与train overlap<=20%。否则任务 INVALID。
- **H-SG0-TASK**：LSTM/Transformer 每 seed test teacher NLL 均比未训练下降>=`.10`；至少一个 ANN 的三-seed mean greedy edit similarity 必须比 `max(copy_observation, action_majority)` 高>=`.05`，否则生成预算/任务无效。
- **H-SG0-QUALITY**：RA0 每 seed NLL改善>=`.10`；mean NLL<=最佳ANN+`.25`，与BPTT/AT1 mean gap各<=`.10`；RA0 mean edit similarity>=最佳ANN-.10、与BPTT/AT1差各<=`.05`，且比两种非神经 baseline最佳值高>=`.05`。paired target不同的样本中，RA0生成必须至少50%随action改变。
- **H-SG0-SPEED**：真实 update p50（prompt+连续target、CE/backward/clip/fused AdamW）RA0 比AT1与BPTT均快>=`1.25x`且<=LSTM；同时报告 K 与 total T 分布。
- **H-SG0-STREAM**：RA0 greedy token p50/p95均<=LSTM；prompt prefill p50<=LSTM。Transformer cache与完整 generation state bytes单列。
- 全门通过才进入 online closed-loop；质量失败转 byte/更强 adaptive-spike dynamics，短序列速度失败转 native fused constant-scan/batched variable-query kernel。runner/产物固定为 `experiments/e3_sg0_counterfactual_generation.py`、`results/e3_scan/e3_sg0_counterfactual_generation.json`。

### 正式前任务仪器修正

首个2-epoch smoke 发现，直接复用 raw event vocabulary 会把 JSON 中的转义换行与相邻自然语言粘成 `nYou've` 一类 token，而 SG0 规范化后实际 token 是 `You've`；compact marker `next` 也因此成为40次伪 OOV。这不是模型误差，而是表示仪器不一致。该 smoke 只用于暴露 runner 问题，不参与任何正式判定或结果比较。

正式运行前已在不改变 tokenizer、split、prompt、target、模型与冻结门的前提下修正为 normalized-train-only task vocabulary：只扫描40个 train examples 的 prompt+target 建表，得到 size `183`、fingerprint `dd3e51c6deb5b1aede57b71b9d9745f390a301ba4b7ccd3a66a237f066717364`；train prompt/target unknown均为0，valid/test target unknown为 `21/273` 与 `17/258`（`7.69%/6.59%`）。修正后2-epoch全链路 smoke 的 DATA gate PASS；正式100-epoch结果尚未运行，继续保持“进行中”。

### 正式结果

- 正式命令：`.venv-wsl/bin/python experiments/e3_sg0_counterfactual_generation.py --device cpu --threads 4 --seeds 0 1 2 --epochs 100 --output results/e3_scan/e3_sg0_counterfactual_generation.json`；wall time `143.8 s`；产物 SHA-256 `734A095B984AAC495A06329565B59783116EEC421942640E269AAB60B0EFF05D`。
- 环境：commit `1ae35d22bdeb9ec4011a49446fafdf59fc6d3c8e`、PyTorch `2.13.0+cpu`、Ryzen 9 7950X、32 logical CPUs、4 intra-op threads、MKLDNN enabled；CUDA unavailable，因此只支持 CPU 多核结论。
- 五模型参数为 SNN `14,505`、LSTM `14,551`、Transformer `14,711`，spread `1.415%`，通过2%公平门；三种 SNN wrapper/初值共享。

三 seed mean：

| model | test teacher NLL ↓ | greedy edit ↑ | update p50 ms ↓ |
|---|---:|---:|---:|
| SNN-BPTT | **2.7514** | .6324 | 1.8419 |
| SNN-AT1 | 2.8801 | **.6465** | 4.4285 |
| **SNN-RA0** | 2.8976 | .6257 | **1.1322** |
| LSTM | 2.8544 | .6205 | **.7971** |
| Transformer | 3.8830 | .4510 | 1.0669 |

| 冻结门 | 判定 | 直接证据 |
|---|---|---|
| H-SG0-DATA | **PASS** | counts/pairs/长度/overlap均过门；valid/test target OOV=`7.69%/6.59%` |
| H-SG0-TASK | **FAIL** | 两个ANN每seed teacher NLL均改善>.10，但最佳ANN edit `.6205` 未达到强非神经 baseline `.6116 + .05 = .6616` |
| H-SG0-QUALITY | **FAIL** | RA0 对最佳ANN NLL仅差`.0432`且 edit 高`.0051`，但对BPTT NLL gap `.1462>.10`，并且只比非神经 baseline高`.0141<.05`；paired sensitivity=`1.0` |
| H-SG0-SPEED | **FAIL** | RA0 对AT1/BPTT为 `3.911x/1.627x`，但 `1.1322 ms` 比 LSTM `.7971 ms` 慢约`1.42x` |
| H-SG0-STREAM | **FAIL** | RA0 token p50/p95 三seed均快于LSTM；prefill仅seed0通过，seed1/2为`.464/.464 ms` vs LSTM `.197/.202 ms` |

**overall FAIL。** 这是严格按预注册门判定，没有因三种 SNN 的 edit 均值高于 LSTM 而放宽任务有效性，也没有因 RA0 相对其他 SNN 很快而放宽 ANN 绝对速度门。

### 观察、解释与任务可识别性

- **观察：**全部模型 train NLL 已到约`.01–.10`，但 test NLL仍为`2.64–4.38`；所有模型 test room accuracy 都是`0`。除 Transformer seed2 外，greedy exact 都固定为`.4`，恰好对应4/10个恒定的 `inventory -> You are carrying nothing.` 样本。action-majority baseline也靠 inventory 得到 exact `.4`、edit `.6116`、paired sensitivity `1.0`。
- **观察：**RA0 seed0的移动输出能生成语法与出口结构合理的完整房间描述，却把 `Cookhouse/Bedchamber/...` 预测成训练中其他房间；这不是 EOS 或句法崩溃，而是目标世界身份错误。
- **解释：**当前 prompt 只有单个 current observation 与 candidate action。对 seed-disjoint 的未见 TextWorld 游戏，出口通常不暴露目标房间名称/描述；首次穿过一条边时，完整 next room surface form 并不能从输入唯一决定。40条训练样本又加剧记忆模板与过拟合，但单纯扩数据或增加 epoch 仍不能消除这部分条件熵。
- **边界：**SG0 因 ANN 也未通过 H-TASK，应判“当前单观测 free-generation task 不足以决定模型优劣”，不能把 overall FAIL 解释成 SNN 动力学失败。反过来，RA0 质量接近/略高于ANN也不能宣称世界模型成功，因为 room identity 为0。

### 失败后的路线比较与决定

| 下一路线 / epistemic label | 修复对象 | 最小决定实验 | 主要风险 |
|---|---|---|---|
| **history-conditioned known-edge generation** / New task composition | 给完整已观察轨迹，只在目标房间/状态已被历史识别的边上测完整生成 | 先审计 target room 是否在历史出现，再做同五模型 generation | 需要探索/回访轨迹，现有 walkthrough 可能覆盖不足 |
| paired counterfactual candidate ranking / Established contrastive task | 给两个真实候选 next state，测 action 与后果匹配 | pair accuracy + calibration + latency | 不是自由生成，只能作中间因果门 |
| predictable state-delta generation / Established world-model decomposition | 只生成 reward/done/inventory/room-change/exit delta，把不可知 surface 单列 | delta exact/F1 + surface conditional NLL | 结构化目标可能弱化 LLM 生成要求 |
| larger actual+counterfactual corpus / Established data scaling | 更多官方 game seeds，并加入 factual action→next_obs | learning curve与held-out动态macro | 若仍是首次未知房间，扩数据不能修复不可识别性 |
| normalized UTF-8 byte generation / Established representation | 去掉6–8% OOV与 `<unk>` | 同任务 byte NLL/edit | 序列更长，且不解决未知房间身份 |
| multiscale adaptive-spike dynamics / Established SNN direction | 增强时间尺度与条件记忆 | 在有效任务上比较 ALIF/multi-decay RA0 | 当前任务无效时先做会混淆归因 |
| native fused/batched constant scan / Systems specialization | 消除 RA0 短序列 prefill/update dispatch | 同模型同权重 kernel benchmark | 只能修 SPEED，不能修 TASK |

**What if：**把 next observation 显式分解为“由历史与动作决定的 state delta”和“首次发现时具有条件不确定性的 surface realization”，同一个纯 SNN 是否能对前者做严格实时确定预测、对后者维护分布，而不被迫背诵随机房间文案？

**决定：**保留 RA0 exact reverse adjoint 为 SNN 默认训练数学；SG0 不进入 closed loop。先执行 **SG1 task-identifiability audit**，量化 move target room 对 current prompt、episode history 与 train vocabulary 的可见性；若存在足够 known-edge 样本，首选 history-conditioned generation，否则先用 paired ranking + predictable delta 建立有效因果门。同时把 native fused scan 保留为独立 SPEED 路线，但不让系统优化掩盖任务无效。

---

## 2026-07-19：E3-RA0 结果 — exact parallel reverse adjoint 在真实任务训推均超过 LSTM（关键正面结果）

### 从“梯度正确”到“并行形式”的路线收敛

7月18日的 RA0 预注册选择 exact reverse adjoint，但首个实现仍有两处与目标不一致：forward 先算 AT1 的 query/decay eligibility、再算完整 trace；backward 用 Python 按 K 个冲激逐段填充伴随场。两者数学正确，却没有把常系数递推充分映射为并行 tensor scan。正式运行前按同一冻结任务与门做 smoke，不改变 K、模型、数据或质量门。

| 路线 / 标签 | 变化 | smoke 结论 | 决定 |
|---|---|---|---|
| 原型：双 E/I scan + sparse segment adjoint / exact prototype | E/I 分开 forward；K 段反向闭式 | 梯度通过，但有重复 trace/eligibility 与 Python 分段 dispatch | 淘汰 |
| 合并 E/I forward scan / algebraic fusion | 把两套同构递推拼成 `2S` 一次 scan | 减少一半 scan 循环/拼接，梯度不变 | 采用 |
| 常系数 bias-only forward scan / exact specialization | 利用 block coefficient=`lambda^offset`，只扫描 bias | 独立算子 p50 `0.170 ms` vs 通用 affine `0.233 ms` | 采用 |
| sparse impulses + parallel reverse scan / exact specialization | K 个 learning signal 写入稀疏冲激场，翻转时间后做常系数 prefix scan | 独立算子 p50 `0.175 ms` vs segment `0.251 ms` | **采用为 RA0** |
| 已审计 query/state unchecked hot path / systems invariant | 公共 API 保留完整验证；TW0 data audit 后的热循环不再 `.item()/torch.all` | 消除 CPU 开销与未来 CUDA host sync；AT1/RA0 同样使用 | 采用 |
| foreach clip + fused AdamW / fair optimizer control | 五模型统一使用同一 foreach/fused optimizer 路径 | 降低多参数 tensor dispatch，不改变优化方程 | 正式 runner 固定 |
| embedding-aware scatter / next systems specialization | custom backward 直接 scatter token gradient | profiler 显示 embedding backward 不是当前主瓶颈 | 暂缓；若正式速度失败再做 |

**What if：**既然正向 trace 与反向 adjoint 都是同一个常系数半群，只是 bias 与时间方向不同，是否应把“eligibility vs BPTT”重新表述为双向 prefix-scan 原语，使 CPU 多核与未来 GPU 都只需同一类 kernel？当前 smoke 支持这一表述，但 GPU 尚不可用，不能宣称 GPU 实测成功。

### 当前证据（仅 smoke，不改判正式门）

- 全部 gated-trace 单测仍通过；RA0 对 BPTT 的 input/初态/全部参数梯度保持原 `2e-5/1e-4` 门。
- 4线程、`T512/K16/input-grad` 的50次交错核心样本：RA0 p50 `1.188 ms`，LSTM `1.497`，AT1 `3.476`，BPTT `3.083`；RA0 相对三者分别为 `1.26x / 2.92x / 2.59x`（按 LSTM/AT1/BPTT 除 RA0）。
- 同线程一轮真实 TextWorld smoke：RA0 update p50 `1.459 ms`，LSTM `1.506`，AT1 `3.764`，BPTT `2.885`；三种 SNN 的 first/last loss 与 held-out NLL 保持浮点等价。首次在 trainable tied embedding + K16 + optimizer 的实际路径低于 LSTM，但只有单 seed/单 epoch，不能标 PASS。
- saved-storage smoke 仍过门：RA0/BPTT 为 T512 `17.17%`、T2048 `14.18%`；代价是随 T 线性增长（512→2048 为 `3.74x`），与 AT1 的常数 T 内存形成明确交换。

**决定：**冻结 `combined constant forward scan + parallel reverse scan + audited unchecked hot path + fair fused optimizer`，立即运行原预注册三 seed/20 epoch/1-4-16线程正式矩阵。门槛一项不放宽；若真实速度或质量失败，保留 negative result 并进入 embedding scatter/native fused scan，而不是减少 K 或冻结 embedding。

### 正式产物与等价性

- 正式命令：`.venv-wsl/bin/python experiments/e3_ra0_reverse_adjoint.py --output results/e3_scan/e3_ra0_reverse_adjoint.json`；SHA-256 `C4FBB3554B5C21A2994ED0E671FCE1647CB6742B31EA7B1DF9C902179A9A302B`。
- 环境为 PyTorch `2.13.0+cpu`、Ryzen 9 7950X、32 logical CPUs、MKLDNN enabled。宿主虽有 RX 7800 XT 且 WSL 有 `/dev/dxg/rocminfo`，但 ROCm 只枚举 CPU，`torch.cuda.is_available()==False`；本结果只证明 CPU 多核，不声称 GPU。
- 四个冻结 case 全过。新增 `(B,T,K,input-grad)=(1,512,16,on)` 的 forward 最大误差 `5.96e-7`、全梯度最大误差 `9.69e-8`；其余三 case forward<=`2.98e-7`、gradient<=`3.35e-8`。hard event、query/state、input/初态/全部参数均通过 `2e-5/1e-4`。**H-RA0-EQ PASS。**

input-gradient、K16 的 unique saved storage：

| T | BPTT | AT1 forward eligibility | RA0 reverse adjoint | RA0 / BPTT |
|---:|---:|---:|---:|---:|
| 512 | 3,660,616 B | **591,424 B** | 628,688 B | **17.17%** |
| 2048 | 16,567,112 B | **1,353,280 B** | 2,349,008 B | **14.18%** |

RA0 随 T512→2048 增长 `3.736x`，不具备 AT1 的 T 常数内存，但两档均低于 BPTT 的25%门；custom autograd nodes 为14 vs BPTT `265/313`。**H-RA0-MEM PASS。**

### 多核核心速度

K16、input gradient on 的 p50 ms：

| threads | T | BPTT | AT1 | RA0 | LSTM | RA0 vs AT1 / BPTT | gate |
|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | 512 | 3.188 | 3.310 | 1.265 | **1.215** | 2.62x / 2.52x | absolute FAIL |
| 1 | 2048 | 8.242 | 6.674 | **3.304** | 3.587 | 2.02x / 2.49x | PASS |
| 4 | 512 | 2.621 | 3.495 | **1.233** | 1.345 | 2.84x / 2.13x | PASS |
| 4 | 2048 | 6.165 | 5.827 | **2.111** | 4.788 | 2.76x / 2.92x | PASS |
| 16 | 512 | 5.068 | 7.299 | **2.317** | 2.707 | 3.15x / 2.19x | PASS |
| 16 | 2048 | 13.396 | 10.925 | **3.822** | 9.136 | 2.86x / 3.51x | PASS |

除单线程 T512 外五档通过；长序列上线程越多，RA0 对 fused LSTM 的优势越明显。**H-RA0-SPEED PASS。** 这不是减少监督换来的：K 固定16、input gradient 必须返回，参数与 forward 动力学未改。

### 三 seed 真实 TextWorld 质量与实际 update

五模型都使用同一 data/query/20 epochs、4 CPU threads、trainable tied embedding、foreach clip + fused AdamW。表内为 test sparse NLL / training update p50 ms：

| seed | BPTT | AT1 | RA0 | LSTM | Transformer |
|---:|---:|---:|---:|---:|---:|
| 0 | 2.720 / 2.945 | 2.720 / 4.168 | **2.720 / 1.549** | 2.554 / 1.600 | 3.756 / 4.498 |
| 1 | 2.547 / 2.947 | **2.454 / 3.869** | 2.578 / **1.493** | 2.446 / 1.535 | 3.623 / 4.372 |
| 2 | 2.628 / 2.904 | 2.684 / 3.678 | 2.684 / **1.433** | **2.399 / 1.517** | 3.617 / 4.391 |
| **mean** | 2.632 / 2.932 | 2.619 / 3.905 | **2.660 / 1.492** | **2.467 / 1.550** | 3.666 / 4.420 |

- RA0 每 seed 均从未训练 NLL 改善>=.10；mean 比最佳 ANN 高 `.1937`（门`.25`），与 AT1/BPTT mean gap=`.0413/.0287`（门`.10`）。浮点路径可像 AT1 一样产生不同 hard-event trajectory，但 held-out 功能保持非劣。**H-RA0-TW0 QUALITY PASS。**
- 实际 update 中 RA0 三 seed 都快于各自 LSTM；mean 对 AT1/BPTT 为 `2.618x/1.965x`，并以 `1.492 < 1.550 ms` 首次通过真实任务的 ANN 绝对训练门。**H-RA0-TW0 SPEED PASS。** 相对原 TW0 AT1 mean `4.032 ms`，同类 sparse-query SNN 训练瓶颈已从“慢于 BPTT/LSTM”翻转为最快。
- RA0 cached streaming 三 seed p50/p95 为 `.0880/.1427`、`.0886/.1308`、`.0917/.1262 ms`；LSTM 为 `.1099/.1579`、`.1096/.1757`、`.1137/.1603`，每 seed 两个分位都更快。**H-RA0-TW0 STREAM PASS。**
- full RA0 accuracy `45.8%/46.9%/41.7%`；spike-only `32.3%/39.6%/34.4%`，trace-only `34.4%/28.1%/22.9%`。full 始终更强，继续支持“二值 spike + 连续 trace 互补”的 event-driven trace SNN 边界，仍不能宣称完全 spike-coded。

### 结论与边界

**EQ/MEM/core SPEED 与 TextWorld DATA/TASK/QUALITY/SPEED/STREAM 全 PASS，RA0 overall PASS。** 采用 RA0 exact parallel reverse adjoint 作为当前 gated-trace SNN 的默认训练数学：同一常系数半群支持正向 trace scan 与反向 adjoint scan，既保留 strict binary event forward，也能把真实 trainable-embedding update 压到 LSTM 以下。

这不是最终目标完成：RA0 的 held-out NLL 仍未超过 LSTM，任务仍是 teacher-forced sparse token prediction，尚未验证多 token generation、rollout、closed loop 或多模态；GPU 也未可用。下一正式任务进入 **counterfactual sequence generation**，专门检验 K 从稀疏 token 扩展到连续生成窗口时 RA0 是否仍保持质量/训速；随后接 HomeGrid multimodal dynamics 与闭环 rollout。GPU 路线并行保留为 AMD ROCm/DirectML 可用性与 native constant-scan kernel，而不以 CPU 结果代替。

---

## 2026-07-18：E3-RA0 预注册 — input-gradient reverse adjoint for sparse event LM（进行中）

### TW0 后的训练加速路线

TW0 已证明 AT1 的最终功能质量足以进入真实语言世界模型任务，失败只在训练速度：冻结 embedding 的 register 上 forward eligibility 很快，但真实 LM 必须把梯度传回 trainable embedding；当前 backward 为恢复每个 `x_t` 梯度又做一次 full reverse affine scan，同时 K=16 还维护 `K×4S×D` 参数 eligibility，造成 `4.03 ms` 慢于 AT0-BPTT `3.12`。下一轮保持同一 forward/模型/数据，只改 exact reverse-mode 数学。

| 方向 / 标签 | 计算结构 | 时间/内存预期 | 主要风险 |
|---|---|---|---|
| **sparse-impulse reverse adjoint** / Speculative specialisation | query learning signal 作为 K 个冲激，分段闭式算 `p_t=L_t+lambda*p_{t+1}`；一次 contraction 得全部参数/input gradient | `O(T)` 保存与计算，不再 `O(KSD)` eligibility | 失去 core-only T 常数内存；需保存/recompute trace/event local factor |
| embedding-aware fused scatter / Systems specialisation | 不返回 dense `grad_x`，直接按 token id `index_add` 到 embedding weight | 省一次 `[T,D]` tensor 与 embedding backward | core 与 tokenizer/wrapper耦合，先证明 adjoint 再融合 |
| low-rank/diagonal eligibility / Established approximation | 压缩 K×参数 Jacobian | 仍可 online、常数 T | 近似会引入真实任务质量 gap，当前 exact reverse 已可得 |
| block checkpoint/recompute / Established engineering | 只存 chunk 边界，反向重算事件/trace | 内存可调 | TW0 本身已 chunk=512；重算可能更慢 |
| native C++/Triton contraction / Systems engineering | 融合 event derivative、adjoint、weight/embedding reduction | 最终吞吐潜力最高 | 当前 CPU/无 CUDA，先排除算法重复计算 |
| 冻结或预训练 embedding / Baseline shortcut | 恢复 AT1 core-only 路径 | 立即变快 | 真实 LM 质量仪器改变，不接受为本轮答案 |
| 减少 K / Objective tradeoff | K16→K4 | forward eligibility 更便宜 | 改正式监督与统计量，本轮禁止 |

选择 exact reverse adjoint；它不是一般 recurrent SNN 的免费解，而是 gated diagonal trace 的 reverse-mode 特例。若仍慢，再做 embedding scatter/native kernel，不降低 K、不冻结 embedding。

**What if：**真实世界模型的 K 个 loss 虽稀疏，但 input gradient 必然 dense；此时 forward-mode eligibility 的优势可能被 Jacobian 宽度抵消，而把 K 个 loss 先合成一个反向伴随场，再一次收缩全部 event projection/decay/input gradient，是否才是正确的 CPU/GPU 训练形式？

### 冻结数学与门

对 E/I population，query/final learning signal 仍为 AT1 的 `L_t`，其余时刻为 0：

`p_t = L_t + lambda*p_{t+1}`；

`dL/ddrive_t = (1-lambda)*p_t*[g_t*phi_c, c_t*phi_g]`；

`grad_W = sum_t outer(dL/ddrive_t,x_t)`，`grad_x=sum_rows dL/ddrive_t*W`；

`grad_decay_logit = sum_t p_t*dλ/dlogit*(h_{t-1}-v_t)`，`grad_h_init=lambda*p_0`。

K 个冲激按降序 query 分段；每个无冲激区间用 `lambda^distance` 向前填充，不构建 Hillis–Steele reverse graph。forward 保存 hard content/gate/write local factor、`h_{t-1}`、x 与静态 decay；默认 AT1 `forward_eligibility` 保留，新 mode 明确为 `reverse_adjoint`。

- **H-RA0-EQ**：相对普通 AT0-BPTT 覆盖 AT1 三 case，并新增 `(B,T,K,input_grad)=(1,512,16,on)`；query/state/hard event 与 input/初态/全部参数梯度满足 `2e-5/1e-4`。
- **H-RA0-MEM**：input-gradient、B1/D32/K16、T512/2048；unique saved bytes 均<=AT0-BPTT 的25%，允许随 T 线性增长并与 AT1 forward eligibility并列报告。
- **H-RA0-SPEED**：threads1/4/16、T512/2048、input grad on、K16；至少一档 RA0 比 AT1 forward eligibility快>=1.25x、比 AT0-BPTT快>=1.25x且 p50<=LSTM。
- **H-RA0-TW0**：完全复用 TW0 data/query/20 epochs/三 seed；RA0 每 seed test NLL 改善>=.10，mean NLL<=最佳ANN+.25，且相对 AT1/BPTT mean gap各<=.10。真实 update p50 必须比 AT1快>=1.25x、比AT0快>=1.25x且<=LSTM；streaming沿用同一core并重跑。
- 若 quality PASS/speed FAIL，进入 embedding-aware fused scatter/native kernel；quality FAIL 则 reverse adjoint实现失败或浮点路径对真实任务不稳。runner/产物：`experiments/e3_ra0_reverse_adjoint.py`、`results/e3_scan/e3_ra0_reverse_adjoint.json`。

---

## 2026-07-18：E3-TW0 结果 — 首个真实 action-conditioned LM 质量 PASS，训练 input-gradient 成为新瓶颈（关键混合结果）

### 数据与任务有效性

- 正式命令：`.venv-wsl/bin/python experiments/e3_tw0_sparse_event_lm.py --output results/e3_scan/e3_tw0_sparse_event_lm.json`；SHA-256 `4E029A3D0D2DB662BAD86B1D8D6E7BC377AB7628F5C93BAE3382FCC3F6102DEC`。
- 所有真实 TextWorld L5 manifest/game/event SHA 通过；无 synthetic fallback。512-token chunk 在 train/valid/test 形成 `384/96/96` 个 query，K/T≈`3.5%`，覆盖 114 个不同 token；格式/JSON token 比例 `31.25%`，低于70%无效线。**H-TW0-DATA PASS。**
- smoke 暴露 parallel scan chunk final trace 可能因 FP32 舍入略超 `[0,1]`；正式前只在已 detach 的 chunk 边界投影 `clamp[0,1]`，AT0/AT1共同使用。数学递推本就是 `[0,1]` 凸组合；chunk 内方程、query、门与正式数据未变。
- LSTM/Transformer 每 seed test NLL 都相对未训练值下降远超 .10，故 **H-TW0-TASK PASS**。

### 三 seed held-out 结果

| seed | AT0-BPTT NLL / acc | AT1 NLL / acc | LSTM NLL / acc | Transformer NLL / acc |
|---:|---:|---:|---:|---:|
| 0 | 2.735 / 50.0% | **2.723 / 49.0%** | 2.822 / 52.1% | 3.606 / 33.3% |
| 1 | 2.601 / 52.1% | 2.575 / 51.0% | **2.391 / 53.1%** | 3.706 / 30.2% |
| 2 | 2.709 / 52.1% | 2.698 / 52.1% | **2.511 / 53.1%** | 3.569 / 33.3% |
| **mean** | 2.682 | **2.666** | **2.575** | 3.627 |

- AT1 mean 只比最佳 ANN（LSTM）高 `.091`，小于 `.25` 门，并显著优于该小模型 Transformer；AT1/AT0 mean gap=`.0162`。四者每 seed 均从未训练 NLL 下降，故 **H-TW0-QUALITY PASS**。这是项目首个 real held-out action-conditioned language/world-output 任务上的 SNN 功能非劣证据，不再依赖逐参数轨迹。
- AT1 channel mean NLL：observation `2.754`、counterfactual `2.703`、admissible-actions `2.415`、reward `1.612`、done `1.439`、won `.486`。难点确实位于自然语言 observation/counterfactual，不只是 reward/done 格式。
- 全部 outcome payload 的 dense test NLL 同样保持：AT1 三 seed约 `2.645/2.437/2.508`，不是只在16个位置记标签。

### 训练与实时

真实端到端 update mean p50：AT0-BPTT `3.122 ms`、AT1 `4.032`、LSTM `1.766`、Transformer `4.797`；AT1/AT0 speedup=`0.774x`。因此 **H-TW0-SPEED FAIL**。register 的速度胜利没有迁移到 trainable embedding + K16：dense input adjoint 与 K×parameter eligibility 的重复工作是现行瓶颈。

每 seed完整 embedding+core+LM-head streaming 均过门：AT1 p50/p95约 `0.0874/0.1274`、`0.0874/0.1271`、`0.0867/0.1195 ms`，LSTM为 `0.1087/0.1569`、`0.1086/0.1492`、`0.1072/0.1377`。**H-TW0-STREAM PASS。** SNN state 248 bytes；Transformer cache继续随历史增长。

### 机制边界与决定

- full AT1 test accuracy约49–52%；spike-only为26–34%，trace-only为29–39%。与 register 的 spike-only chance 不同，真实语言中 spike 已含部分信息，但两种单独通道都明显不及 full，证明稀疏 spike 与连续 trace 是互补读出。
- **DATA/TASK/QUALITY/STREAM PASS，SPEED/overall FAIL；不进入 counterfactual generation。** 保留 TW0 为首个真实质量基线，进入 RA0 exact reverse adjoint，专门消除 trainable embedding 场景下的 forward-eligibility重复计算；不修改任务或结构。

---

## 2026-07-18：E3-TW0 预注册 — 真实 TextWorld action-conditioned sparse event LM（已执行）

### 为什么现在进入真实任务

AT1 与 AT2 连续得到同一个事实：两种 exact-surrogate 实现都能把 register 做到 3/3 seed 100%，但 hard threshold 令非凸优化轨迹对 `O(1e-8)` reduction 差异敏感。AT2 进一步证明，把梯度投影到 BF16 仍不能让长轨迹一致。因此从本轮开始，**不再把“与 BPTT 得到同一参数”当作真实模型质量代理**；AT1/AT2 的原冻结门仍记为 FAIL，不追溯改判。新实验用 held-out 真数据、多 seed NLL 和功能复现直接检验加速方法。

本地已有经过 manifest/SHA 验证的真实 TextWorld Coin Collector level-5 事件语料：train 4 个 game seed、valid/test 各 1 个互斥 seed，train/valid/test token 数为 `10,983 / 2,751 / 2,783`，词表仅由 train 构造（344 tokens），episode/chunk 不跨边界。这比再造一个 synthetic register 更接近“动作→环境响应”的世界模型接口。

### 任务方向比较

| 方向 / 标签 | 监督 | 世界模型相关性 | 与 AT1 的适配 | 风险 |
|---|---|---:|---:|---|
| **TextWorld outcome-channel sparse next-token LM** / New composition | 只在 observation/reward/done/won/admissible-actions/counterfactual payload 内选 K 个 causal target | **高：真实 action-conditioned 语言与反事实响应** | **高：K-query 原生** | 稀疏采样可能偏向格式 token；语料小 |
| TextWorld dense event LM / Established baseline | 每个 token next-token | 中高 | 低，K≈T 时 eligibility 优势消失 | 格式/ASCII banner 主导，难定位世界状态能力 |
| counterfactual-only next observation / Focused causal task | 给 action，生成 counterfactual `next_obs` | 很高 | 中 | 训练 transition 数只有几十，held-out 文本开放词表 |
| TextWorld action imitation / Established policy task | observation→expert action | 中 | 高，action boundary 稀疏 | 更像 policy，不直接学习环境动力学 |
| HomeGrid multimodal dynamics / Cross-modal next step | 图像/符号 observation + action→next state/reward | **很高** | 中 | 应在语言门之后单独验证 encoder 与闭环 rollout |
| WikiText dense LM / Established language task | 全 token NLL | 低于 world-model | 低 | 无 action/observation 因果结构 |
| 在线 closed-loop TextWorld rollout / Ultimate evaluation | 模型状态驱动 action，真实环境回馈 | 最高 | 后续 | 当前先要证明 held-out teacher-forced dynamics 非劣 |

先执行第一条，再按 counterfactual-only → HomeGrid multimodal → closed-loop rollout 推进。它不是为了制造容易的 K：正式产物必须报告 K/T 密度、channel 分布、格式 token 占比和 full outcome-channel NLL，若监督主要落在标点/JSON 结构，任务判 INVALID。

**What if：**世界模型的主要学习信号天然集中在动作后的 observation/reward/done 与反事实分支，而非每个叙述 token；若每 512-token chunk 只取最多 16 个真实 outcome payload token，AT1 是否能保持 held-out next-token NLL，同时把端到端训练和 constant-state response 都压到 LSTM 以下？

### 冻结数据与模型

- 数据根固定为 `results/e2_world_model/textworld_l5`；runner 必须复用现有 manifest/game SHA 验证，不生成或回退 synthetic。split 由 game seed 隔离；每 epoch 按 episode/时间原序遍历，episode 首 chunk 清 state，chunk 间显式 state detach。
- `sequence_length=512`，selected channels 固定为 `{observation,reward,done,won,admissible_actions,counterfactual}`。只把 channel closing `>` 后到该行 `<eos>` 前的 payload token 视为候选；每 chunk 候选>16 时按等距索引确定性取 16 个，否则全取。query 是目标 token 的前一 causal position；不选择 step/header/channel-marker token。
- 训练 `20 epochs`、AdamW `lr=1e-3,weight_decay=.01`、clip=1、seeds `{0,1,2}`、CPU 4 threads。embedding/output norm/tied LM head 都训练；四模型共享 wrapper 初值，只替换 `AT0 gated-trace BPTT / AT1 eligibility / LSTM / 1-layer Transformer`，total parameter spread<=2%。AT1 使用默认 segment forward，不使用 AT2 BF16 或 scan-aligned 变体。
- 每模型先记录未训练 valid/test sparse NLL，再训练；正式评估同时报告 sparse query NLL/PPL/top1/channel 分解与全部 outcome payload token 的 dense NLL。Transformer cache window=512；SNN/LSTM state 跨 chunk，Transformer cache也按相同 episode边界重置。

### 冻结门

- **H-TW0-DATA**：所有 manifest/game/event SHA 通过；train/valid/test 均有非零 selected query，每个 query 严格位于指定 payload；候选中 JSON/标点 token 比例必须<70%，否则任务 INVALID。
- **H-TW0-TASK**：LSTM 与 Transformer 在每个 seed 的 held-out test sparse NLL 均比各自未训练值降低>=`0.10`；否则训练预算/任务 INVALID。
- **H-TW0-QUALITY**：AT0/AT1 每 seed test sparse NLL 也均改善>=`0.10`；AT1 三-seed mean NLL 不高于最佳 ANN mean+`0.25`，且 AT1 与 AT0 的 mean absolute NLL gap<=`0.10`。不要求逐 update loss/参数一致。
- **H-TW0-SPEED**：真实训练 update（含 embedding、core、K-query head、CE、backward、clip、AdamW）后 20% warmup 的 p50；AT1 比 AT0 BPTT快>=`1.25x`且 p50<=LSTM。另报 Transformer、每秒 input tokens 与 supervised query tokens。
- **H-TW0-STREAM**：同一 test episode、64 warmup+512 measured、4 threads，完整 embedding+core+LM head 的 p50/p95 均<=LSTM；state bytes 单列。
- **H-TW0-MECH**：AT1 test 做 spike-only/trace-only readout 消融；只报告不作为过门。若 spike-only 仍为 chance/格式基线，继续使用“event-driven trace SNN substrate”边界。
- 全部门通过才进入 counterfactual sequence generation；质量失败转更强 recurrent/adaptive spike，速度失败转 native fused eligibility kernel。runner/产物固定为 `experiments/e3_tw0_sparse_event_lm.py` 与 `results/e3_scan/e3_tw0_sparse_event_lm.json`。

---

## 2026-07-18：E3-AT2 结果 — scan forward 逐位对齐，BF16 梯度仍无法锁定轨迹（负面结果）

### 证据

- 正式命令：`.venv-wsl/bin/python experiments/e3_at2_bf16_canonical.py --output results/e3_scan/e3_at2_bf16_canonical.json`；SHA-256 `F57B91FF6A32669ADA84D0CECC06C2FE206D636D875A7ADC3334E2B9D89FDA65`。
- `scan_aligned` 令 T=`1/32/512` 的 query raw、sequence 与 final state 相对 AT0 全部 **bit-exact**；资格迹全梯度仍在 AT1 tolerance 内。saved bytes 与 AT1 相同：T2048 ratio=`0.735%`、128→2048 growth=`1.0x`。**H-AT2-EQ/MEM PASS。**
- scan-aligned 增加了 forward 常量，但 4-thread T2048 为 AT2 `4.022 ms` vs AT0 `5.840` / LSTM `4.515`，16-thread T2048 为 `6.538` vs `13.445 / 8.857`；**H-AT2-SPEED PASS**。streaming 不受训练 forward mode影响；4-thread cached p50/p95 `0.0656/0.0884 ms` vs LSTM `0.0709/0.0936`，故同线程 **H-AT2-STREAM/ANN PASS**。

### canonicalization 失败

| seed | BF16 后 gradient mismatch | first mismatch | loss max diff | param max diff | AT0 / AT2 test |
|---:|---:|---:|---:|---:|---:|
| 0 | 3,184,090 / 5,396,400 = 59.00% | update 2 | 1.809 | .2267 | 100% / 100% |
| 1 | 2,994,000 / 5,396,400 = 55.48% | update 2 | 1.752 | .2866 | 100% / 100% |
| 2 | 3,026,227 / 5,396,400 = 56.08% | update 2 | 2.057 | .2097 | 100% / 100% |

- update 1 的 8,994 个 trainable gradient 元素全部量化一致，smoke 的三步小 batch 也全一致；正式 batch32 在 update2 即出现少量不同，随后 hard events 放大为不同路径。BF16 提供相对 mantissa 精度，对接近零梯度没有固定宽度的共同零槽，所以它不是 reduction-order canonicalizer。
- 两个 SNN 仍在每 seed 达到 100%，LSTM/Transformer 也验证任务；失败仅是预注册 canonical/trajectory gate，但必须记为 **H-AT2-CANON/QUALITY/overall FAIL**，不追溯放宽。
- power-of-two INT8 可以提供绝对 block bin，但任何离散量化仍有边界，继续用“是否复刻 BPTT 参数”决定世界模型质量会把研究引向数值仪器而非任务。保留 INT8 为未来低精度吞吐实验；当前更有信息量的下一步是 TW0：在真实 held-out action-conditioned language 上比较功能质量、端到端速度与多 seed 稳定性。

---

## 2026-07-18：E3-AT2 预注册 — scan-aligned forward + BF16 canonical gradient（已执行）

### AT1 后的数值鲁棒路线比较

AT1 已经在三 seed 上与 AT0 同为 100%，并同时通过训练、streaming、内存和同线程 ANN 门；失败只来自“逐 update 轨迹必须接近”的冻结子门。单步最大梯度误差仅 `5.22e-8`，但 hard threshold 将这种合法浮点 reduction 差异在 seed 1/2 放大为不同的事件/优化路径。下一轮不修改 gated trace 动力学，而是寻找既适合低精度硬件、又能把等价梯度映射到同一更新的 canonical training math。

| 路线 / 标签 | 数学手段 | 预期作用 | 主要风险 |
|---|---|---|---|
| **scan-aligned query forward + BF16 gradient projection** / Cross-domain transfer | query trace 复用 AT0 同一 Hillis–Steele reduction；`Q(g)=float32(bfloat16(g))` 后再 clip/AdamW | forward bit 对齐；小 reduction 差异落到同一低精度格点 | BF16 梯度可能损害小梯度；全 trace forward 增加常量 |
| power-of-two INT8 block gradient / Established compression | 每 tensor 共享 2 的幂 scale，梯度映射到 signed int8 | 更强 canonicalization，未来通信/硬件更省 | 量化过粗，decay/rare-event gradient 易归零 |
| threshold-margin regularization / Established robustness | 惩罚 event drive 与 output trace 靠近阈值 | 从动力学上降低离散翻转敏感度 | 新增损失权重，可能压低事件稀疏性或改任务解 |
| hysteretic/dead-zone spike / Established dynamical idea | 阈值附近保持旧事件或输出 0 | 显式消除窄边界抖动 | 增加状态/串行依赖，破坏当前 exact scan |
| compensated/pairwise eligibility reduction / Established numerics | Kahan 或固定 reduction tree 计算资格迹收缩 | 降低 FP32 舍入误差且不量化 | 很难逐位复刻 autograd graph，CPU 常数更大 |
| soft-to-hard curriculum / Established training | 早期连续概率，后期 hard event | 减少早期混沌分叉 | 训练 forward 不再始终严格二值，偏离当前 SNN 边界 |
| 统计功能等价而非轨迹等价 / Evaluation alternative | 多 seed 比较 accuracy/NLL/event 分布，不要求同参数 | 更符合非凸训练实际 | 只改变证据标准，不能解释/修复数值脆弱性 |

先执行第一条：它不靠 AT0/AT1 梯度互相通信，部署训练只需对自身梯度做确定性 BF16 投影；这也是最小、可证伪且保留 strict binary forward 的方案。INT8、margin、curriculum 作为后续独立实验，禁止在本轮失败后临时扫参。

**What if：**资格迹与 BPTT 的数学梯度相同，但浮点 reduction tree 不同；若先让 query forward 使用完全相同的 scan，再把两种合法 FP32 梯度投影到同一 BF16 格点，是否能在不恢复长 autograd graph 的情况下得到同一 hard-event 优化路径？

### 冻结实现与门

- 新增 `eligibility_forward_mode="scan_aligned"`：custom forward 对 trace 使用 AT0 完全相同的 full affine prefix scan，仅在 custom Function 内取 K 个 query；eligibility/自定义 backward 不变。默认仍为 AT1 的 `segment`，确保 AT1 正式产物可复现。
- 质量训练的 AT0-BPTT 与 AT2 都在 `backward` 后执行 `Q_BF16(g)`，再执行原来的 global norm clip=1 与 AdamW；参数和 optimizer moment 保持 FP32。没有 loss/参数互传，也不量化 inference。
- **H-AT2-EQ**：复用 AT1 三个 case；scan-aligned query raw/final state 的 hard event 与 AT0 bit-exact，全部梯度仍满足 `2e-5/1e-4`。
- **H-AT2-CANON**：三 seed×600 update，每一步报告量化后梯度逐元素 mismatch；总体 mismatch rate 必须 `<=1e-5`，且不得出现非有限值。
- **H-AT2-QUALITY**：同一 register、初始化、batch 与 test；AT0-BF16/AT2-BF16 每 seed 都须 100%，逐 update loss max<=`1e-3`、最终参数 max<=`5e-3`。LSTM/Transformer 的任务有效性由 AT1 同日正式运行再次确认，不改变任务。
- **H-AT2-SPEED/MEM**：同 AT1 的 threads、T、K；scan-aligned AT2 至少一档相对 AT0-BPTT `>=1.25x` 且 p50<=LSTM，T=2048 core-only saved bytes<=25% 且增长<=1.25x。BF16 projection 的 optimizer-level质量 timing 单列，不拿 core-only speed 冒充端到端。
- **H-AT2-ANN**：训练速度门必须与 AT1 已验证的 cached-decay streaming 门在同一线程成立；本轮重跑 streaming，不能跨运行拼最佳数字。
- 正式 runner/产物：`experiments/e3_at2_bf16_canonical.py`、`results/e3_scan/e3_at2_bf16_canonical.json`。只有全部通过才升级真实 action-conditioned language task。

---

## 2026-07-18：E3-AT1 结果 — 训推/内存/最终质量全过，但 hard-threshold 轨迹门失败（关键混合结果）

### 证据与工程门

- 正式命令：`.venv-wsl/bin/python experiments/e3_at1_trace_eligibility.py --output results/e3_scan/e3_at1_trace_eligibility.json`；SHA-256 `7ED99BCED90AE01F4F2EADE2C481C2043A9EBC98443C099172A385C05BCC2658`。环境仍为 PyTorch `2.13.0+cpu`、Ryzen 9 7950X、CUDA unavailable。
- 实现 exact gated-trace K-query custom backward：fused content/gate event projection、decay logit、初态与 input gradient 均覆盖；另新增缓存 bounded decay 的 tensor-only event step。
- 三个冻结 case 全通过。T=`1/32/512` 的 query raw 最大误差为 `0 / 1.19e-7 / 2.68e-7`，全部梯度最大误差为 `7.45e-9 / 3.73e-8 / 5.22e-8`；非法 query 与 cached/full/uncached step 等价也全部 PASS。**H-AT1-EQ PASS。**

core-only、K=4 的 unique saved bytes：

| T | AT0-BPTT | AT1 | ratio |
|---:|---:|---:|---:|
| 128 | 797,416 | **121,568** | 15.25% |
| 512 | 3,643,112 | **121,568** | 3.34% |
| 2048 | 16,549,608 | **121,568** | **0.735%** |

- AT1 的 128→2048 growth=`1.0x`；input-grad T2048 为 `1,137,376` bytes，仍比对应 AT0 的 `16,565,480` 少 93.1%。T512 的 K=`1/4/16/32` 保存量为 `67,592/121,568/337,472/625,344`，按 K 线性退化但 K32 仍仅为 AT0 的 17.1%。**H-AT1-MEM PASS。**

K=4 query loss 的正式 p50 ms：

| threads | T | AT0-BPTT | AT1 | speedup | LSTM | IC0-EL1 | Transformer | gate |
|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | 512 | 3.85 | 1.92 | 2.01x | **1.42** | 1.14 | 7.40 | FAIL absolute |
| 1 | 2048 | 8.36 | 4.91 | 1.70x | **3.73** | 1.88 | 97.23 | FAIL absolute |
| 4 | 512 | 2.79 | 1.76 | 1.58x | **1.41** | 1.14 | 2.78 | FAIL absolute |
| 4 | 2048 | 7.35 | **3.65** | 2.01x | 4.98 | 1.69 | 27.69 | **PASS** |
| 16 | 512 | 4.54 | **2.06** | 2.21x | 2.81 | 1.72 | 3.66 | **PASS** |
| 16 | 2048 | 14.03 | **5.14** | 2.73x | 9.49 | 2.27 | 29.48 | **PASS** |

AT1 autograd node 为 13，AT0 随 T512→2048 为 263→311。**H-AT1-SPEED PASS。** cached step 在 threads `1/4/16` 的 p50/p95 为 `0.0653/0.1070`、`0.0650/0.1075`、`0.0788/0.1730 ms`，三档均不慢于 LSTM 的 `0.0709/0.1117`、`0.0746/0.1290`、`0.1054/0.2165`；uncached AT0 p50 为 `0.0728/0.0737/0.0902`。**H-AT1-STREAM PASS；4-thread T2048 与 16-thread 两个 T 均同时满足训练，因此 H-AT1-ANN PASS。**

### 最终任务质量与预注册失败点

所有模型在 16,384 query/seed 上仍为 100%：

| seed | AT0 acc/NLL | AT1 acc/NLL | LSTM | Transformer | loss max diff | parameter max diff |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 100% / .00583 | 100% / .00563 | 100% | 100% | .00117 | .00978 |
| 1 | 100% / .00722 | 100% / .00751 | 100% | 100% | .95599 | .04690 |
| 2 | 100% / .00566 | 100% / .00547 | 100% | 100% | 1.88749 | .07735 |

- 功能质量本身完全复现，且 AT1 NLL 与 AT0 同量级；但三个 seed 均超过预注册的 `loss<=1e-3 / parameter<=5e-3` 轨迹门，seed 1/2 被微小 reduction 差异跨 hard threshold 后显著放大。因此 **H-AT1-QUALITY FAIL**，不能因最终 accuracy 好看而改门。
- 单步梯度误差仅 `O(1e-8)` 与长轨迹分叉同时成立：这是 hard-event 优化的数值脆弱性证据，不是 custom backward 数学不等价证据。正式判定为 **EQ/MEM/SPEED/STREAM/ANN PASS，QUALITY/overall FAIL；不进入 TextWorld。**

### spike/trace 机制消融

- spike-only accuracy 为 seed `0/1/2 = 6.37% / 6.23% / 5.92%`，即 chance；输出 spike 单独没有承载可解码记忆。
- trace-only 为 `100% / 88.59% / 92.99%`，显著高于 chance 但后两 seed 不足以完全复现 full 100%。因此正确表述是：**连续 signed trace 是主要记忆载体，稀疏 output spike 提供互补读出，但 spike-only code 尚未学会任务。** 当前结果只能称“事件驱动 trace SNN substrate”，不能称完全 spike-coded 世界模型。
- 保留 AT1 作为已证明的工程加速原语：T-independent exact sparse-query backward、长序列多核训练超过 LSTM、constant-state streaming 也超过 LSTM。下一步 AT2 解决数值 canonicalization；若通过，再进入真实 action-conditioned language task，同时另开结构路线提高 spike-only 信息量。

---

## 2026-07-18：E3-AT1 预注册 — gated-trace exact K-query eligibility + cached-decay step（已执行）

### AT0 成功后的加速路线选择

AT0 已证明动力学质量，不再改事件、decay 范围、trace 或 readout。AT1 只压缩同一函数的训练反向与 streaming 常量开销：

| 路线 / 标签 | 机制 | 精确性 | 最小实验 | 主要风险 |
|---|---|---:|---|---|
| **K-query forward eligibility** / Speculative specialisation | 为 fused event projection、decay 与初态递推 exact prefix Jacobian，只保存 K 个快照 | **exact surrogate gradient** | 全梯度矩阵 + saved storage + 600-step 复现 | eligibility 为 `4S×D`，K 密集时增长 |
| reverse adjoint + forward recompute / Established engineering | backward 重算 events/trace，再做反向 affine scan | exact | 时间/峰值内存对照 | 保存或重算完整 T，未必比 scan BPTT 快 |
| block checkpoint / Established engineering | 分块保存 trace，块内重算 | exact | block sweep | 不能消除 autograd 与重算成本 |
| `torch.compile` whole scan / Established engineering | 编译 Hillis–Steele 图与 backward | exact within compiler | cold/steady 分列 | 动态 T、图大、CPU compile 成本高 |
| C++/Triton fused affine scan / Established engineering | 专用 forward/backward kernel | exact | GPU/CPU extension benchmark | 当前无 CUDA；开发成本高且不先修数学 |
| pp-prop/e-prop low-rank trace / Established approximation | 用 pre/post 因子减少 full Jacobian | approximate | 与 AT1 exact 做 cosine/quality gap | 当前单层 exact 特例没必要先牺牲精度 |

先执行 exact K-query eligibility；它是 AT0 方程本身的 forward-mode Jacobian，不依赖近似 learning signal。cached-decay tensor step 同时消除 streaming 每 token 重算 `sigmoid(decay_logits)`。若两者仍差 LSTM，才将同一数学交给编译/原生 kernel，而不是改质量已通过的动力学。

**What if：**AT0 的 trace recurrence 与它对参数的 eligibility recurrence具有同一个 decay `lambda`；如果把输出 learning signal 留到 query 才乘，是否能像 EL1 一样把 263/311-node scan backward 折叠成一个 custom node，同时保持可学习 memory？

### 精确梯度构造

对一个 population：`h_t=lambda*h_{t-1}+(1-lambda)v_t`，`v_t=c_t*g_t`；query raw 为 `[H(h_q-theta),h_q]`。query learning signal：

`L_q = g^h_q + g^s_q*phi_out(h_q-theta)`。

content/gate 权重 eligibility 分别递推：

`E^c_t=lambda E^c_{t-1}+(1-lambda) g_t phi_c(d^c_t) outer x_t`；

`E^g_t=lambda E^g_{t-1}+(1-lambda) c_t phi_g(d^g_t) outer x_t`。

decay-logit eligibility 为：

`R_t=lambda R_{t-1}+d(lambda)/d(logit)*(h_{t-1}-v_t)`。

于是 `grad_W=sum_{batch,q} L_q E_q`、`grad_decay=sum L_q R_q`，final-state learning signal 以 T 末快照同样加入；初态系数为 `lambda^(q+1)`。forward 按 query segment 用指数加权 einsum 更新 snapshot，时间点只遍历一次，core-only 保存量 `O(K*4BSD)`、对 T 常数。需要 input gradient 时保存 local event derivative，并用反向 affine adjoint `p_t=L_t+lambda p_{t+1}` 精确恢复 `grad_x`，该模式单独报告。

### 冻结门

- **H-AT1-EQ**：相对普通 AT0 scan，覆盖 `(B,T,K,input_grad)=(1,1,1,on),(2,32,4,on),(1,512,8,off)`、外部初态；query output/final trace、input/initial/fused event projection/decay/output norm/projection 全梯度满足 `atol=2e-5,rtol=1e-4`，hard events/spikes bit-exact；非法 query 索引复用 EL1 validation。
- **H-AT1-MEM**：`B=1,D=32,K=4,T={128,512,2048}`，core-only T=2048 unique saved bytes <= 普通 AT0 query-BPTT 的 25%，且 T=128→2048 growth<=1.25x；input-gradient 与 K={1,4,16,32} 单列。
- **H-AT1-SPEED**：threads 1/4/16、T={512,2048}，AT1 至少一档比普通 AT0 scan query-BPTT 快 `>=1.25x`，且绝对 p50<=同任务 LSTM。IC0-EL1 与 Transformer 继续报告。
- **H-AT1-QUALITY**：完全复用 AT0 三 seed register、初始化、600 updates；AT1 与普通 AT0 每 seed 都须 100%，AT1/AT0 逐 update loss max<=`1e-3`、最终参数 max<=`5e-3`。较宽的轨迹 tolerance 是在 EL1 已观察到 hard-threshold 浮点分叉后预先冻结，不放宽单步 H-EQ。
- **H-AT1-STREAM**：inference 预计算一次 bounded decay，逐 token tensor step 不重复 sigmoid；threads 1/4/16，64 warmup+512 measured，至少一档 p50/p95 同时<=LSTM。cached 与 uncached AT0、IC0 tensor、Transformer 都报告；缓存只在参数更新后失效一次。
- **H-AT1-ANN**：H-SPEED 与 H-STREAM 必须在同一线程 lane 成立。所有 AT1 门通过后才进入 TextWorld；质量通过但 ANN 门失败则保留 AT1，转 C++/Inductor kernel。
- **机制消融（报告项）**：在训练后同一 test 上将 `[spike channels]` 或 `[trace channels]` 置零，报告 full/spike-only/trace-only accuracy。若只有 trace-only 保持质量，结论必须写成“事件驱动 trace SNN”，不得声称输出 spike code 已承载记忆。

### 产物

- 代码/API：`E3GatedTraceScanCore.forward_multi_query_eligibility`、cached-decay tensor step；runner `experiments/e3_at1_trace_eligibility.py`。
- 正式产物：`results/e3_scan/e3_at1_trace_eligibility.json`；smoke 不能覆盖。

---

## 2026-07-18：E3-AT0 结果 — 三 seed 短期记忆达到 100%，训推速度仍差临门一脚（混合正面结果）

### 证据 / 实现

- 正式命令：`.venv-wsl/bin/python experiments/e3_at0_gated_trace.py --output results/e3_scan/e3_at0_gated_trace.json`；产物 SHA-256 `EB4636FF39B1DED55A4B432C9E6BFDCA41933B16636C5B32EDF323D86F9F13D2`。环境为 PyTorch `2.13.0+cpu`、Ryzen 9 7950X、CUDA unavailable。
- 新增 `E3GatedTraceScanCore`：单个 fused linear 产生 E/I content 与 write-gate binary events，`content*gate` 驱动 31 维异质 decay trace；输出为 trace threshold spike + signed trace。另实现 exact tensor-only streaming step，不改变训练方程。
- core 参数/state 为 `8,402 / 248 bytes`，LSTM 为 `8,448 / 256 bytes`；含共同 wrapper 后 AT0/LSTM/Transformer 为 `8,994/9,040/9,200`，全部在 2% 门内。

### 精确 scan 与并行门

- serial/scan 的 `(B,T)=(1,1),(4,32),(1,512)` 全部通过；T=512 最大 forward/gradient abs 为 `9.54e-7/5.44e-7`。content/gate/write/output spike 逐元素 bit-exact 且 binary，trace 保持 `[0,1]`。连续 64-token tensor/generic streaming 对 full scan 的 sequence/state 最大误差不超过 `5.36e-7`。**H-AT0-EQ PASS**。

K=4 query loss，p50 ms：

| threads | T | AT0 scan | AT0 serial | speedup | node ratio | LSTM | IC0-EL1 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 512 | 2.841 | 21.773 | 7.66x | 6.36% | **1.237** | 1.120 |
| 1 | 2048 | 7.448 | 110.034 | 14.77x | 1.89% | **3.512** | 1.883 |
| 4 | 512 | 2.534 | 20.954 | 8.27x | 6.36% | **1.407** | 1.046 |
| 4 | 2048 | 6.262 | 118.222 | **18.88x** | 1.89% | **5.116** | 1.768 |
| 16 | 512 | 3.235 | 25.799 | 7.97x | 6.36% | **2.764** | 1.913 |
| 16 | 2048 | 13.514 | 171.258 | 12.67x | 1.89% | **9.340** | 2.979 |

所有 lane 都超过 5x 且 node ratio<25%，故 **H-AT0-PAR PASS**。但 AT0 scan 在所有 lane 都慢于 LSTM；最好是 4-thread T=2048 的 `6.262 vs 5.116 ms`，仍慢 22.4%。

### 首个有效的纯 SNN delay4 质量结果

EL1 四段 WRITE→delay4→QUERY register 完全复用，三个模型的 test query 均为 16,384/seed：

| seed | AT0 accuracy / NLL | LSTM accuracy / NLL | Transformer accuracy / NLL | AT0 train p50 | LSTM | Transformer |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | **100% / 0.00594** | 100% / 0.00761 | 100% / 0.00798 | 3.181 ms | **1.066** | 2.173 |
| 1 | **100% / 0.00526** | 100% / 0.00795 | 100% / 0.00819 | 3.358 ms | **1.203** | 2.536 |
| 2 | **100% / 0.00611** | 100% / 0.00754 | 100% / 0.00802 | 3.514 ms | **1.092** | 2.281 |

- **H-AT0-QUALITY PASS。** 这是本项目 strict binary input/write event SNN 首次在有效延迟任务上三 seed 达到 ANN 质量；NLL 还略低于两种 ANN，但任务很小，不能外推为总体质量优势。
- 训练后 write event rate 约 `5.99–12.18%`，output spike rate 约 `0–6.35%`，trace 范围合法，decay 仍覆盖约 `0.548–0.990`。seed 2 的 excitatory output spike rate 为 0 但仍 100%，说明本任务可能主要使用 signed slow trace 与 inhibitory event；后续必须做 spike-only/trace-only 消融，不能把成功全归因于输出 spike code。
- 与 IC0 的 6–8% 对照支持结构解释：可学习 memory 来自 gated non-reset slow trace，而不是仅增加 update 或换任务；输入、wrapper、预算和 ANN controls 都与 EL1 相同。

### 实时门与总判定

exact tensor step 的 p50/p95 ms：threads 1 为 AT0 `0.0793/0.1211` vs LSTM `0.0701/0.1085`；4 为 `0.0778/0.1616` vs `0.0718/0.1258`；16 为 `0.1069/0.1838` vs `0.1063/0.1800`。16-thread 只差 `0.62%/2.10%`，但冻结门要求两者都不慢，三档仍全部 FAIL。

- **H-EQ PASS；H-QUALITY PASS；H-PAR PASS；H-ANN FAIL；overall=FAIL。** 不直接进入 TextWorld；质量成功不能替训练慢 22%–112% 与 streaming 尾延迟落后的工程事实过门。
- **保留 AT0 动力学并进入 AT1。** 训练侧为 gated affine trace 推导 exact K-query eligibility，避免构建 263/311-node scan backward；推理侧缓存静态 decay 并测试 fused/compiled tensor step。AT1 必须复现三 seed 100% 且同线程训推都达到 LSTM，才升级真实任务。
- AT0 仍有明确边界：slow trace 不被 output spike reset，且 readout 可见 continuous trace。它是纯事件驱动 SNN 的 synaptic-state substrate，不是最终完整 hard-reset recurrent 世界模型；后续真实任务与 spike/trace 消融决定是否需要 ALIF/recurrent E/I。

---

## 2026-07-18：E3-AT0 预注册 — gated synaptic-trace exact scan（已执行）

### EL1 后的结构路线比较

EL1 在有效 delay4 任务上把“训练图太慢”和“状态不会记忆”明确拆开：即使梯度精确且长序列训练超过 LSTM，additive modulo IC0 仍为 chance。因此下一轮必须先改变纯 SNN 的时间状态，再为成功动力学推导 online/eligibility；不能继续对 IC0 加训练技巧。

| 结构方向 / 认识论标签 | 状态机制 | 记忆潜力 | 时间并行 | 工程成本 | 主要风险 |
|---|---|---:|---:|---:|---|
| **AT0 gated synaptic trace** / Speculative new specialisation | 二值 content×write events 驱动异质指数突触 trace；spike 只读取 trace | 中高 | **exact affine scan** | 低 | trace readout 成功但 spike code 无效；无 reset/recurrent interaction |
| ALIF/LSNN adaptation / Established direction | spike 提高慢阈值，阈值指数衰减 | 高 | 低；spike/reset 决定未来阈值 | 中 | serial BPTT/RTRL 成本高，CPU 难追 ANN |
| recurrent E/I event path / Established direction | 上一步 E/I spike 经符号固定 recurrent weights 回注 | 高 | 低；需 fixed point、event segmentation 或 online gradient | 高 | dense recurrent GEMM 与因果链同时变慢 |
| multi-timescale delay line / Cross-domain analogy | 多个固定/可学习 decay bank 近似 SSM basis | 高 | exact scan/卷积 | 中 | 状态字节随 timescale 数增长，容易变成 ANN SSM |
| oscillatory phase memory / Established direction | stable complex/real oscillator 以相位保存事件 | 中高 | exact affine scan | 中 | P0 已在本机速度门失败，phase code 质量未证 |
| event-addressed key/value synapses / Speculative new idea | spike key 选择写入局部 trace，query key 选择读出 | 很高 | key-local 可稀疏并行 | 很高 | 可能等价偷渡 attention，学习规则与硬件均未定 |

推荐顺序为 AT0 → ALIF/recurrent E/I。AT0 的决定性优点不是“更像 LSTM”，而是把非线性 spike/reset 从记忆递推中移出：slow trace 仍只由 0/1 事件更新，输出仍有 0/1 threshold spike，但 trace recurrence 可被精确 scan；若它连 delay4 都学不会，则直接转真正 recurrent/adaptive spike，而不再调 trace。

**What if：**把 hard reset 只留在快速输出膜，而让长期记忆位于不被 spike 清零的慢突触电流中，是否能同时获得 SNN 事件接口、可学习延迟状态和 exact time-parallel training？AT0 只测试这个最小命题，不把 reset-free slow trace 冒充最终完整神经元模型。

### 冻结动力学

`state_dim=31`，每个时间点由单个 fused linear 产生四组 binary events：`c_E,c_I`（content）与 `g_E,g_I`（write gate），全部使用 IC0 同一 hard surrogate threshold；实际写事件为 `v_E=c_E*g_E`、`v_I=c_I*g_I`，仍严格 0/1。每个神经元有可学习但有界的静态 decay：

`lambda = 0.5 + (0.995-0.5)*sigmoid(decay_logit)`；

`h_t = lambda*h_{t-1} + (1-lambda)*v_t`；

`s_t = H(h_t-0.5)`；

`y_t = Linear(LayerNorm([s_E,-s_I,h_E,-h_I]))`。

`h` 解释为慢突触/膜 trace，若初态与事件在 `[0,1]`，forward 应保持 `[0,1]`；`s/c/g/v` 全为 binary。没有 sigmoid/tanh recurrent activation、没有 ANN hidden-to-hidden matrix；continuous trace 是 SNN 标准内部状态。当前 slow trace 不因 spike reset，边界必须保留。

scan 把每步视为 affine pair `(lambda,(1-lambda)v_t)`，用已验证的 Hillis–Steele composition 得到全部前缀；serial 是逐步真值。decay 初值在 31 个神经元上均匀覆盖 `[0.55,0.99]` 的 logit 空间，避免全群同一时间常数。fused input event projection 参数为 `D×4S`；AT0 core 预期约 `8,402` 参数、persistent state `248 bytes`，对 LSTM `8,448/256` 在 2% 内。

### 冻结门

- **H-AT0-EQ**：serial/scan 覆盖 `(B,T)=(1,1),(4,32),(1,512)`、随机外部 trace；sequence/state/逐时刻 trace、input/initial/全部参数 gradient 满足 `atol=2e-5,rtol=1e-4`，content/gate/write/output spike 逐元素 bit-exact 且 binary，trace 始终 `[0,1]`；连续 64-token `step` 与 full scan 最终状态/输出等价。
- **H-AT0-QUALITY**：完全复用 EL1 已冻结且有效的四段 WRITE→delay4→QUERY 数据、正交 embedding、seeds `{0,1,2}`、600 updates、batch32、test 4096、AdamW/clip。LSTM/Transformer 每 seed 仍须 `>=99%`；AT0 每 seed query accuracy 也须 `>=99%`。不因结果改变 decay 范围、threshold、update 或 embedding。
- **H-AT0-PAR**：`B=1,D=32,T in {512,2048}`，threads 1/4/16，scan/serial 相同 K=4 query loss；至少一档 scan 比 serial 快 `>=5x` 且 autograd node `<=25%`。
- **H-AT0-ANN**：同一长序列 benchmark 中至少一档 scan train p50 不慢于 fused LSTM，并在相同线程、B=1 的 continuous streaming p50 与 p95 都不慢于 LSTM。streaming 使用与 EK0 同边界的 exact tensor-only inference step，另以 full scan/连续 step 等价测试约束；编译不是必需门。另报 Transformer、IC0-EL1、参数、state bytes、firing/event rate；不能跨线程拼接 train/step。
- 质量 PASS、速度 FAIL：保留动力学并进入 AT1 exact K-query trace eligibility/融合 kernel；质量 FAIL：AT0 关闭，转 ALIF 或 explicit recurrent E/I。只有质量与 ANN 训推门都通过才进入真实 TextWorld；AT0 即使通过也仍是 reset-free slow-trace 技术基底，不是最终完整世界模型。

### 来源 / 产物

- e-prop 原始结果把慢适应变量及其 eligibility 描述为跨越延迟监督的“future highway”，同时指出没有慢变量的普通 RSNN 在相同延迟任务上可能失败：<https://www.nature.com/articles/s41467-020-17236-y>。AT0 借用“慢内部变量承载信用”的结构动机，但其 gated affine trace 与 exact scan 是本项目特化，论文没有验证该方程。
- runner/产物冻结为 `experiments/e3_at0_gated_trace.py` 与 `results/e3_scan/e3_at0_gated_trace.json`；正式结果必须另起条目，失败同样保留。

---

## 2026-07-18：E3-EL1 结果 — K-query 精确加速成立，但 IC0 四步记忆彻底失败（负面结果）

### 证据 / 实现

- 正式命令：`.venv-wsl/bin/python experiments/e3_el1_multi_query_eligibility.py --output results/e3_scan/e3_el1_multi_query_eligibility.json`；产物 SHA-256 `3B53C5409B19459748B0E15B4D6EC59C4594480FC4B0B68F74F28BD4B3197BF3`。环境为 PyTorch `2.13.0+cpu`、Ryzen 9 7950X、CUDA unavailable。
- 新增 `E3InputCodedScanCore.forward_multi_query_eligibility` 与 custom backward。每个 query 只保存 exact prefix eligibility；`E_{tau-1}` 由 `E_tau-phi(d_tau) outer x_tau` 恢复，避免重复保存第二份 `[B,K,H,D]`。query validation 拒绝非 tensor、非一维、非 long、空、越界、重复与降序索引。
- formal 前 quick smoke 只检查执行链，未用于改门或正式结论；临时 smoke JSON 在结果冻结后删除。

### 等价与保存量

- `(B,T,K,input_grad)=(1,1,1,on),(2,32,4,on),(1,512,8,off)` 全通过；query raw/state bit-exact，spike 严格 0/1，sequence 与输入/初态/全参数梯度满足 `2e-6/1e-5`。三组最大 gradient abs 分别为 `2.98e-8 / 2.38e-7 / 1.91e-6`；八个非法索引 case 全被预期错误拒绝。**H-EL1-EQ PASS**。

| mode，K=4 | T=128 unique bytes / nodes | T=512 | T=2048 |
|---|---:|---:|---:|
| IC0-BPTT core-only | 298,848 / 43 | 1,125,216 / 43 | 4,430,688 / 43 |
| **EL1 core-only** | **99,504 / 14** | **99,504 / 14** | **99,504 / 14** |
| IC0-BPTT input-grad | 309,600 / 46 | 1,135,968 / 46 | 4,441,440 / 46 |
| EL1 input-grad | 142,512 / 15 | 271,536 / 15 | 787,632 / 15 |

core-only T=2048 ratio=`2.25%`，T=128→2048 growth=`1.0x`，故 **H-EL1-MEM PASS**。输入梯度仍随 T 线性增长，但比对应 BPTT T=2048 少 82.3%。T=512 的 K 扫描进一步给出：K=`1/4/16/32` 时 EL1/BPTT unique-byte ratio=`5.19/8.84/23.42/42.80%`；收益按 K 退化，但到 K=32 仍未交叉。

### 多核长序列训练速度

`B=1,D=32,K=4`，query-only forward+backward p50 ms：

| threads | T | EL1 | IC0-BPTT | speedup | LSTM | Transformer | gate |
|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | 512 | 1.126 | **1.103** | 0.98x | 1.119 | 7.145 | FAIL |
| 1 | 2048 | **1.820** | 2.960 | **1.63x** | 3.547 | 92.862 | **PASS** |
| 4 | 512 | 1.059 | **1.013** | 0.96x | 1.323 | 2.630 | FAIL |
| 4 | 2048 | **1.683** | 2.171 | **1.29x** | 4.847 | 25.911 | **PASS** |
| 16 | 512 | **1.780** | 1.998 | 1.12x | 2.844 | 3.616 | FAIL（<1.25x） |
| 16 | 2048 | **3.176** | 3.723 | 1.17x | 9.811 | 29.493 | FAIL（<1.25x） |

T=2048 的 1/4-thread lane 同时达到 `>=1.25x` 且快于 LSTM；**H-EL1-SPEED PASS**。T=512 没有加速，说明 K 个分段 einsum 的固定成本只在长序列摊薄；16 线程对该小矩阵反而增加调度开销。

### 有效质量任务上的结构失败

四段 WRITE→delay4→QUERY register 完全通过任务有效门：LSTM 与 Transformer 在 seeds `0/1/2` 的 16,384 个独立 test query 上均为 **100%**，NLL 分别约 `0.0062–0.0078 / 0.0076–0.0083`。因此本轮不再出现 EL0 的 INVALID 问题。

| seed | IC0-BPTT acc | EL1 acc | LSTM | Transformer | EL1 train p50 / BPTT / LSTM ms |
|---:|---:|---:|---:|---:|---:|
| 0 | 6.30% | 6.30% | 100% | 100% | 1.946 / 1.654 / 0.999 |
| 1 | 6.10% | 6.11% | 100% | 100% | 2.308 / 1.909 / 1.081 |
| 2 | 7.63% | 7.63% | 100% | 100% | 2.001 / 1.721 / 1.129 |

- IC0/EL1 loss 保持在随机猜测 `ln(16)≈2.773`，600 updates 后没有形成四步可读状态；input events 与 output spikes 全程严格二值，平均 firing/event rate 约 `0.52–0.62`，失败不是“没有 spike”。**H-EL1-QUALITY FAIL**。
- seeds 0/2 的 600-step loss/参数最大差仅 `7.15e-7/1.55e-6`；seed 1 则为 `1.33e-3/3.22e-3`。观察上，前期微小 floating reduction 差异在 hard input threshold 附近被离散事件放大，令两条优化轨迹后期分叉；两者准确率仍同为随机水平。该现象不推翻单步全梯度 tolerance 等价，但否定“长优化轨迹必然保持 1e-4”的质量子门。
- 在这个真实有效但短 T=32 的任务上，EL1 训练反而比普通 BPTT 慢约 16–21%，也慢于 LSTM；长 T 的 core benchmark 速度胜利不能替代质量约束下的训速结论。

### 判定 / 决定

- **H-EQ PASS；H-MEM PASS；H-SPEED PASS；H-QUALITY FAIL；overall=FAIL。** 不运行 TextWorld K-query，不把 EL1 宣称为世界模型训练方案。
- 保留 EL1 为“已证明的 sparse-query 长序列梯度压缩原语”，但不再对 additive/non-recurrent IC0 追加任务调参。EL0/EL1 共同说明：训练图与实时 kernel 已可超过 ANN，当前决定性瓶颈是 SNN state transition 不会保持可学习的短期内容。
- 下一实验必须改变**纯 SNN 动力学**而不是增加 ANN memory：优先测试带可塑时间常数的 adaptive spiking state（ALIF/LSNN 式 adaptation）或显式 recurrent E/I event path，并为其构造 online eligibility/low-rank RTRL；blockwise checkpoint 只解决 dense 训练内存，不能修复本轮表示失败。

---

## 2026-07-18：E3-EL1 预注册 — exact multi-query eligibility + 短延迟寄存器任务（已执行）

### 路线比较与执行顺序

用户已授权“沿用所有可能的数学方法”，所以这里不是排他单选，而是冻结执行顺序：先做当前 IC0 上可证明精确的 EL1，再把近似 online/local 方法作为独立实验，禁止用其中任一条的优点替另一条过门。

| 路线 / 认识论标签 | 核心机制 | 新颖性 | 可行性 / 最小判官 | 证据强度 | 潜在价值 | 主要失败方式 |
|---|---|---|---|---|---|---|
| **EL1 exact K-query prefix eligibility** / Speculative new specialisation | 对 K 个监督时刻保存 exact prefix eligibility，反向只组合 K 个 learning signal | 针对 IC0 方程的新特例 | **高**；直接与普通 BPTT 做全梯度矩阵等价 | EL0 已证明 K=1 特例 | 稀疏 action/observation 边界可得到与 T 解耦的 core-only 保存量 | K 接近 T 时状态按 K 线性增长；只适用 additive IC0 |
| blockwise exact checkpoint/adjoint / Established engineering direction | 每块保留边界状态，块内重算或 BPTT | 低 | 高；扫 block size 与重算时间 | 自动微分/checkpoint 已成熟 | 可覆盖 dense LM，不依赖稀疏标签 | 重算吞吐抵消内存收益，仍非严格在线 |
| e-prop learning signal × eligibility / Established direction | 前向递推局部 eligibility，当前/广播 learning signal 调制 | 中 | 中；先在同一寄存器任务比较 BPTT gap | 原论文覆盖延迟监督与 RSNN | 真正在线更新，可延伸 recurrent SNN | learning signal 忽略跨神经元未来路径，精度可能掉 |
| OTTT presynaptic trace / Established direction | detach temporal route，以 presynaptic trace 配即时 loss | 中 | 中；dense query loss 下对照 | 已有图像/事件任务证据 | 常数时间图、适合逐事件监督 | 稀疏延迟 query 缺少及时 learning signal |
| SLTT / SLTT-K / Established direction | 删除不重要 temporal route，只在 K 个时刻反传 | 中 | 高；K={1,4,16} 与 exact EL1 同台 | 论文报告内存与 T 无关并在视觉任务验证 K 稀疏反传 | dense 监督也能抽样，GPU 友好 | 随机 K 可能跳过世界模型罕见关键边界 |
| NDOT dynamics sensitivity / Established direction | 用神经动力学敏感度把 temporal/spatial gradient 前向分解 | 中 | 中低；需重推 modulo reset sensitivity | ICML 论文只在较短视觉时间步验证 | 比纯即时梯度保留更多历史依赖 | 对 exact hard modulo 的近似误差未知 |
| BrainTrace/pp-prop low-rank RTRL / Established direction | 把 pre/post trace 因子化为线性神经元内存 | 高 | 中低；先实现 IC0 单层特例 | 2026 原论文给出模型无关编译器与多任务结果 | 可支持后续真正 recurrent E/I SNN | 近似/编译器复杂度高，迁移成本最大 |

推荐并首先执行 EL1：它把 EL0 的“terminal-only”边界推进到 action/observation 多边界，同时仍能用普通 IC0-BPTT 给出逐元素真值。第二顺位是 blockwise exact 路线，用于 EL1 在 K/T 过密时的退化区；e-prop、OTTT、SLTT-K、NDOT、BrainTrace 后续分别预注册，不能混为一个调参池。

**What if：**如果世界模型真正决定行为的监督主要位于动作提交、下一观察、reward/done 与 rollout 检查点，而不是每个文本子 token，那么 K-query exact eligibility 可能覆盖主要信用分配，同时避开 dense BPTT 的 T 倍保存量。EL1 必须先测 K 密度曲线；若 K/T 很快抹平收益，这个设想即被否证。

### 数学构造

保持 IC0 forward 不变。对任一 E/I population，`Q_t=u_0+sum_{k<=t}q_k`、`s_t=floor(Q_t)-floor(Q_{t-1})`、`u_t=Q_t-stopgrad(floor(Q_t))`。监督索引为严格递增的 `tau_j`，`j=1..K`；event/floor surrogate 分别记为 `phi(d)`、`psi(Q)`，query 上游 spike/residual learning signal 为 `g^s_j/g^u_j`：

`A_j = g^u_j + g^s_j*psi(Q_{tau_j})`；

`B_j = g^s_j*psi(Q_{tau_j-1})`（`tau_j=0` 时为 0）；

`E_t = sum_{k<=t} phi(d_k) outer x_k`；

`grad_W = eta * sum_batch,j [A_j*E_{tau_j} - B_j*E_{tau_j-1}]`。

若 loss 也读取最终 recurrent state，再加 `g_state*E_{T-1}`；bias eligibility 把 `x_k` 换成 1，初态梯度为 `sum_j(A_j-B_j)+g_state`。forward 按 query 分段累计 eligibility，每个时间点只进入一次 einsum，保存 K 个 prefix snapshot 而不是完整 `[B,T,H,D]`。因此冻结输入的 backward 保存量为 `O(KBHD)`、对 T 常数；输入需要梯度时仍保存逐时刻 `phi(d_t)` 并由 query range coefficient 精确恢复 `grad_x`，该模式为 `O(BTH)`，必须单列。

### 冻结等价、内存与速度门

- **H-EL1-EQ**：覆盖 `(B,T,K,input_grad)=(1,1,1,on),(2,32,4,on),(1,512,8,off)`，含外部初态、首/中/末 query。query output/final state、`x/u0/input_to_{e,i}`、output norm/projection 全参数梯度对普通 IC0 scan 满足 `atol=2e-6,rtol=1e-5`；state/spike forward bit-exact。query 索引必须是一维、非空、strictly increasing、无重复且位于 `[0,T)`，错误输入必须拒绝。
- **H-EL1-MEM**：用 `saved_tensors_hooks` 测 `B=1,D=32,K=4,T in {128,512,2048}`。core-only 的 T=2048 unique saved bytes 必须不超过普通 IC0 K-query BPTT 的 25%，且 EL1 T=128→2048 增长不超过 1.25x；input-gradient 单列。另扫 `K={1,4,16,32}`、T=512，报告 K 线性成本与相对 BPTT break-even，不用 K=1 代替 K=4 过门。
- **H-EL1-SPEED**：`B=1,D=32,K=4,T in {512,2048}`、CPU threads 1/4/16，query loss 的 forward+backward 交错计时。至少一档 EL1 p50 比普通 IC0-BPTT 快 `>=1.25x`，且绝对不慢于同任务 fused LSTM；Transformer 为第二 ANN 对照。参数、query 索引、输入与 loss 完全一致，不从不同线程拼接。
- 本机 CUDA unavailable 时 GPU 指标为 null；runner 保留 device/synchronize 路径。多核只按同线程比较，不以 16-thread 绝对时间替 1-thread baseline。

### 冻结质量任务：四段 action-like WRITE/QUERY register

- `T=32,K=4,D=32`。四段起点为 `{0,8,16,24}`：起点 token 是独立均匀的 `WRITE(payload in 0..15)`，四步后的 `{4,12,20,28}` 为共同 `QUERY`，其余为共同 distractor；每个 query 预测本段 payload。随机 payload 防止靠位置猜标签，四步 delay 模拟动作写入后在观察边界读取短期 latent state。
- 输入词表为 distractor、16 个 WRITE、QUERY，共 18 个 token；所有模型共享冻结的 32 维正交 one-hot code，不训练 embedding。共享 output LayerNorm/decoder 初始化；trainable 总参数相对 LSTM 必须在 `±2%`。这只消除 EL0 随机 embedding 的任务仪器风险，不给任何模型额外未来信息。
- seeds `{0,1,2}`；每 seed 600 update、batch 32、每 update 数据 seed 固定为 `8_930_000 + 10_000*seed + update`；AdamW `lr=1e-3,weight_decay=0.01`、clip 1.0。test 使用独立 seed、4096 条序列，分块评估；所有 query 共 16,384 个标签。
- **任务有效门**：每个 seed 的 LSTM 与 Transformer test query accuracy 都须 `>=99%`，否则 H-QUALITY=`INVALID`，不得事后加 update/改 embedding/降门。
- **H-EL1-QUALITY**：任务有效后，每 seed 普通 IC0-BPTT 与 EL1 都须 `>=99%`；两者逐 update loss 最大差、最终参数最大差均 `<=1e-4`。同时报告 worst-seed NLL、训练 p50、spike/event 二值率。失败时区分“IC0 表示质量不足”与“EL1 梯度不等价”。
- 只有 H-EQ/MEM/SPEED/QUALITY 全通过才运行独立的真实 TextWorld action/observation-boundary K-query 实验；本任务本身不是语言模型或世界模型成功证据。

### 来源边界与产物

- e-prop 把梯度写成 learning signal × local eligibility，并明确说明实际 online learning signal 会忽略经其他 recurrent neurons 传播的未来影响：<https://www.nature.com/articles/s41467-020-17236-y>。
- OTTT 提供 online-through-time 参照：<https://arxiv.org/abs/2210.04195>。SLTT-K 随机选择 K 个反传时刻并报告内存对总时间步常数，但实验主要是静态/事件视觉分类，不是 action-conditioned LM：<https://openaccess.thecvf.com/content/ICCV2023/papers/Meng_Towards_Memory-_and_Time-Efficient_Backpropagation_for_Training_Spiking_Neural_Networks_ICCV_2023_paper.pdf>。
- NDOT 用 neuronal dynamics sensitivity 分解 temporal/spatial gradient：<https://proceedings.mlr.press/v235/jiang24a.html>。BrainTrace/pp-prop 把 RTRL eligibility 近似成 pre/post 因子并提供线性 neuron-memory 编译路径：<https://www.nature.com/articles/s41467-026-68453-w>。这些来源都没有证明本项目 exact modulo IC0 的 K-query 闭式；EL1 的“精确”只由上述推导与本地全梯度等价判官支持。
- 正式 runner/产物冻结为 `experiments/e3_el1_multi_query_eligibility.py` 与 `results/e3_scan/e3_el1_multi_query_eligibility.json`；任何 smoke 产物不得覆盖正式 JSON。

---

## 2026-07-18：E3-EK0 结果 — exact tensor step 过单流实时门，编译覆盖仍有限（正面结果）

### 证据 / 实现

- 正式命令：`.venv-wsl/bin/python experiments/e3_ek0_compiled_streaming.py --output results/e3_scan/e3_ek0_compiled_streaming.json`；产物 SHA-256 `8411FAE1DFE2B396C4669CB7D58C002AA51421A39031F6A1775A8D43E4EF997D`。环境为 PyTorch `2.13.0+cpu`、Ryzen 9 7950X、CUDA unavailable。
- 新增 `E3InputCodedScanCore.forward_step_tensors`：直接对 E/I tensor state 执行 `event=(Wx+b>=0)`、`pre=u+0.125+0.75*event`、`spike=floor(pre)`、`u'=pre-spike`，随后复用原 LayerNorm/projection。它不改变 IC0 参数或动力学，只移除 T=1 sequence、`cumsum/cat`、dataclass 与重复 validation。
- 首轮 smoke 暴露 `torch.compile` 的 tensor dispatch key 变化：普通 tensor 首次编译后再输入 inference tensor 会触发重编译。正式运行前已统一在 `inference_mode` 内创建 token/state；这是执行接口修复，不改方程、门槛或正式数据。
- 完整回归 `.venv-wsl/bin/python -m pytest -q` 为 `118 passed, 46 subtests passed`；`compileall -q vpsc experiments tests` 与 `git diff --check` 通过。WSL venv 未安装 Ruff（`No module named ruff`），不把缺失的 lint 运行伪装成通过。

### 等价门

- `B=1/8,T=512` 全通过；tensor-eager 对 generic 的 sequence/state/spike 为 bit-exact。compiled sequence 最大绝对误差分别为 `4.7684e-7/1.4305e-6`，末态与 spike bit-exact，spike 全为 0/1、residual 全在 `[0,1)`。
- `fullgraph=True,mode=reduce-overhead` 在等价阶段成功；fresh B1 首次调用为 `61.23 s`，B8 为 `1.84 s`，均明确排除在稳态延迟之外。**H-EK0-EQ PASS**。

### 实时逐 token 结果

共同 64 warmup + 512 measured token，逐模型交错计时；数值为 p50 / p95 ms：

| threads / batch | generic IC0 | tensor eager | tensor compiled | fused eager LSTM | compiled 判定 |
|---|---:|---:|---:|---:|---|
| 1 / 1 | 0.1091 / 0.1467 | **0.0545 / 0.0888** | 0.0698 / 0.0985 | 0.0707 / **0.0972** | FAIL（p95 慢 1.3%） |
| 1 / 8 | 0.1211 / 0.1676 | **0.0618 / 0.1021** | 0.0787 / 0.1116 | 0.0736 / **0.0981** | FAIL |
| 4 / 1 | 0.1102 / 0.1492 | **0.0554 / 0.0729** | **0.0671 / 0.0877** | 0.0754 / 0.1018 | **PASS** |
| 4 / 8 | 0.1299 / 0.1848 | **0.0698 / 0.1121** | 0.0833 / 0.1278 | 0.0781 / **0.1048** | FAIL |
| 16 / 1,8 | — | — | NOT RUN | — | `FailOnRecompileLimitHit` |

- 4-thread B1 同一 lane 内，compiled IC0 比 LSTM 快 `11.1%` p50、`13.8%` p95，故按“至少一档同时不慢于 LSTM”的冻结规则 **H-EK0-RT PASS**。不能用 1-thread p50 与 4-thread p95 拼门；表中没有这样做。
- generic→tensor-eager p50 加速为 `1.86–2.00x`。compiled 在所有成功 lane 都比 tensor-eager 慢，说明主要收益来自精确单步代数与低开销 tensor 接口，不是编译器；不过 4-thread B1 compiled 仍独立过了预注册门。
- 16-thread 在按线程数产生第 9 个 Dynamo specialization 时达到 cache/recompile 上限，B1/B8 都未测；B8 在 1/4 线程也慢于 LSTM。IC0 参数/state 为 `8516 / 336 bytes`，LSTM 为 `8448 / 256 bytes`，状态开销仍多 31.25%。

### 判定 / 下一步

- **H-EK0-EQ PASS；H-EK0-RT PASS；overall=PASS。** 这是 IC0 首次在 exact streaming 语义下以同线程 p50+p95 越过 fused LSTM；结论只覆盖 4-thread、B1、CPU steady-state，不外推到 batch throughput、冷启动、GPU、真实 LM 或完整世界模型。
- 当前工程默认候选应优先保留 tensor-eager 路径；`torch.compile` 作为可选缓存层，需先解决线程数导致的 specialization 上限与约一分钟 fresh compile 成本。下一实现把 dispatch-free tensor state 接到完整 transition runtime，并引入真正的 event-skip/segment kernel，而不是继续把 compile 当作主要数学收益。
- 训练侧与 EL0 结论并列：EL0 已解决 terminal/query-sparse exact backward，但真实任务质量门仍无效。下一轮必须先设计 ANN controls 可学的 EL1 短延迟/K-query，再进入真实 TextWorld/action-conditioned 任务；EK0 不能替代该质量证据。

---

## 2026-07-18：E3-EK0 预注册 — exact tensor streaming + compiled fusion（已执行）

### 动机 / 构造

IC0 已有 T=512 streaming p95 `0.186/0.168/0.192 ms`，仍慢于 LSTM `0.114/0.125/0.163 ms`；但 generic `step()` 实际把单 token 扩成 T=1 sequence，再走 `cumsum/cat/dataclass/validation`，没有使用 IC0 在单步上的最简闭式。

EK0 不改任何权重或动力学。对每个 E/I population 直接计算：`z=1[Wx+b>=0]`，`p=u+0.125+0.75z`，`s=floor(p)`，`u'=p-s`。因 `u in [0,1)` 且单步 charge `<1`，`s` 严格为 0/1，等价于 generic cumulative floor/difference/modulo。readout 仍为原 `LayerNorm([s_E,-s_I,u'_E,-u'_I]) + Linear`。

实现分三层比较：现有 dataclass generic step；只收发 tensor tuple 的 eager exact step；同一 tensor module 经 `torch.compile(fullgraph=True,mode="reduce-overhead")` 的 compiled step。LSTM 使用当前 PyTorch oneDNN/fused `nn.LSTM` eager step 作为 ANN 下界；若 `nn.LSTM` 自身不能被 fullgraph compile，不以较慢的手写 LSTMCell 替换它来制造胜利。

### 冻结门

- **H-EK0-EQ**：`B in {1,4}`、连续 T=512、随机外部 residual；generic/tensor-eager/compiled 的逐 token sequence 通过 `atol=2e-6,rtol=1e-5`，最终 E/I state bit-exact，全部 spike 仅 0/1、residual 始终 `[0,1)`。compiled 必须 `fullgraph=True` 成功；编译时间单列，不计稳态延迟。
- **H-EK0-RT**：CPU threads 1/4/16，`B=1,D=32`，共同 64 warmup + 512 measured token，逐 token交错顺序，报告 p50/p95/p99。至少一档 compiled IC0 的 p50 和 p95 都不慢于 fused eager LSTM，才过实时门；不能从不同线程拼接。
- 另报 generic→tensor-eager 与 tensor-eager→compiled speedup、persistent state bytes、参数量和 batch 8 throughput；质量不重训，因为 EQ 要求 forward 完全相同。若只 p50 赢或尾延迟排序不稳，RT FAIL。
- 本机 CUDA unavailable 只报 CPU；runner 保留 device-aware synchronize。EK0 通过也只解决 IC0 推理核，必须与 EL0 的 query-sparse 训练边界并列，不得宣称 dense LM/世界模型已经达到 ANN 训推速度。
- 正式产物冻结为 `results/e3_scan/e3_ek0_compiled_streaming.json`。

---

## 2026-07-18：E3-EL0 结果 — 终端 eligibility 在长序列同时过等价/内存/速度门，质量任务无效（混合结果）

### 证据 / 实现

- 正式命令：`.venv-wsl/bin/python experiments/e3_el0_terminal_eligibility.py --output results/e3_scan/e3_el0_terminal_eligibility.json`；产物 SHA-256 `0969162D20F7AE8D877A16FE298D7D348D8F9131821B2E68D334C147B7BA1BC2`。环境为 PyTorch `2.13.0+cpu`、Ryzen 9 7950X、CUDA unavailable。
- 新增 `E3InputCodedScanCore.forward_terminal_eligibility`：forward 仍是 IC0 的 strict binary event / exact hard modulo reset；custom backward 只为 terminal sparse loss 保存 `E_T/E_{T-1}` 与末两步 learning signal，输入需要梯度时再保存逐时刻 event derivative。
- 首轮 smoke 暴露一个真实内存陷阱：保存 `Q_T/Q_{T-1}` 的 view 会 pin 完整 `[B,T,H]` cumulative storage。结果前已把四个末端 view clone；修复后 logical/unique saved bytes 一致，core-only 不再随 T 增长。这是实现修复，不改变公式、门槛或正式数据。

### 等价与 backward 保存量

- `(B,T,input_grad)=(1,1,on),(4,32,on),(1,512,off)` 三个 case 全通过；terminal sequence/state、输入/初态/全部参数 gradient 均满足冻结 `atol=2e-6,rtol=1e-5`。最大原始绝对误差 `1.1444e-5` 出现在 T=512 inhibitory bias gradient，因对应量级满足 relative tolerance；300-update 轨迹提供了更强的累计核验（见质量段）。**H-EL0-EQ PASS**。

| mode | T=128 saved bytes / nodes | T=512 saved bytes / nodes | T=512 vs BPTT |
|---|---:|---:|---:|
| IC0 terminal-BPTT, frozen input | 314,688 / 43 | 1,190,208 / 43 | 100% |
| EL0 core-only, frozen input | **57,928 / 13** | **57,928 / 13** | **4.87%** |
| IC0 BPTT, input grad | 325,440 / 46 | 1,200,960 / 46 | 100% |
| EL0, input grad | 100,936 / 14 | 229,960 / 14 | 19.15% |

core-only 的 T=128→512 growth=`1.0×`，T=512 ratio=`4.87%<25%`；**H-EL0-MEM PASS**。需要 encoder/input gradient 时保存量仍随 T 线性增长，不能冒充完全常数内存，但时间 autograd node 仍被压到 14。

### 终端训练速度

`B=1,D=32`，p50 ms；loss 只读取 terminal output，所有核心输入冻结，参数量 IC0/EL0/LSTM/Transformer=`8516/8516/8448/8608`。

| threads | T | EL0 | IC0 BPTT | speedup | LSTM | Transformer |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 512 | **0.82** | 1.13 | 1.38x | 1.13 | 6.86 |
| 1 | 2048 | **1.62** | 2.84 | **1.75x** | 3.51 | 92.55 |
| 4 | 512 | **0.74** | 1.02 | 1.38x | 1.30 | 2.57 |
| 4 | 2048 | **1.43** | 2.00 | 1.40x | 4.45 | 25.54 |
| 16 | 512 | **1.23** | 1.69 | 1.37x | 2.51 | 3.16 |
| 16 | 2048 | **2.06** | 3.69 | **1.79x** | 9.27 | 29.28 |

T=512 三档都已快于 LSTM，但没有达到相对 IC0 的 1.5x 门；T=2048 在 1/16 线程同时达到 `>=1.5x` 且快于 LSTM，故按预注册 **H-EL0-SPEED PASS**。4 线程 T=2048 虽绝对最快，也只达到 1.40x，未拼接为该子门的胜项。

### 质量判官无效，但训练轨迹验证闭式梯度

冻结 terminal-delay 任务 300 update 后：IC0-BPTT/EL0 accuracy 都为 `93.75%`、NLL 都为 `0.81537`；LSTM 仅 `6.25%`，Transformer `43.75%`。两个 ANN control 都未到预注册 99%，故 **H-EL0-QUALITY INVALID**，不是 EL0 FAIL/PASS，也不运行 TextWorld K-query。

尽管任务无效，EL0 与普通 IC0 的数值对应是强实现证据：300 次逐 update loss 最大差 `4.7684e-7`，最终全部参数最大差 `3.9116e-7`，test logits 导出的 accuracy/NLL 完全相同。完整训练 p50 为 EL0 `1.38 ms`、IC0-BPTT `1.71 ms`、LSTM `1.70 ms`、Transformer `2.08 ms`；由于 control 质量失败，这些 task-level 时间不能升级为质量约束下的 ANN 胜利。

### 判定 / 下一步

- **H-EL0-EQ PASS；H-EL0-MEM PASS；H-EL0-SPEED PASS；H-EL0-QUALITY INVALID；overall=MIXED。** EL0 是目前第一个在 T=2048 同时保持 strict IC0 forward、精确 surrogate gradient、常数 core-only backward 保存量并超过 LSTM terminal training 的数学加速方案。
- `forward_terminal_eligibility` 保留为实验 API，不设为 dense LM 默认：它只覆盖 terminal/query-sparse objective；输入梯度模式也不是常数内存。LSTM/Transformer control 失败意味着尚不能把它接到真实 TextWorld 后声称质量成立。
- 下一步独立执行 EK0 exact compiled streaming step，解决 IC0 已知的实时单步负债；另行预注册一个 ANN controls 可学的 EL1 短延迟/分块 K-query 预算，不能在本轮事后增加 update 或降低 99% 门。

---

## 2026-07-18：E3-EL0 预注册 — exact terminal eligibility scan（已执行）

### 路线比较与选择

当前瓶颈被拆成“时间信用分配”和“单步执行核”两个独立问题。原始资料支持的候选并不等价：

| 路线 | 认识论标签 | 最小机制 | 本项目主要风险 |
|---|---|---|---|
| e-prop 三因子规则 | Established direction | learning signal × synaptic eligibility | recurrent loop 下局部 trace 不是完整梯度，状态量可随连接数膨胀 |
| OTTT presynaptic trace | Established direction | detach reset 后递推突触前活动，以即时 loss 独立求梯度 | 假设即时监督；对延迟 query 的学习信号需要额外 eligibility |
| SLTT / SLTT-K | Established direction | 删除 temporal-gradient route，并只在 K 个时刻反传 | 长依赖任务上被删掉的正是目标信用路径 |
| NDOT dynamics sensitivity | Established direction | 用神经动力学敏感度近似在线时间梯度 | 需要为本项目 modulo reset 重新推导，近似误差尚不清楚 |
| BrainTrace factorised trace | Established direction | pre/post trace 因子化，把在线学习内存降到线性 | 2026 方法与编译器迁移成本高，先做最小代数特例才能定位收益来源 |
| **EL0 additive exact eligibility** | **Speculative new specialisation** | 利用 IC0 的加性 cumulative charge，把 terminal surrogate gradient 闭式分解为 learning signal × 累积 eligibility | 只天然覆盖稀疏/终端监督；dense LM 仍需分块或 K-query 扩展 |
| exact sparse-event kernel | Established engineering direction | 事件段 associative scan + 融合 step kernel | 只解决执行，不会自动修复信用分配或质量 |

优先执行 EL0：它是对当前 IC0 方程的可证明特例，不依赖“近似梯度应该够用”的希望；若梯度等价而速度/内存仍失败，就能直接排除这类因子化在当前 PyTorch/CPU 上的工程价值。exact sparse-event/compiled step 作为随后独立 EK0，不用其推理收益替 EL0 训练门过关。

### 数学构造

对单个 E/I population，IC0 保持原 forward：`d_t=Wx_t+b`，`z_t=H(d_t)`，`q_t=q_base+eta*z_t`，`Q_t=u_0+sum_{k<=t}q_k`，`C_t=floor(Q_t)`，`s_t=C_t-C_{t-1}`，`u_t=Q_t-stopgrad(C_t)`。event surrogate 记为 `phi(d)`，periodic-floor surrogate 记为 `psi(Q)`。

若 loss 只读取末时刻 `[s_T,u_T]`，令上游 learning signal 为 `(g_s,g_u)`，则冻结 surrogate 语义下：

`a_T = g_u + g_s*psi(Q_T)`；

`E_T = sum_{k<=T} phi(d_k) outer x_k`，`E_{T-1}` 同理；

`grad_W = eta * sum_batch [a_T*E_T - g_s*psi(Q_{T-1})*E_{T-1}]`。

bias eligibility 把 `x_k` 换成 1；初态梯度使用同一系数。对需要 encoder/input gradient 的模式，保存逐时刻 `phi(d_t)` 并一次矩阵乘得到精确 `grad_x`；对冻结输入的 core-only 模式，只保存聚合 `E_T/E_{T-1}`，反向保存量应与 T 无关。forward 仍是严格 0/1 input event、0/1 output spike、hard modulo reset；EL0 只改训练反向图，不改推理模型。

### 冻结门与边界

- **H-EL0-EQ**：覆盖 `(B,T)=(1,1),(4,32),(1,512)`、外部初态以及 input-gradient 开/关；terminal sequence/state、`x/u0/input_to_{e,i}` 与 output norm/projection 全参数梯度相对普通 IC0 scan 通过 `atol=2e-6,rtol=1e-5`。任一不等价即停止速度/质量解释。
- **H-EL0-MEM**：用 `saved_tensors_hooks` 统计真实 backward-saved tensor bytes。core-only 在 T=512 必须不超过普通 IC0 terminal-BPTT 的 25%，且 T=128→512 增长不超过 1.25×；full-input-gradient 单独报告，禁止把其线性输入保存冒充常数内存。
- **H-EL0-SPEED**：`B=1,D=32,T in {512,2048}`、threads 1/4/16，terminal forward+backward 交错计时。EL0 至少一档须比普通 IC0 terminal-BPTT 快 1.5×，并且绝对 p50 不慢于同任务 LSTM；Transformer 保留为第二 ANN 对照。
- **H-EL0-QUALITY**：冻结 embedding 的 16 类 terminal delayed-token task：payload 只出现在 `t=0` 的 16 种 WRITE token，中间全为同一 distractor，`t=T-1` 为所有样本相同的 QUERY；`T=64,B=8`，embedding 用 seed `8700001` 的 `Normal(0,0.2)` 后冻结，train batch seed `8710000+update`，test 穷举 16 类。训练 300 update、AdamW `lr=1e-3,weight_decay=0.01`、clip 1.0，相同 batch/初始化；LSTM/Transformer test accuracy 均须>=99% 才验证任务。EL0 与普通 IC0 都须>=99%，且逐 update loss 与最终参数最大差不超过 `1e-4`；否则分别记录架构质量失败或梯度实现失败。
- EL0 PASS 只证明**稀疏终端监督**的精确 eligibility 训法。它不等于 dense next-token LM、一般 recurrent e-prop、完全在线参数更新或世界模型成功；通过后才扩展为 chunked K-query 并接真实 TextWorld event next-token，仍需同时比较 LSTM/Transformer。
- 正式产物冻结为 `results/e3_scan/e3_el0_terminal_eligibility.json`；本机 CUDA unavailable 时 GPU 指标必须为 null，runner 保留 device-aware 路径但不借文献 GPU 数字过门。

### 来源边界

- e-prop 将权重更新分解为 learning signal 与 eligibility trace；对 recurrent loop 的局部计算边界见 Nature Communications 2020：<https://www.nature.com/articles/s41467-020-17236-y>。
- OTTT 在 detach-reset 语义下递推 presynaptic activity 并形成三因子在线梯度：<https://arxiv.org/abs/2210.04195>。
- SLTT/SLTT-K 删除 temporal gradient route，并把 backward 时刻从 T 降到 K：<https://openaccess.thecvf.com/content/ICCV2023/papers/Meng_Towards_Memory-_and_Time-Efficient_Backpropagation_for_Training_Spiking_Neural_Networks_ICCV_2023_paper.pdf>。
- NDOT 与 BrainTrace 分别提供 dynamics-sensitivity 与 factorised linear-memory 参照：<https://proceedings.mlr.press/v235/jiang24a.html>、<https://www.nature.com/articles/s41467-026-68453-w>。这些来源没有测试本项目 exact modulo IC0；EL0 的闭式等价仍须由本地梯度矩阵证明。

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
