# ALICE 设计与实现记录（ADR 源码核对版）

> 更新日期：2026-11-05。
> 本文保留 ADR-001～009 编号，用于解释公开实现；它不是作者当时的决策会议记录。
> “可比较方案”是阅读者整理的设计空间，不能据此认定作者实际尝试或否决过这些方案。
> 代码可确认的事实、论文报告的事实和分析性判断分别说明；未验证的效果或因果不作为结论。

核对依据为本仓库 `ALICE/` 中对应上游版本 `e91627ff7fdb9df6637fa1047e8f9f7c9bc51141` 的源码、随库数据、[论文补充材料](ALICE/reproduction/references/Document_S1.pdf)和[本机复现报告](ALICE/复现报告.md)。以下相对链接从本仓库根目录解析；PDF 页码为文件的 1-based 页码。

本次已经执行的是作者公开输入和已训练模型的筛选、推理与统计复算。下文对生成、预训练和进化代码的检查属于源码核对，不代表这些模块已从头训练通过。

## 模块关系

```text
原版 ALICE：
  7-mer 设计范围与功能标签（001、002）
    ├─ 序列模型预训练和生成（007）
    ├─ 描述符与功能预测器（003、006）：用于前置筛选和末端评估
    └─ 参考集对比评分（004）＋启发式搜索（005）
         → 最终八步筛选（008）

ALICE-X：
  另有生成训练入口和 AutoGluon 预测器（006、007）
    → Stage 1 参考评分 / Stage 2 预测与距离评分（009）
```

这表示模块之间的计算关系。代码没有证明“选定参考评分就必须使用某一种搜索算法”，也没有证明“删除已报道序列直接促成了 Stage 2”。

## ADR-001　以 7-mer 插入序列作为主要设计变量

**背景**

论文补充方法说明，原版数据来自在 AAV9 VP1 的 588–589 位之间插入 7 个氨基酸的文库；设计和评估均依赖这一骨架、插入位置及实验背景。[补充材料，第 3–4 页](ALICE/reproduction/references/Document_S1.pdf#page=3)

**可比较方案**

可考虑全长序列修改、选定位点突变、固定位置插入短肽等方案。公开代码实现了固定长度短序列的主要流程；不能由此推断其他方案必然失败，或作者逐一排除过它们。

**实现事实**

- 搜索字母表是 20 种氨基酸，7-mer 理论组合数为 `20^7 = 1,280,000,000`。
- 长度分别出现在[模型参数](ALICE/alice/ALICE/tools/parameters.py#L83)、[语义训练入口](ALICE/alice/ALICE/step2_semantic_tuning/Train_SeqGAN.py#L282)和[进化入口](ALICE/alice/ALICE/step4_function_guided_evolution/function_guided_evolution.py#L68)。
- 部分值是命名配置字段，部分是脚本内常量；不存在统一控制全部模块长度的单一配置项。

**代价与边界**

固定局部设计范围降低了组合空间，但不能保证任意 7-mer 都可正常组装或达到目标功能。功能还取决于骨架、插入位置和实验条件；改变这些条件后需要重新验证数据和模型的适用性，不能直接断言全部知识必然作废，也不能保证模型可直接迁移。

## ADR-002　以生产适应度和受体相关富集值作为功能标签

**背景**

原版包含生产适应度、Ly6a 和 Ly6c1 三个预测任务；ALICE-X 的相关数据与预测器使用生产适应度和 hTfR1 两个任务。不能把二者的数据规模、靶标或测量列混写。

**可比较方案**

可使用实验富集标签、结构计算评分或更接近最终应用的测量。各方案的成本、误差和适用范围需要具体证据，不能从当前实现推出某一类标签必然无效。

**实现事实**

ALICE-X 的[语义数据表](ALICE/alice_x/finetuning/data/semantic_tuning_data.csv)包含 `plasmid_mean`、`virus_mean`、`hutfr1_mean` 及相应的富集和 log2 富集列。其标签可按下式理解，具体计数归一化与预处理仍应遵循实验方法：

```text
生产适应度：log2(virus / plasmid)
受体富集：  log2(target / virus)
```

这是两个对数比值；共享 `virus` 项可能带来相关误差，不能笼统称为“两个比值都是比值的比值”。log2 使互为倒数的倍数变化在数值上对称。

[ALICE-X 预处理脚本](ALICE/alice_x/ranking/s1_data_preprocessing.py#L87)会处理标签缺失、按序列去重，并可过滤含 `_` 的序列；[训练入口](ALICE/alice_x/ranking/s2_auto_gb_regressor.py#L122)还处理非数值列、无穷值和缺失行。因此“仓库只过滤含 `_` 的行”不准确。

**代价与边界**

这些标签是实验条件下的功能代理量，不能直接等同于疗效。低计数、归一化方式和批次效应可能影响可靠性，但单个低计数比值不能直接判定为纯噪声。上述特征预处理入口没有显式的原始计数下限；这不足以证明上游实验数据没有做质量控制。论文补充方法明确描述了低质量、异常或冗余数据处理。[补充材料，第 4 页](ALICE/reproduction/references/Document_S1.pdf#page=4)

## ADR-003　采用混合流程：序列模型生成、描述符回归和参考集搜索

**背景**

ALICE 的生成器、功能预测器和进化评分承担不同任务。把整个项目概括为“描述符＋传统回归，不使用深度学习”会遗漏预训练 Transformer 和生成网络。

**可比较方案**

功能预测可以比较 one-hot、语言模型表征或理化描述符；搜索可以比较结构评分、学习式代理评分和参考序列评分。这些选项并不互斥。

**实现事实**

- [预训练模块](ALICE/alice/ALICE/step1_pretrain/minibert.py#L56)实现 Transformer 编码器；[生成器](ALICE/alice/ALICE/tools/SeqGAN.py#L14)将编码器表征接入后续网络。
- [功能评估器](ALICE/alice/ALICE/evaluation)使用描述符和多种回归算法。
- [原版 FE](ALICE/alice/ALICE/step4_function_guided_evolution/evaluation.py#L157)主要通过正负参考集的比对评分引导搜索。
- 当前这些入口没有把三维结构预测和分子对接作为候选逐条评分步骤。

**代价与边界**

描述符与经验评分可能降低建模成本，但仍有信息丢失和分布变化风险。不能把“柔性环没有稳定构象”“端到端模型必然过拟合”“参考评分能够补偿所有外推问题”写成已证实结论。

原版论文所述生产适应度及两种受体任务分别有 119,851、89,169、89,040 个数据点；ALICE-X 随库数据是另一组任务。因此用“全项目只有约 6000 条标签”推导技术路线不成立。[补充材料，第 3 页](ALICE/reproduction/references/Document_S1.pdf#page=3)

## ADR-004　原版 FE 使用正负参考集的序列比对分差

**背景**

原版 FE 需要对候选给出搜索分数。该分数与回归器预测的功能值属于不同指标；前置筛选、末端筛选和 ALICE-X Stage 2 仍会使用预测器。

**可比较方案**

可使用回归预测、序列参考评分、结构评分或组合目标。当前原版 FE 实现了参考评分；选择它不能保证免受分布外风险影响。

**实现事实**

[MSAEva.evaluate](ALICE/alice/ALICE/step4_function_guided_evolution/evaluation.py#L157)的分数为：

```text
FE(s) = 0.5 + [M_p({SW(s,r): r∈Ref+}) − M_p({SW(s,r): r∈Ref−})]
                  / [2 × len(s) × (u − l)]

M_p(values) = (mean(value^p))^(1/p)
默认 p=1，因此默认是算术平均；p=∞ 时代码取最大绝对值。
l = extend；u = 替换矩阵对角元素的平均值。
```

`M_p` 是[广义均值](ALICE/alice/ALICE/step4_function_guided_evolution/basic.py#L42)，不是直接取未经样本数归一化的 p-范数。`MSAEva` 名称虽含 MSA，实际是候选与每条参考序列进行成对比对。

原版返回分数时没有 `clamp`；上述缩放公式本身不足以保证任意输入、矩阵和参数下的值都落在 `[0,1]`。

- [参考 JSON](ALICE/alice/ALICE/step4_function_guided_evolution/seq_LY6A_LY6C1.json)含 **76 条正参考、53 条负参考**。
- [BLOSSUM62.txt](ALICE/alice/ALICE/step4_function_guided_evolution/BLOSSUM62.txt)实际为 **23×23**，包含 B、Z、X，不能写成 20×20。搜索字母表仍是 20 个字符。
- [比对实现](ALICE/alice/ALICE/step4_function_guided_evolution/evaluation.py#L108)默认局部模式，`gap=-4`、`extend=-2`；单次比对的主体是两个位置循环。它通过保存胜出方向处理延伸罚分。
- [ALICE-X 实现](ALICE/alice_x/FE/src/evaluation.py#L40)使用 Numba、缓存及独立的 gap 状态；这是同类评分思路，不能未经对照就宣称与旧实现逐点等价。

**代价与边界**

分数依赖参考集、替换矩阵与参数，不能保证新候选具有相应实验功能，也不能自动保证多功能融合。接近正参考、远离负参考是一种序列层面的偏好，仍需要独立验证。

论文已说明参考集筛选原则，Table S2 给出名单和来源；所以“完全没有记录参考集怎么挑”不成立。代码 JSON 本身未提供完整的从原始实验数据自动重建参考集的流程，这才是需要区分的复现边界。[补充方法，第 4 页](ALICE/reproduction/references/Document_S1.pdf#page=4) · [Table S2，第 45 页](ALICE/reproduction/references/Document_S1.pdf#page=45)

原版末端确实排除三条指定已报道序列，但这只证明存在排除规则；不能单凭它认定搜索必然复制参考集，或推导 ALICE-X Stage 2 的历史开发动机。

## ADR-005　提供 GA、EDA、SA 和 EDG 搜索实现，原版默认 EDG

**背景**

候选是离散序列，原版参考评分未提供可供普通反向传播直接使用的梯度。启发式搜索与这一实现相适配，但并不意味着策略梯度、离散松弛等其他搜索方式在原则上不可用。

**可比较方案**

全枚举、随机搜索、遗传算法、分布估计算法、模拟退火及其组合均可作为比较对象。其效率和质量需要按预算与实验验证，不宜写成“随机采样等于无效”或“混合必然更优”。

**实现事实**

[原版入口](ALICE/alice/ALICE/step4_function_guided_evolution/function_guided_evolution.py#L17)通过 `--group_type` 选择算子，默认 `EDG`；循环执行 11 次进化，没有在该入口实现收敛停止条件。

| 实现 | 默认值或行为 |
|---|---|
| [GA](ALICE/alice/ALICE/step4_function_guided_evolution/GA.py#L31) | 逐位变异概率 0.3、保留率 0.2，默认单点交叉，支持切换交叉策略 |
| [EDA](ALICE/alice/ALICE/step4_function_guided_evolution/EDA.py#L9) | `rbest=0.5`、`alpha=0.6`，根据优选个体估计逐位分布 |
| [SA](ALICE/alice/ALICE/step4_function_guided_evolution/SA.py#L13) | 默认 `t0=1.6`，实现不同降温方式 |
| [EDG](ALICE/alice/ALICE/step4_function_guided_evolution/mixture.py#L20) | `r_eda=0.3`、`r_ga_copy=0.2`，同时把 `alpha` 设为 1 |

按 EDG 的种群划分公式，若总数为 1000，则为 **300 个 EDA 采样、560 个 GA 交叉变异产物、140 个直接保留个体**。笼统称为“70% GA 后代＋30% EDA”会遗漏 GA 部分内部的精英保留。

论文补充 Figure S8 已比较 SA、GA、EDA、EDG 的 FE 轨迹和功能分布，所以“没有算法比较”不成立。当前代码的默认值不必然覆盖论文每一组实验配置；本次也没有重新执行这些算法比较。[补充材料，第 73 页](ALICE/reproduction/references/Document_S1.pdf#page=73)

**代价与已确认的实现问题**

这些参数有构造参数、setter 和部分命令行入口，并非全部无法配置；但分散在多个模块中，不能仅凭默认值认定它们经过最优调参。`alpha=1` 表示不保留旧分布的平滑贡献，其设计动机不能由常量本身确定。

[basic.py](ALICE/alice/ALICE/step4_function_guided_evolution/basic.py#L1)先导入 NumPy 的 `random` 函数，后又以同名导入标准库模块；[weighted_choice](ALICE/alice/ALICE/step4_function_guided_evolution/basic.py#L65)调用 `random()` 时会触发模块不可调用的错误，影响使用它的采样路径。

[Group.calcDis](ALICE/alice/ALICE/step4_function_guided_evolution/group.py#L95)对一维字符串种群使用二维索引，在这种输入下调用会失败；主进化路径并未据此证明会调用它，不能把它写成所有运行的必经错误。GA 的升序排列后覆盖低分个体、保留末尾高分个体的写法则符合其精英保留意图。

## ADR-006　功能预测器使用描述符；原版比较回归器，ALICE-X 使用 AutoGluon

**背景**

参考分数与实验标签量纲不同。功能预测器为候选提供生产适应度或受体富集的预测，既用于原版筛选，也用于 ALICE-X 的模型引导阶段。

**可比较方案**

特征可比较 one-hot、语言模型表征及理化描述符；模型可比较预设算法、交叉验证或自动调参。不能仅凭输入维数认定某方案必然失败，也不能把描述符加树模型称为对所有小数据任务最稳的组合。

**实现事实**

[cal_pep](ALICE/alice/ALICE/tools/parameters.py#L16)组合氨基酸组成、二肽组成、自相关、CTD、序列顺序、伪氨基酸组成和基础性质等描述。原版本次复现实际得到 **674 列**特征，并保持原特征顺序。

三个原版训练脚本配置 KNN、SVR、BayesianRidge、RandomForest、GradientBoosting、AdaBoost、XGBoost，以及五折划分和六项指标。[Ly6a 训练脚本](ALICE/alice/ALICE/evaluation/ly6a/Train_Target_Enrichment_LY6A.py#L64)

论文补充方法明确报告按综合性能选择 **Gradient Boosting**，超参数和指标见 Table S3–S6；本次实际加载的三个原版模型也都是 `GradientBoostingRegressor`。不能写成“完全没有记录最终选择了哪个模型”。但训练脚本本身没有完整展示从横评结果自动选择、汇总重训并导出三个最终命名模型的过程。[补充材料，第 8 页](ALICE/reproduction/references/Document_S1.pdf#page=8) · [实际加载记录](ALICE/reproduction/results/original/metrics.json)

[ALICE-X 训练入口](ALICE/alice_x/ranking/s2_auto_gb_regressor.py#L34)使用 AutoGluon 的 GBM 搜索，默认 `num_trials=120`，同时有时间限制和兼容性分支；这不是实际完成 120 次试验的保证。其 `test_*` 数据被传入 `tuning_data`，本次复测不能作为独立测试性能。

随库数据的精确规模以[复测审计](ALICE/reproduction/results/models/metrics.json)为准：fitness 训练可用 6131 行、hTfR1 训练可用 6134 行，两任务调参集各 256 行。四张 CSV 各有 676 列，包括序列标识、674 列描述符和标签；加载模型实际要求的输入特征分别为 645 和 648 列。不能把所有表格都写成“6134 行×674 维”。

**代价与已确认的实现问题**

- 原版描述符函数没有显式特征选择；ALICE-X 模型保留的列少于原描述符数，不能把这一描述推广为整个仓库没有任何特征处理。
- [原版训练循环](ALICE/alice/ALICE/evaluation/ly6a/Train_Target_Enrichment_LY6A.py#L28)在每折内部重置指标列表，循环后的 `mean` 只覆盖最后一折。应限定为当前脚本的输出问题，不能据此断定论文表格也这样计算。
- 同一脚本写预测文件时用外层最后一折的 `test` 决定遍历次数，而真正取值使用当前折索引；当前 KFold 划分在折大小不等时会少写较大折的部分预测。这不等于所有写出的标签都来自错误折。
- 多处把换行写成 `'/n'`，会破坏预期 CSV 行结构。
- 这些旧训练脚本的问题不改变本次已完成的“加载作者模型再预测”结果，也不证明从头重训已复现。

## ADR-007　编码器预训练后供生成网络使用；须区分原版与 ALICE-X

**背景**

生成模块为筛选和搜索提供候选。原版论文的语义训练集为 **72,753 条**；ALICE-X 随库 `semantic_tuning_data.csv` 有 **1388 行**。后者不能作为“整个原版只用约 1400 条训练”的依据。[补充材料，第 3 页](ALICE/reproduction/references/Document_S1.pdf#page=3)

**可比较方案**

可比较随机候选、从头训练生成模型、冻结预训练编码器或联合微调。公开代码能说明具体入口如何连接模型；不能据样本数直接断言其他方案必然失败。

**实现事实**

[预训练配置](ALICE/alice/ALICE/step1_pretrain/param.json)为隐藏维度 4096、词表 22、32 个注意力头和 1 个 Transformer 层。它包含上下文自注意力与前馈网络，即使冻结也仍是上下文编码器，不能称为固定的氨基酸查找表。

论文说明预训练使用长度不超过 50 的序列，并解释其考虑；当前数据脚本采用 `MAX_LEN=50`。不能另行断言其唯一原因是“下游只有 7 位，所以长程依赖无用”。[补充材料，第 3 页](ALICE/reproduction/references/Document_S1.pdf#page=3)

[原版语义入口](ALICE/alice/ALICE/step2_semantic_tuning/Train_SeqGAN.py#L277)加载编码器并设置 `requires_grad=False`。[Generator](ALICE/alice/ALICE/tools/SeqGAN.py#L14)还包含投影层、双向 LSTM、输出层和 highway，训练流程也更新判别器，所以“只训练一个小 LSTM”“预训练是唯一使用 GPU 的部分”均不准确。

**代价与已确认的实现问题**

1. **冻结不防止显式重初始化。** 若旧入口解析到仓库的 `tools/SeqGAN.py`，`Generator.init_params()` 会遍历包括 `self.emb` 在内的参数并写入随机值；随后训练脚本的 `generator.apply(weights_init)` 又会处理其中的线性层。冻结只阻止梯度更新，不阻止这种写入。因此不能保证此路径保留了加载的预训练参数，也不能据此反推已发布模型的实际训练历史。[生成器初始化](ALICE/alice/ALICE/tools/SeqGAN.py#L25) · [训练初始化](ALICE/alice/ALICE/step2_semantic_tuning/Train_SeqGAN.py#L42)

2. **掩码标签与损失口径不一致。** 主程序传入 `MaskSequence(0.3)`，其中 80% 换成 X、20% 保留、0% 随机替换；保留分支提前 `continue`，标签仍为 0。真实字符中期望 24% 有非零恢复标签，6% 被选中但保留，另 70% 未选中。当前 `CrossEntropyLoss()` 没有忽略 0，所以后两类及 PAD 位置仍以类别 0 计损，不能写成“6% 不计损”。准确率则只统计非零标签。[数据构造](ALICE/alice/ALICE/step1_pretrain/msk_datasets_construction.py#L37) · [损失与准确率](ALICE/alice/ALICE/step1_pretrain/pretrain_roberta.py#L138) · [PyTorch 损失定义](https://docs.pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html)

3. **LogSoftmax 后再用交叉熵是冗余，不能单独判成错误。** 在精确计算下 `logsoftmax(logsoftmax(z)) = logsoftmax(z)`；需要关注的是上一条未忽略 0 标签的真实目标变化。

4. **原版旧入口的 G 更新没有判别器反馈。** `g_loss` 由真实样本和自采样样本的两项交叉熵组成，判别器项单独更新 D；硬采样、one-hot 和 `detach()` 没有把 D 的目标接入 G 更新。这个结论限定于旧的 `Train_SeqGAN.py`，不能推广为整个仓库的生成训练均无对抗项。[原版损失](ALICE/alice/ALICE/step2_semantic_tuning/Train_SeqGAN.py#L123)

5. **ALICE-X 新入口已有可微对抗项。** `train_seqgan_htfr1.py` 在预热后使用 Gumbel-Softmax 直通采样，把判别器输出对应的 `g_adv_loss` 加入生成器损失。因此“必须给 ALICE-X 补上对抗梯度”不符合当前源码；已有损失连接也不等于整个训练入口已运行验证。[ALICE-X 生成器更新](ALICE/alice_x/finetuning/train_seqgan_htfr1.py#L295)

   该入口也存在冻结后重新初始化的问题；其 [Generator.forward](ALICE/alice_x/finetuning/SeqGAN.py#L35)返回的是展平为 `[B*T,V]` 的 Gumbel-Softmax 概率，而采样函数按 logits 处理。默认批量为 10、前缀长度大于 1 时，[形状处理分支](ALICE/alice_x/finetuning/train_seqgan_htfr1.py#L145)还会把第一行复制给整批，未正确恢复各样本最后一个位置。因此不能将“已有对抗项”等同于“新入口已完整修好”。

6. **预训练运行和保存还有限制。** 数据集写死 `.cuda()`；测试未切换 `eval()`；外层 10 次重建 trainer 会复用编码器但重建预测头、Adam 和调度器，每次内层训练 50 个 epoch；同名检查点会覆盖，保存的只有编码器，不含完整续训状态。[数据加载与主循环](ALICE/alice/ALICE/step1_pretrain/pretrain_roberta.py#L186)

使用已发布候选可以跳过重新生成，但改变了复现范围；ALICE-X 的候选也不能直接当作原版相同靶标、相同来源的等价输入。其他进化模块的导入和实现问题仍需单独处理。

## ADR-008　原版最终筛选结合排序、非单调正弦评分和固定阈值

**背景**

前置筛选和末端筛选都要使用多项分数，但不同阶段的公式、归一化范围和选择规则不同。

**可比较方案**

线性加权、最小值、非线性变换和多目标排序都可以作为方案；没有证据支持“线性加权在末端必然失败”或“必须禁止线性加权”。

**实现事实**

[RankFiltPro](ALICE/alice/ALICE/step3_rankfiltpro/RankFiltPro.py#L7)使用线性加权粗筛；末端 EA 对当前进入该步骤的候选做逐列 min-max 归一化，再计算：

```text
EA = 3.5*sin(1.85*C_norm) + 3.5*sin(2.70*A_norm)
     + 0.1*Prod_Fit_norm + 0.15*FE_score_norm
保留 EA >= 6.7
```

其中 `A` 对应 Ly6a，`C` 对应 Ly6c1。[EA 原始函数](ALICE/alice/ALICE/step5_elite_ascendancy/6-EliteAscendancy.py#L13)

**正弦项在实际区间内先升后降，不是单调饱和函数，也不等同于 soft-min。** 两个分量的峰值分别位于 `C_norm≈0.8491` 和 `A_norm≈0.5818`；超过峰值后分数下降。因此原始预测值更大，有时反而降低 EA。

本次输入和规则的实际行数为：

```text
输入 1000
→ 三次模型预测后仍为 1000
→ FE 排序取 500
→ 生产适应度排序取 100
→ EA 阈值后 12
→ multi_tar 阈值后 11
→ 指定序列排除后 8
```

第 4、5 步是按名次保留固定数量，不能称为固定数值阈值。第 7 步对两个原始靶标预测取均值并保留 `>=-1`，排序字段确为 Ly6c1；这能说明实现选择不同，不能仅据此判定排序字段是笔误。本次 12 条没有重复，该阈值实际剔除了 1 条。[筛选记录](ALICE/reproduction/results/original/selection_counts.csv) · [联合筛选代码](ALICE/alice/ALICE/step5_elite_ascendancy/7-multi-target-binding.py)

**代价与边界**

min-max 依赖当前候选的极值，同一序列在不同候选批次可能得到不同 EA，跨批比较需统一标尺。系数及阈值在代码中固定，但没有查到“为了恰好留下八条而反推 6.7”的证据；应删除这一作者动机断言。最终数量还由前后多道规则共同决定。

本次最终八条与论文 Table S13 的序列集合一致，预测生产适应度符合论文舍入值；这验证的是已发布输入与模型的筛选重放。[逐条核对](ALICE/reproduction/results/original/paper_table_s13_comparison.csv)

## ADR-009　ALICE-X Stage 2 使用预测和距离项，具体行为取决于模式

**背景**

当前 Stage 2 源码在候选功能预测之外加入与参考集、上一代的距离项。它体现了功能分数和序列差异的组合，但“先进入训练分布后模型就最可信”“因此消除了外推风险”没有被代码或本次实验验证。

论文补充方法 §7.4.2 给出了两阶段的设计解释：讨论熟悉序列模式导致的局部最优和过早收敛，将功能整合与后续探索分开。应把它表述为论文陈述的动机，而不是从黑名单规则反推的研发历史，也不代表 Stage 1 保证了训练分布覆盖。[补充材料，第 25 页](ALICE/reproduction/references/Document_S1.pdf#page=25)

**可比较方案**

可比较仅预测分数、预测分数加距离项、分阶段优化或其他多目标方式。不能把原版黑名单规则与 Stage 2 的出现之间写成已经证明的研发因果。

**实现事实**

[stage2_run_one_mode](ALICE/alice_x/FE/src/stage2_evolve.py#L82)调用 [evaluate_stage2_scalar](ALICE/alice_x/FE/src/evaluation.py#L207)，按模式决定奖励。

```text
默认 sum： reward = fit_pred + ht_pred
all 模式： reward = fit_pred + ht_pred + w_div*div_prev + w_ref*ref_dist

penalty = w_ref_pen*(1-ref_dist)
        + w_fit_pen*max(0, fit_base-fit_pred)
        + w_ht_pen*max(0, ht_base-ht_pred)

score = clamp(reward-penalty, 0, 1)
```

评价函数显式接受 8 个模式名称：`sum`、`min`、`rank_sum`、`sum+div`、`sum+ref`、`min+div`、`min+ref`、`all`。其中 `rank_sum` 当前与 `sum` 同公式，并未真正计算排名。[主入口](ALICE/alice_x/FE/src/stage2_evolve.py#L194)只运行 `modes=["sum"]`，也没有暴露选择模式的命令行参数。

默认 `sum` 没有上一代距离奖励；虽然仍计算并输出 `div_prev`，它不参与该模式的分数。参考距离惩罚仍然存在。`ref_dist` 是到正负参考并集的最小归一化 Hamming 距离，`div_prev` 是到上一代的最小归一化 Hamming 距离。

**代价与已确认的实现问题**

- `fit_base`、`ht_base` 来自初始种群的**原始预测值**分位数，但每代传给评分函数的是 min-max 后的 `[0,1]` 预测。二者直接相减存在尺度不一致，不能把它描述为已经验证有效的功能保持约束。[基线](ALICE/alice_x/FE/src/stage2_evolve.py#L102) · [归一化和调用](ALICE/alice_x/FE/src/stage2_evolve.py#L124)
- 每代单独归一化导致分数标尺变化，`clamp` 还可能把多个高分候选截为相同值；跨代奖励不能直接当作固定物理量比较。
- `pd.DtaFrame` 是类型注解拼写错误，在当前 Python 3.10 的即时注解求值下，若执行到该定义会报错；但 `mixture.py` 导入的 `GA.py` 缺失可能更早阻断 Stage 2。另一个缺失文件 `SA.py` 影响 Stage 1 入口，不能混作同一调用链，也不能认定第一个错误必然是类型注解拼写。[定义位置](ALICE/alice_x/FE/src/stage2_evolve.py#L33)
- [评价器构造函数](ALICE/alice_x/FE/src/evaluation.py#L70)先读取相对路径 `./datapack/sequences.json`；在 FE 的 `src/` 目录直接启动时，该默认文件不存在，即使后续准备另设参考集，也需要先处理初始化路径。
- [plot_reward_efficacy.py](ALICE/code_for_analysis/plot_reward_efficacy.py)分析单个实验目录中的距离分组与后续变化；它本身不是同时重跑、比较多个奖励模式的消融实验。
- 仓库保存的 epoch 预测表与当前入口输出的命名和预测尺度存在差别，不能直接用存档图表证明当前 Stage 2 入口已经完整运行通过。本次仅重算这些已存档数据。

## 附录 A　常量与配置入口索引

仓库有 `param.json`、参数对象、构造参数、setter 和命令行选项；问题是配置分散，而非“没有任何配置层”。

| 内容 | 入口 |
|---|---|
| 原版 7-mer、词表和生成规模 | [tools/parameters.py](ALICE/alice/ALICE/tools/parameters.py#L52) |
| 预训练结构 | [step1_pretrain/param.json](ALICE/alice/ALICE/step1_pretrain/param.json) |
| 预训练长度及实际掩码参数 | [数据主程序](ALICE/alice/ALICE/step1_pretrain/msk_datasets_construction.py#L108) |
| 原版搜索算子、固定进化循环 | [FE 入口](ALICE/alice/ALICE/step4_function_guided_evolution/function_guided_evolution.py#L17) |
| EDG 种群比例 | [mixture.py](ALICE/alice/ALICE/step4_function_guided_evolution/mixture.py#L20) |
| 比对 gap 与 extend | [MSAEva 初始化](ALICE/alice/ALICE/step4_function_guided_evolution/evaluation.py#L51) |
| EA 系数及阈值 | [EA 评分](ALICE/alice/ALICE/step5_elite_ascendancy/6-EliteAscendancy.py#L20) |
| ALICE-X 搜索超参数 | [ALICE-X FE 入口](ALICE/alice_x/FE/src/function_guided_evolution.py) |
| ALICE-X Stage 2 模式和权重 | [Stage 2 主程序](ALICE/alice_x/FE/src/stage2_evolve.py#L171) |

## 附录 B　导入、路径及复现范围

1. **扁平导入依赖正确的模块搜索路径。** 多个入口使用 `import parameters`、`from metrics import *` 等，相关文件位于不同目录。需要核对入口和 `sys.path`；仅把 `tools/` 加入路径并不能解决所有依赖，例如 `tools/minibert.py` 引用的 `attention.py` 及其依赖 `util_layers.py` 位于原版 `step1_pretrain/`。

2. **缺少 `__init__.py` 不自动构成错误。** Python 支持隐式命名空间包。扁平的 `import parameters` 与 `import tools.parameters` 对路径有不同要求，可通过明确包结构和导入方式改善，但不是简单添加一个文件就能保证全部入口可运行。[Python 导入规则](https://docs.python.org/3.10/reference/import.html#namespace-packages)

3. **`evaluation` 同名风险取决于实际搜索路径和缓存。** 直接执行 step4 的脚本通常会将脚本目录置于搜索路径前部；同时能找到该目录下的 `evaluation.py` 时，不能仅因当前工作目录是 `alice/ALICE/` 就断言导入到了命名空间目录。应检查实际模块来源及 `sys.modules`，再决定是否统一包式导入或重命名。

4. **区分修正文档与修复训练代码。** 本文记录公开实现的已确认问题，没有宣称这些未执行的训练、进化模块已经修好。已有复现脚本采用作者提供的模型和输入，并保留实际使用的特征顺序、评分公式及筛选阈值。

## 附录 C　与本次复现结果的对应

| 项目 | 已确认结果 | 边界 |
|---|---|---|
| 原版八步筛选 | 最终 8 条序列与 Table S13 集合一致，预测值符合论文舍入精度 | 使用作者公开输入及已训练模型 |
| ALICE-X 模型复测 | 24 项参考指标均在 `1e-6` 容差内；最大差约 `1.11e-16` | 使用上游调参集，不是新的独立测试 |
| Figure 6F 相关性复算 | 选用 `huTfR1fc_log2_enrichment` 列时 `r=0.522840787`，舍入为 `0.523` | 原绘图脚本另一列给出 `0.423468353`；具体实验列映射尚未完全确认 |
| 从头预训练、生成、完整进化及湿实验 | 本次未重新执行 | 不可将源码分析或存档结果当成重训成功 |

详见[复现报告](ALICE/复现报告.md)、[验证记录](ALICE/reproduction/verification.json)和[运行说明](ALICE/reproduction/README_复现说明.md)。

