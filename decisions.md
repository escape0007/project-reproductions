# ALICE 决策记录（ADR）

> **这个文件补的是什么**
> 原仓库里所有设计决策都只固化在**常量和代码结构**里（`train_len=7`、
> `requires_grad=False`、`3.5*sin(...)`），没有任何地方写下"为什么这么选、
> 代价是什么"。读者只能从常量反推意图，而反推不出被否掉的方案 ——
> 那恰恰是最有价值的信息。
>
> **格式**：每条记录四段 —— 背景 / 选项 / 决定 / **代价**。
>
> **引用约定**：所有路径以仓库根为起点，`文件:行号` 

---

## 决策依赖图

```
ADR-001 设计变量 = 7-mer
   └─► ADR-002 标签 = 两个 log2 比值
          └─► ADR-003 走序列统计路线
                 ├─► ADR-004 ★ 打分 = 对比参考集（不训模型当适应度）
                 │      ├─► ADR-005 搜索 = 种群启发式（因为打分不可微）
                 │      └─► ADR-009 第二轮目标反转（因为 004 会"抄答案"）
                 ├─► ADR-006 代理模型 = 手工描述符 + 回归器横评
                 │      └─► ADR-008 多目标 = 饱和变换（不用线性加权）
                 └─► ADR-007 生成器 = 预训练后冻结，只训小 LSTM
```

读法：ADR-004 是枢纽。它决定了必须用 ADR-005 的搜索方式，也直接导致了
ADR-009 的存在。改动 004 等于重做整个项目。

---

## ADR-001　设计变量只取插槽里的 7 个氨基酸

**背景**

外壳蛋白全长 700+ 个氨基酸。"设计一个更好的外壳"听起来是正确的问法，
但它的搜索空间是 20^700，而且绝大多数改动会让外壳直接组装失败 ——
没有任何可获得的数据量能支撑这种设计。

**选项**

1. 全长设计（20^700）
2. 选几个关键位点做点突变（空间小，但"选哪些位点"本身就需要结构知识）
3. 只设计表面插槽里的 7 个氨基酸（20^7 ≈ 1.28e9）

**决定**

取方案 3。该插槽是实验上成熟的"可改写区"：塞 7 个氨基酸不破坏整体结构，
而且这 7 位直接暴露在最外侧、直接决定与受体的结合。

```
固化在：
  alice/ALICE/tools/parameters.py:89          GAN.train_len = 7
  alice/ALICE/tools/parameters.py:59,74       bert.seq_len = 7 / roberta.seq_len = 7
  alice/ALICE/step2_semantic_tuning/Train_SeqGAN.py:282   start_end = (7,7)
  alice/ALICE/step4_function_guided_evolution/function_guided_evolution.py:68
                                              seqlen = 7
  alice/ALICE/step4_function_guided_evolution/group.py:26
                                              self.lib = 'ACDEFGHIKLMNPQRSTVWY'
```

**代价**

- 放弃了外壳其他部位可能带来的改进（免疫逃逸、稳定性、组织特异性）。
- **天花板被插槽本身限制住**：如果最优解不在这 7 位里，本方法永远找不到。
- 换一个插槽位置，参考集和所有模型全部作废，要从头再来。
- 7 这个长度是硬编码在至少 5 个文件里的，不是配置项 —— 想试 9-mer 要改多处。

> **通用经验**：选对设计变量比选对模型重要一个数量级。
> 要找的是"自由度小、对目标影响大、改了不破坏系统"的切入点。

---

## ADR-002　目标函数由"能测什么"反推，不由"想要什么"正推

**背景**

真正想要的是"疗效好"。但疗效测不起、测不快、测不了上千条，
没法当成能批量打标签的监督信号。

**选项**

1. 用疗效当标签 —— 样本量只能到几十条，训不出任何东西
2. 用理论打分（结合自由能等）当标签 —— 本身就不准，等于拿噪声当真值
3. 用测序能廉价给出的计数比值当标签

**决定**

取方案 3。测序便宜地给出三组计数（投料池 / 产出池 / 通关池），定义两个标签：

```
prod_fit          = log2(virus  / plasmid)    还造得出来吗
target_enrichment = log2(target / virus)      造出来的能刷开锁吗
```

取 log2 是为了让"翻倍"和"减半"在数值上对称。

```
固化在列名里：
  alice/ALICE/evaluation/production fitness/Train_Production_Fitness.py:80
        y = train_data['Prod_Fit'].values
  alice/ALICE/evaluation/ly6a/Train_Target_Enrichment_LY6A.py:81
        y = train_data['LY6A_log2enr'].values
  alice/ALICE/step5_elite_ascendancy/6-EliteAscendancy.py:34
        ['Target_LY6A_Enri', 'Target_LY6C1_Enri', 'Prod_Fit', 'FE_score']

原始三列计数可直接看（第二轮的数据，仓库里带了）：
  alice_x/finetuning/data/semantic_tuning_data.csv
        iAAs, plasmid_mean, virus_mean, hutfr1_mean,
        V_mean__P_mean_log2_enr, hT_mean__V_mean_log2_enr
```

**代价**

- 这两个比值是**代理量，不是疗效**。优化到极致也不保证能治病。
- **比值对低计数极度敏感**：投料 3 / 产出 1 算出 0.33，纯属噪声；
  而噪声样本的比值往往最极端，模型会优先去拟合这些垃圾。
  ⇒ 必须设计数下限。原仓库只过滤了含 `_` 的行
  （`alice_x/ranking/s1_data_preprocessing.py:126`），**没有计数下限** —— 这是个缺口。
- 三个池之间的实验批次差异会直接污染标签，且事后无法纠正。
- 两个标签是**比值的比值**（target/virus 本身已经除过一次），误差会叠加。

> **通用经验**：先问"我能廉价、批量、定量地测到什么"，那个东西才有资格当标签。
> 并且要始终清楚代理量和真实目标之间的差距在哪。

---

## ADR-003　走序列统计路线，不走结构/对接

**背景**

两条路都成立：预测三维结构再做对接打分，或者直接学"序列 → 数值标签"。

**选项**

1. 结构预测 + 分子对接
2. 端到端深度学习（序列直接出预测值，大模型）
3. 描述符 + 传统回归 + 启发式搜索

**决定**

取方案 3。

**为什么否掉前两个**

- 方案 1：这 7 个氨基酸形成的是**柔性环、无稳定构象**（结构预测对这类区域最不准）；
  外壳是 60 聚体巨复合物；且无法对百万级候选逐一对接。算不准，也算不起。
- 方案 2：带标签样本约 6×10³，空间 1.28×10⁹，**覆盖率约 5×10⁻⁶**。
  端到端模型必然过拟合，且在分布外完全失效 —— 而设计任务本质上就是在分布外找东西。


**代价**

- 放弃了任何机理层面的理解，拿到的是**纯相关性**。
- 无法外推到训练分布之外（这个代价后来由 ADR-004 来补偿）。
- 必须靠 `code_for_analysis/` 里的可解释性分析来补充说服力，
  并且最终只能由湿实验裁决。

> **通用经验**：技术路线的选择标准不是"哪个更接近本质"，
> 而是"哪个跟我的数据量、算力和时间预算匹配"。

---

## ADR-004　★ 打分用"对比参考集"，不训神经网络当适应度 ★

> **这是整个项目的枢纽决策。** 

**背景**

搜索需要一个适应度函数。

**选项**

1. 回归模型的预测值直接当适应度
2. 结构对接能量当适应度（已被 ADR-003 否掉）
3. 跟"已知好序列"的相似度 **减去** 跟"已知坏序列"的相似度

**决定**

取方案 3。相似度用 BLOSUM62 加权的 Smith-Waterman 局部对齐。

```
score(s) = [ mean_{p∈Pos} SW(s,p) − mean_{n∈Neg} SW(s,n) ] / 归一化 + 0.5
```

```
实现：
  alice/ALICE/step4_function_guided_evolution/evaluation.py
        :23   class MSAEva(EvaMethod)
        :79   loadSample()   → 读 seq_LY6A_LY6C1.json
        :88   loadMatrix()   → 读 BLOSSUM62.txt，顺带定出 scorebound
        :108  msa(a, b)      ← Smith-Waterman 动态规划
        :157  evaluate(seq)  ← 对比打分
        :162  return (calculate_mean(dim, posScores)
                      - calculate_mean(dim, negScores)) / ... + 0.5
  alice/ALICE/step4_function_guided_evolution/basic.py:42
        calculate_mean(dim, values)   ← evaluate 的 dim 参数其实是 p-范数开关

数据（打分器的全部"知识"）：
  alice/ALICE/step4_function_guided_evolution/seq_LY6A_LY6C1.json   76 正 / 53 负
  alice/ALICE/step4_function_guided_evolution/BLOSSUM62.txt         20×20 相似度表

第二轮的加速版（同一逻辑 + numba + 配对缓存 + 仿射 gap）：
  alice_x/FE/src/evaluation.py:40   @njit _sw_score_numba()
```

**为什么**

- 方案 1 有个**致命循环**：模型只在训练分布附近可信，而搜索恰恰要往外走。
  用它当适应度，搜索会直奔模型的"幻觉高分区"—— 分数越刷越高，实际越来越差。
- 方案 3 是**纯几何**的，不存在分布外崩塌。
- 离散、不可微也无妨 —— 正好配合 ADR-005 的启发式搜索。
- **白捡的最大好处：正样本集里同时放多个功能的好序列，
  就自动实现了"多功能融合"。** 文件名 `seq_LY6A_LY6C1.json` 就是这个意思。
  这不是靠模型，是靠**参考集的构成**。
- 可解释：进化完能直接画 motif 图，看出哪段来自哪个功能
  （`code_for_analysis/motif_change_delta.py`）。

**代价**

- 整个打分器的信息量全在那 **76+53 条**参考序列里。选错参考集，全盘皆错。
  而原仓库没有记录这些序列是怎么挑的 —— **不可复现的关键一环**。
- 负样本**不能**用随机序列：否则打分器学到的只是"像不像一个正常蛋白片段"，
  对本任务毫无区分力。原仓库的负样本是实测差的，这点做对了。
- 它只回答"像不像已知的好序列"，**不回答绝对量纲**
  ⇒ 所以 ADR-006 的回归模型仍然需要，但降级到漏斗末端当排序器用。
- **"向参考集回归"的固有风险**：优化到底就是把参考序列抄一遍。
  证据就是 `alice/ALICE/step5_elite_ascendancy/8-delete_reported_seqs.py`
  不得不硬删 `IRQGYSS` / `LRVGYSS` / `LRAGYSS` 三条已报道序列。
  ⇒ 这个代价直接催生了 ADR-009。
- `MSAEva.msa()` 是纯 Python 三重循环，1000 条种群 × 129 条参考 × 11 代
  = 约 142 万次 DP，慢。第二轮才用 numba 修掉。

---

## ADR-005　搜索用种群启发式，默认 GA+EDA 混合

**背景**

ADR-004 的打分函数不可微，搜索空间离散，梯度法完全不可用。

**选项**

1. 全枚举 —— 1.28e9 × 129 次对齐，不可能
2. 随机采样 —— 覆盖率 ~10⁻⁶，等于白撒
3. 单一启发式（只用 GA，或只用 EDA，或只用 SA）
4. 多算子并存 + 默认用混合算子

**决定**

取方案 4。实现四种算子，默认 `EDG = 70% GA 后代 + 30% EDA 采样`。

```
  alice/ALICE/step4_function_guided_evolution/group.py:7
        class Group(ABC)      ← 种群抽象基类，evolution() 是抽象方法
                                 evamet 是 property，setter 里自动重建 np.vectorize
  alice/ALICE/step4_function_guided_evolution/GA.py:18
        class GAGroup         rmutate=0.3  rcopy=0.2  单点/双点交叉
  alice/ALICE/step4_function_guided_evolution/EDA.py:9
        class EDAGroup        rbest=0.5  alpha=0.6（PBIL 式分布平滑）
  alice/ALICE/step4_function_guided_evolution/SA.py:13
        class SAGroup         t0=1.6  经典/加速退火两种降温
  alice/ALICE/step4_function_guided_evolution/mixture.py:10
        class EDGGroup(EDAGroup, GAGroup)   ← 多继承拼算子
        :20   r_eda=0.3, r_ga_copy=0.2
        :22   self.alpha = 1   ← 注意：覆盖了 EDA 的 0.6，等于退化成 UMDA（不平滑）
  alice/ALICE/step4_function_guided_evolution/function_guided_evolution.py
        :17   --group_type 选算子
        :87   total_epoch = 11
  alice/ALICE/step4_function_guided_evolution/basic.py:67  weighted_choice()   ← 按分数加权抽亲本
```

**为什么是混合**

- GA 擅长**保住好的局部片段**（"积木"），但容易卡在少数几个序列家族里出不来。
- EDA 擅长**抓全局共识**（逐位字母频率），但会飞快坍缩到一条共识序列、多样性全丢。
- 混合让两者互相救：GA 提供多样性来源，EDA 提供收敛压力。

**代价**

- 四套算子的超参（`rmutate` `rcopy` `rbest` `alpha` `r_eda` `t0`）全部是写死的默认值，
  原仓库没有调参记录，也没有消融对比 —— 无法判断 EDG 是否真的优于单一算子。
  （第二轮补上了 `alice_x/FE/src/function_guided_evolution.py` 的
  `--edg_r_eda` / `--ga_rmutate` 等开关和 `run_summary.json`。）
- `mixture.py:22` 把 `alpha` 覆盖成 1，使 EDA 的跨代分布平滑失效。
  这可能是有意的（配合 GA 已经提供了惯性），也可能是漏改 —— **代码里没说**。
- `total_epoch = 11` 是硬编码的魔法数，没有收敛判据。
- `group.py:95 calcDis()` 对一维字符串数组做二维索引（`self.inds[:, i]`），
  **必定抛 IndexError**。它是死代码 —— EDA 用的是自己的 `calcBestdis()`。
- GA 的精英保留写法很隐晦：`newinds = self.inds[np.argsort(self.score)]`（升序）
  后覆写前 `ncross` 个，于是存活下来的是**数组末尾的高分个体**。
  正确但极易误读。

---

## ADR-006　代理模型：手工描述符 + 多回归器横评

**背景**

ADR-004 的对比打分不给绝对量纲，但漏斗末端要在真实量纲上排序、设阈值。

**选项**

特征表示：
1. one-hot（7×20 = 140 维）
2. 预训练语言模型的 embedding（4096 维/位）
3. 手工理化描述符（约 674 维）

模型：
4. 预设一个模型（比如直接上 XGBoost）
5. 多个模型横评 + 交叉验证

**决定**

特征取方案 3，模型取方案 5：7 个回归器 × 5 折交叉验证，看 Pearson 挑。

```
特征工程：
  alice/ALICE/tools/parameters.py:16
        cal_pep(seq)   ← 把 8 类描述符拼成一个 dict
  alice/ALICE/tools/featured_data_generated/
        AAComposition.py       氨基酸组成 + 二肽组成
        Autocorrelation.py     Moreau-Broto 归一化自相关（lamba=5）
        CTD.py                 组成 / 转换 / 分布
        PseudoAAC.py           伪氨基酸组成（lamda=4）+ 两性伪氨基酸组成
        QuasiSequenceOrder.py  准序列顺序耦合数（maxlag=5）
        BasicDes.py            基础理化性质

训练（三个任务，三个结构完全相同的脚本）：
  alice/ALICE/evaluation/production fitness/Train_Production_Fitness.py
  alice/ALICE/evaluation/ly6a/Train_Target_Enrichment_LY6A.py
  alice/ALICE/evaluation/ly6c1/Train_Target_Enrichment_LY6C1.py
        :66  KNN / SVR / BayesianRidge / RandomForest /
             GradientBoosting / AdaBoost / XGBoost
        :79  KFold(5, shuffle=True, random_state=42)
  alice/ALICE/tools/metrics.py
        reg_scores()   ← rmse / pearson / spearman / CI / R² / MedAE

第二轮改为 AutoML：
  alice_x/ranking/s2_auto_gb_regressor.py   AutoGluon GBM + HPO（num_trials=120）
  alice_x/ranking/datapack/*.csv            6134 行 × 674 维（仓库里就有）
```

**为什么**

- one-hot 太稀疏，树模型学不到"这两个字母化学性质接近"这种信息。
- 预训练 embedding 维度过高，6×10³ 样本撑不住。
- 674 维描述符里已经编码了疏水性、电荷、分子量、序列顺序等物理量，
  配 6000 样本 + 树模型是小数据上最稳的组合。
- 不预设模型、让交叉验证说话 —— 小样本上这是正确做法。

**代价**

- 674 维描述符里大量冗余和共线，原仓库**没有特征选择步骤**。
- 三个训练脚本是**复制粘贴**的（除了输入路径和标签列名完全相同），
  任何修改要改三处。
- 训练脚本的 bug：`rmse, pearson, ... = [], [], ...` 在每个 fold 内部重新初始化
  （`Train_Target_Enrichment_LY6A.py:29`），所以最后 `np.mean(rmse)`
  **只统计了最后一折** —— 打印出来的"5 折平均"是假的。
- `preds_score.csv` 的写入循环里用了外层泄漏的 `test` 变量
  （`:50  for num, sc in enumerate(y[test])`），而不是 `tst_data[iter_]`。
- 换行符写成了 `'/n'` 而不是 `'\n'`（多处），输出的 csv 是一整行。
- 没有记录最终选了哪个模型 —— `checkpoint/.../LY6A_model.sav` 是哪个回归器，
  代码里查不到。

---

## ADR-007　生成器：预训练后冻结，只训小 LSTM

**背景**

进化需要初始种群。纯随机 7-mer 里绝大多数一开始就不合格，
要烧掉很多代才能拉回可用区域。希望初始种群就"像真实文库里的序列"。

但真实文库标注数据只有约 1400 条。

**选项**

1. 纯随机初始化种群
2. 从零训一个 Transformer 生成器（1400 条样本）
3. 在无标注语料上预训练，然后**全参数微调**
4. 在无标注语料上预训练，**冻结**，只训一个小 LSTM 生成器

**决定**

取方案 4。

```
预训练（唯一吃 GPU 的部分）：
  alice/ALICE/step1_pretrain/param.json
        hidden size 4096 / vocab 22 / attn heads 32 / n layers 1 / dropout 0.1
  alice/ALICE/step1_pretrain/attention.py      Attention / MultiHeadedAttention / TransformerBlock
  alice/ALICE/step1_pretrain/util_layers.py    PositionwiseFeedForward / SublayerConnection
  alice/ALICE/step1_pretrain/minibert.py       BERTEmbedding / BERT / MaskedRoberta
  alice/ALICE/step1_pretrain/optim_schedule.py warmup 调度
  alice/ALICE/step1_pretrain/msk_datasets_construction.py
        :109  MAX_LEN = 50          ← 刻意不用 512
        :128  MaskSequence(0.3)     ← 掩码率 0.3，不是 BERT 的 0.15
  alice/ALICE/step1_pretrain/pretrain_roberta.py   class BERTTrainer + 训练循环

语义微调：
  alice/ALICE/tools/SeqGAN.py
        :14   class Generator       冻结 BERT 当 embedding → Linear → BiLSTM → highway
        :149  class Discriminator   多尺度 Conv1d(k=1,3,5,7) + ResBlock×5 + highway
  alice/ALICE/tools/utils.py  gumbel_softmax()
  alice/ALICE/step2_semantic_tuning/Train_SeqGAN.py:281
        ★ for para in pretrain_pth.parameters(): para.requires_grad = False
  alice/ALICE/step2_semantic_tuning/Inference_SeqGAN.py
        20 轮 × gen_num=1000 → 约 2 万条候选 → infer_gen_seqs.csv
```

**为什么**

- 方案 2 必死：1400 条样本训不出 Transformer。
- 方案 3 也会过拟合：hidden=4096 的模型用 1400 条样本做全参微调，参数量碾压数据量。
- 方案 4 把大模型降级为**固定的 embedding 查找表** —— 它提供的价值是
  "化学性质相近的氨基酸，向量也相近"，而不是"懂蛋白质"。
  真正学习"什么像衣壳序列"的是那个小 LSTM。
- `MAX_LEN=50`（而非 512）是有意的：下游只处理 7 个字母，
  学长程依赖无用，短序列能把 batch 开到 512，训得快得多。

**代价**

- **"semantic tuning"这个命名有误导性**：被微调的是生成器，不是语言模型。
- 预训练只有 **1 层** transformer（`param.json: "n layers": 1`），
  能学到的"蛋白语法"非常有限 —— 它基本就是个带位置编码的 embedding。
- `pretrain_roberta.py` 里有三处实现偏差，**代码里没有任何说明**：
  1. `MaskedRoberta.forward` 已经 `LogSoftmax`，外层又用 `CrossEntropyLoss`
     （`:69`）—— **double log-softmax**。
  2. 损失没有 `ignore_index=0`（`:68` 把 `NLLLoss(ignore_index=0)` 注释掉了，损失调用在 `:138`），
     于是模型被训练成"在所有未掩码位置输出 `<PAD>`"。
  3. `msk_stg=(0.8, 0.0, 0.2)` 下"随机替换"分支永远走不到
     （`msk_datasets_construction.py:43`），
     实际是 24% 位置换成 `X` 并计损、6% 位置原样保留且**不计损**，
     而非 BERT 的 80/10/10。
- **最严重的一处**：`Train_SeqGAN.py` 里
  `g_loss = (g_loss_1 + g_loss_2)/2`，两项都是交叉熵 MLE，
  **判别器的梯度从未传到生成器**。D 被训练、准确率被记录，
  但 G 是纯 MLE 训练的 —— 所以这个"GAN"实际没有对抗。
  若要忠实复现，这点必须知道；若要改进，需补 REINFORCE 或 Gumbel-ST 直通。
- 这一层**可以整体跳过**：用 `alice_x/FE/data/infer_gen_seqs.csv`
  （仓库里现成的 1000 条）顶替初始种群，ADR-004/005/006 照样跑。

---

## ADR-008　多目标用饱和变换，禁止线性加权

**背景**

最终要把"生产适应度 + A 锁结合 + B 锁结合 + 对比分"合成一个数来排序。

**选项**

1. 线性加权求和
2. 取最小值（硬 min）
3. 凹的饱和变换后求和（如 `sin`，在接近峰值处封顶）

**决定**

漏斗**前段**用方案 1 做粗筛（便宜、够用），**末端**用方案 3 做精排。

```
粗筛（前段，线性加权可接受，因为后面还有硬阈值兜）：
  alice/ALICE/step3_rankfiltpro/RankFiltPro.py:7
        filter_score = 0.3*Prod_Fit + 0.35*LY6A + 0.35*LY6C1
  :8    → 排序取 Top1000

精排（末端，必须用饱和变换）：
  alice/ALICE/step5_elite_ascendancy/6-EliteAscendancy.py:22
        EA_Score = 3.5*sin(Target_LY6C1_Enri_norm * 1.85)
                 + 3.5*sin(Target_LY6A_Enri_norm  * 2.70)
                 + 0.1 *Prod_Fit_norm
                 + 0.15*FE_score_norm
  :42   阈值 EA_Score >= 6.7

硬阈值漏斗：
  4-FE_score_Filter.py      FE_score  排序 → Top500
  5-Prod_Fit_Filter.py      Prod_Fit  排序 → Top100
  7-multi-target-binding.py multi_tar = 0.5*A + 0.5*B，保留 >= -1
  8-delete_reported_seqs.py 剔除 3 条已报道序列 → output/EA/ALICE_TOP.csv
```

**为什么用 sin**

`sin` 在接近峰值处**饱和**。系数 `2.70` 已超过 π/2 ≈ 1.571，
意味着 A 锁的归一化分数越过峰值后**不再加分**。两个 3.5 权重的饱和项相加，
效果是一个**平滑的 AND（软 min）**：

> "两把锁都够好"的总分 **远高于** "一把极好、另一把很差"。

`Prod_Fit` / `FE_score` 只给 0.1 / 0.15 的小权重，因为它们已经在
第 4、5 号脚本被硬阈值筛过一遍了 —— 不需要在这里再竞争。

**代价**

- **线性加权必然失败**（如果用在末端）：一项极好 + 一项极差，
  可以和两项都不错得同样的分，而优化器一定会选前者，因为堆单项更容易。
  这就是为什么末端不能沿用 `RankFiltPro` 那套权重。
- `1.85` / `2.70` / `3.5` / `0.1` / `0.15` / `6.7` 全是**手调的魔法数**，
  原仓库没有任何推导或调参记录。
- `6.7` 这个阈值是**反推出来的**：它的作用是让最终剩下恰好能进实验的条数。
  这是合理的工程做法（实验通量是硬约束），但必须说清楚，否则看起来像过拟合。
- `normalize_column` 是 min-max 归一化（`6-EliteAscendancy.py:17`），
  **依赖当前批次的极值** ⇒ 换一批候选，同一条序列的 EA_Score 会变。
  不同轮次之间的 EA_Score **不可比**。
- `7-multi-target-binding.py` 里 `multi_tar >= -1` 几乎不过滤任何东西，
  而且排序用的是 `Target_LY6C1_Enri` 而非刚算出的 `multi_tar` —— 可能是笔误。

---

## ADR-009　第二轮把目标从"像已知"反转为"远离已知"

**背景**

ADR-004 的代价暴露了：优化"像已知的好序列"，优化到底就是把参考序列抄一遍。
直接证据是 `8-delete_reported_seqs.py` 不得不硬删三条已报道序列。

**选项**

1. 继续用 ADR-004 的打分，靠事后黑名单剔除已知序列
2. 在打分里加新颖性惩罚项（一个阶段内平衡"像"和"不像"）
3. **两阶段**：Stage 1 仍用"像好的"把种群拉进可行区，
   Stage 2 换成"预测分高 + 远离参考集 + 远离上一代"

**决定**

取方案 3。

```
  alice_x/FE/src/evaluation.py
        :142  evaluate()                  ← Stage 1：仍是"像好的 − 像坏的"
        :184  hamming()
        :188  ref_distance_norm()         ← 到 Ref+/Ref- 的最小归一化距离
        :200  diversity_prev_norm()       ← 到上一代的最小归一化距离
        :207  evaluate_stage2_scalar()    ← ★ Stage 2 奖励
              reward  = fit_pred + ht_pred
                      + w_div * div_prev      离上一代远
                      + w_ref * ref_dist      离参考集远
              penalty = w_ref_pen*(1-ref_dist)
                      + w_fit_pen*max(0, fit_base - fit_pred)
                      + w_ht_pen *max(0, ht_base  - ht_pred)
              score   = clamp(reward - penalty, 0, 1)
  alice_x/FE/src/stage2_evolve.py
        :82   stage2_run_one_mode()  ← 7 种 reward 组合模式（sum/min/+div/+ref/all）
        :107  fit_base / ht_base 取初始种群预测值的分位数当基线
  code_for_analysis/plot_reward_efficacy.py   ← 不同 reward 模式的效果对比（消融）
```

**为什么**

- 方案 1 是打补丁：黑名单只能剔除你**已知**的已报道序列，
  对"实质相同但差一两个字母"的序列无效。
- 方案 2 的两个方向在同一个标量里互相抵消，权重极难调。
- 方案 3 把两个矛盾目标**在时间上分开**：先拉进来，再推出去。
  并且 Stage 2 的预测分来自 ADR-006 的回归模型，
  此时种群已经在训练分布附近，模型可信度最高 ——
  **正好绕开了 ADR-004 当初否掉"模型当适应度"的那个理由**。

**代价**

- 引入 5 个新权重（`w_div` `w_ref` `w_ref_pen` `w_fit_pen` `w_ht_pen`），
  默认值 `0.25/0.25/0.35/0.3/0.3` 仍是手调的。
- Stage 2 依赖 ADR-006 的模型预测，**又把模型外推的风险请回来了** ——
  靠 `fit_base`/`ht_base` 分位数基线惩罚来约束，但这只是软约束。
- `minmax01()` 对每代预测值重新归一化（`stage2_evolve.py:67`），
  所以 **跨代的 reward 不可比**，训练曲线的含义需要小心解读。
- `stage2_evolve.py:33` 的返回标注写成 `pd.DtaFrame`（拼写错误），
  **模块 import 即报 AttributeError** —— 这个文件现在跑不起来，必须先修。
- `modes = ["sum"]`（`:194`）把 7 种模式写死成只跑一种，
  README 里宣传的其它 6 种模式需要改代码才能用。

---

## 附录 A　"决策只存在于常量里"索引

原仓库没有配置层，所有决策散落在代码常量中。想改题目时按此表查：

| 决策 | 常量 | 位置 |
|---|---|---|
| 序列长度 7 | `train_len=7` / `seq_len=7` / `seqlen=7` / `start_end=(7,7)` | `tools/parameters.py:59,74,89`；`step4/function_guided_evolution.py:68`；`step2/Train_SeqGAN.py:282` |
| 字母表 20 | `self.lib='ACDEFGHIKLMNPQRSTVWY'` | `step4/group.py:26` |
| 词表 22（含 PAD/X） | `"vocab": 22` | `step1_pretrain/param.json` |
| 预训练长度上限 50 | `MAX_LEN = 50` | `step1_pretrain/msk_datasets_construction.py:109` |
| 掩码率 0.3 | `MaskSequence(0.3)` | `step1_pretrain/msk_datasets_construction.py:128` |
| 大模型冻结 | `requires_grad = False` | `step2/Train_SeqGAN.py:281` |
| 候选规模 2 万 | `gen_num=1000` × `range(20)` | `tools/parameters.py:88`；`step2/Inference_SeqGAN.py:16` |
| 粗筛权重 | `0.3 / 0.35 / 0.35` | `step3_rankfiltpro/RankFiltPro.py:7` |
| 粗筛保留 1000 | `.head(1000)` | `step3_rankfiltpro/RankFiltPro.py:8` |
| 进化代数 11 | `total_epoch = 11` | `step4/function_guided_evolution.py:87` |
| 默认算子 EDG | `default='EDG'` | `step4/function_guided_evolution.py:17` |
| EDG 配比 | `r_eda=0.3, r_ga_copy=0.2` | `step4/mixture.py:20` |
| GA 变异/复制率 | `rmutate=0.3, rcopy=0.2` | `step4/GA.py:33-34` |
| EDA 精英率/平滑 | `rbest=0.5, alpha=0.6`（被 EDG 覆盖为 1） | `step4/EDA.py:21-22`；`step4/mixture.py:22` |
| gap 罚分 | `gap=-4, extend=-2` | `step4/evaluation.py:53-54` |
| EA Score 系数 | `3.5 / 1.85 / 2.70 / 0.1 / 0.15` | `step5/6-EliteAscendancy.py:22-25` |
| EA 阈值 | `>= 6.7` | `step5/6-EliteAscendancy.py:42` |
| 漏斗各级 | `500 → 100` | `step5/4-...py:4`；`step5/5-...py:4` |

## 附录 B　复现前必须解决的三个结构性问题

1. **没有任何 `__init__.py`，却跨目录 import 扁平模块名。**
   `step4/function_guided_evolution.py:13 import parameters`、
   `step5/1-*.py:9 from metrics import *`、
   `evaluation/ly6a/*.py:8 from metrics import *`
   —— 这些模块都在 `alice/ALICE/tools/`。必须把 `tools/` 加进 `PYTHONPATH`。

2. **两个脚本的 import 风格不一致。**
   `step2/Train_SeqGAN.py:13` 用 `import parameters`（扁平），
   `step2/Inference_SeqGAN.py:8` 用 `import tools.parameters`（包式）。
   这两个脚本需要不同的 `sys.path` 配置。

3. **名字冲突：`step4/evaluation.py` 与 `evaluation/` 目录同名。**
   `step4/group.py:1` 写的是 `import evaluation as eva`。
   若以 `alice/ALICE/` 为工作目录运行，`import evaluation` 可能解析到那个
   **目录**（隐式命名空间包）而非 `step4/evaluation.py`，
   报一个很难定位的 `AttributeError: module 'evaluation' has no attribute 'MSAEva'`。

