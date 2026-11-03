# 项目复现 · Project Reproductions

本仓库整理科研项目的源码、复现脚本、运行记录与结果，按项目分别存放。目前收录 **ALICE / ALICE-X**，包括上游代码以及基于公开输入和已发布模型完成的计算复现材料。

## 文件夹架构

下面展示主要目录与入口；数据集、模型文件和逐次运行产物未逐一展开。

```text
project-reproductions/                  # 对应本地“项目复现”文件夹
├── README.md                           # 仓库总览与目录说明
├── .gitignore                          # 本机环境、缓存等文件的排除规则
└── ALICE/
    ├── README.md                       # 上游项目介绍
    ├── LICENSE                         # 上游 MIT 许可证
    ├── 复现报告.md                     # 已有复现报告
    ├── alice/                          # 原版 ALICE
    │   ├── README.md                   # 原版安装与使用说明
    │   ├── environment.yml             # 上游环境配置
    │   ├── ALICE/                      # 模型及各阶段实现
    │   ├── pre_train_data/             # 预训练数据说明
    │   └── statistical_code/           # 统计分析脚本
    ├── alice_x/                        # ALICE-X
    │   ├── FE/                         # 特征工程代码与数据
    │   ├── finetuning/                 # 微调与生成相关代码
    │   ├── ranking/                    # 预测、排序、模型与数据
    │   └── tools/                      # 公用工具
    ├── code_for_analysis/              # 上游分析与可视化代码
    │   └── data/                       # 分析使用的数据
    └── reproduction/                  # 新增的复现材料
        ├── README_复现说明.md          # 复现范围、环境和运行说明
        ├── setup.sh                    # 重建隔离环境
        ├── run.sh                      # 一键运行入口
        ├── run_all.py                  # 阶段调度、超时与运行记录
        ├── reproduce_analysis.py       # 已存档数据再分析
        ├── reproduce_models.py         # 已发布模型复测
        ├── reproduce_original.py       # 原版流程复现
        ├── download_inputs.py          # 外部输入下载与哈希校验
        ├── requirements-*.txt          # 两套环境的依赖要求
        ├── environment-*-freeze.txt    # 已运行环境的精确依赖快照
        ├── external/alice_original/    # 原版公开输入、模型与下载清单
        ├── lib/                        # 本机 OpenMP 库及来源说明
        ├── references/                 # 论文补充材料及证据记录
        ├── results/                    # 已保存结果
        │   ├── analysis/               # 分析指标、表格与图形
        │   ├── models/                 # 模型复测指标与预测结果
        │   └── original/               # 原版流程结果与论文对照
        ├── logs/                       # 各阶段运行日志
        ├── history/                    # 复跑前保存的历史结果
        ├── run_receipt_*.json           # 运行回执与产物哈希
        ├── verification.json           # 重复运行验证记录
        └── independent_pearson_check.json # 独立数值核验记录
```

本机的虚拟环境、Python 运行时、安装缓存和嵌套 Git 元数据不属于复现交付文件；环境应通过安装脚本重建。后续项目可与 `ALICE/` 并列存放。

## ALICE 项目与复现范围

- 上游项目：[XHW-ZJU/ALICE](https://github.com/XHW-ZJU/ALICE)。
- 本次复现的上游基线：[e91627ff7fdb9df6637fa1047e8f9f7c9bc51141](https://github.com/XHW-ZJU/ALICE/tree/e91627ff7fdb9df6637fa1047e8f9f7c9bc51141)。
- 论文：[*Mapping AAV Capsid Sequences to Functions through Function-Guided In Silico Evolution*](https://doi.org/10.1016/j.cpblue.2026.100016)。
- 详细材料：[复现说明](ALICE/reproduction/README_复现说明.md)、[复现报告](ALICE/复现报告.md)、[上游 README](ALICE/README.md)。

已有复现覆盖原版流程、ALICE-X 已发布模型的推理复测，以及公开数据与已存档预测的再分析。它不包含从头训练完整模型、完整生成流程或湿实验复现。ALICE-X 复测使用的数据在上游训练中曾作为 `tuning_data`，因此相应结果应理解为调参集复测。

## 安装与运行

在克隆仓库后进入项目目录；如果仓库文件夹采用其他名称，相应调整第一行路径：

```bash
cd project-reproductions/ALICE
bash reproduction/setup.sh
```

脚本需要联网，并在项目内建立 Python 3.10.18 的两套环境：`.venv` 用于分析和 ALICE-X 模型，`.venv-original` 用于加载原版模型。具体依赖见 [models 清单](ALICE/reproduction/requirements-models.txt) 和 [original 清单](ALICE/reproduction/requirements-original.txt)。

已有运行记录来自 Apple M3 / macOS 14.3。`reproduction/lib/libomp.dylib` 适用于该 arm64 Mac 环境；其他平台应使用相应架构的 OpenMP 运行库，参见 [库文件说明](ALICE/reproduction/lib/README.md)。

校验原版流程所需的公开输入和模型：

```bash
.venv/bin/python reproduction/download_inputs.py --verify-only
```

若缺少资源，可去掉 `--verify-only` 从脚本中记录的作者公开来源下载。下载会检查预先固定的文件大小及 SHA-256；外部来源的可用性与下载配额可能影响获取。

运行全部阶段或选择其中一个阶段：

```bash
bash reproduction/run.sh
bash reproduction/run.sh --only analysis
bash reproduction/run.sh --only models
bash reproduction/run.sh --only original
```

默认每个阶段超时为 1800 秒，可用 `--timeout` 调整。再次运行会将已有阶段结果移入 `reproduction/history/`，新结果写入 `reproduction/results/`，并更新运行日志和回执。

## 已有结果与核验记录

仓库保留 2026-10-31 的计算结果和运行记录。本次仓库整理没有重新执行实验；下列状态依据已有回执与验证文件。

| 内容 | 查阅入口 |
| --- | --- |
| 原版流程结果与论文对照 | [结果目录](ALICE/reproduction/results/original/)、[指标记录](ALICE/reproduction/results/original/metrics.json) |
| ALICE-X 已发布模型复测 | [结果目录](ALICE/reproduction/results/models/)、[指标记录](ALICE/reproduction/results/models/metrics.json) |
| 公开数据与已存档预测的再分析 | [结果目录](ALICE/reproduction/results/analysis/)、[指标与来源](ALICE/reproduction/results/analysis/metrics.json) |
| 三阶段运行状态与文件哈希 | [完整运行回执](ALICE/reproduction/run_receipt_all.json) |
| 重复运行一致性检查 | [验证记录](ALICE/reproduction/verification.json) |
| 独立数值核验 | [核验记录](ALICE/reproduction/independent_pearson_check.json) |

已有完整回执记录三个阶段均完成，验证文件记录关键 CSV 在重复运行间逐字节一致。完整解释和适用边界请结合复现说明阅读。存档文件可能保留原机器的绝对路径；当前仓库入口请使用本 README 中的相对链接。

## 来源与许可

`ALICE/` 中的上游代码来自 XHW-ZJU/ALICE，保留其 [MIT 许可证](ALICE/LICENSE) 与版权声明。外部数据、预训练模型、论文补充材料和第三方运行库的来源见各自说明、下载清单及证据记录；使用时请遵循相应来源的许可条件。复现材料不代表上游作者的官方发布。
