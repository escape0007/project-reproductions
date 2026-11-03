# ALICE 本机复现

源码固定在 `e91627ff7fdb9df6637fa1047e8f9f7c9bc51141`。上游跟踪文件保持原样；新增代码、外部数据、日志和结果全部放在本目录。

## 运行

本机已经配置两个项目内隔离环境，终端执行：

```bash
cd '/Users/zhuang/Desktop/项目复现/ALICE'
bash reproduction/run.sh
```

三个阶段依次运行：

| 阶段 | 实际计算 | 结果目录 |
|---|---|---|
| analysis | 五组公开数据、两类测量列的相关性与 65 个已存档 epoch 的再分析 | `results/analysis/` |
| models | 两个已发布 ALICE-X 预测器的真实推理、作者 leaderboard 复测 | `results/models/` |
| original | 原版 ALICE 的 README 八步筛选，使用作者的输入及三个原始模型 | `results/original/` |

可单独运行，例如 `bash reproduction/run.sh --only original`。每阶段默认硬超时为 1800 秒，可用 `--timeout` 更改。每 30 秒记录存活进度，完整标准输出和错误保存在 `logs/`。复跑前会把旧结果移到 `history/`，防止失败时把旧成功结果当作本次产物。运行回执在 `run_receipt_all.json` 或相应单阶段文件。

## 环境

本机为 Apple M3 / 16 GiB RAM / macOS 14.3，两个环境均使用项目内 Python 3.10.18。

- `.venv`：AutoGluon 1.4.0、LightGBM 4.6.0、NumPy 2.1.3、pandas 2.3.2、SciPy 1.15.2、scikit-learn 1.7.1。
- `.venv-original`：scikit-learn 1.0.2、NumPy 1.22.4、pandas 1.5.3、SciPy 1.7.3；用来原样加载作者 2024 年发布的 GradientBoostingRegressor。
- 依赖清单：`requirements-models.txt`、`requirements-original.txt`；精确安装快照见 `environment-*-freeze.txt`。
- macOS 的 LightGBM 使用 `lib/libomp.dylib`，来源与 SHA-256 见 `lib/README.md`。运行器仅为当前子进程设置库路径，不改系统配置。

从新机器重建环境可执行 `bash reproduction/setup.sh`。另一个平台应安装对应架构的 OpenMP；本目录的 dylib 仅用于当前 arm64 Mac。重建命令需要联网。

## 数据与来源

`external/alice_original/` 包含作者公开 Google Drive 提供的输入 CSV 和三个模型，共约 127 MB。`download_manifest.json` 记录文件 ID、大小与 SHA-256。初次获取使用 Google Drive 连接器提供的完整原始文件，再落盘校验；不会上传本地文件。

检查资源：

```bash
.venv/bin/python reproduction/download_inputs.py --verify-only
```

资源缺失时可运行同一命令去掉 `--verify-only` 从作者的公开 Drive 下载；文件内容若与本次固定哈希不同会停止，不自动接受另一版本。该下载备用路径受公开 Drive 的下载配额和权限影响。

论文依据保存在 `references/Document_S1.pdf`，`paper_evidence.json` 记录来源与页码：补充方法第 8 页定义 CP，Table S13 第 61 页列出 N1–N8。相关两页的截图也已保存。

## 复现范围与解释

1. 原版流程直接导入作者的 `cal_pep` 和 EA 评分函数，其余步骤逐项保留原排序、阈值、去重、CSV 往返和特征顺序。只处理路径、显示配置及环境兼容问题；没有修改模型或为凑结果调阈值。结果逐条对照 Table S13，而不是只对照数量。
2. ALICE-X 的 `test_*` 文件在上游训练时被用作 `tuning_data`，所以这里是**调参集复测**，不能解释为独立泛化测试。重复推理必须逐值一致，模型哈希必须保持不变；与上游记录不符时保存实际结果和差值，不覆盖参考数据。
3. Figure 6F 使用综合评分 CP：生产适应度和 hTfR1 测量或预测的算术平均。仓库同时存在 `huTfR1fc_log2_enrichment` 与 `huTfR1_fc_log2_enrichment`；两列结果分别报告。补充方法确认了公式，CSV 列到具体实验的对应仍是基于公开数据和数值的推断。
4. 分析部分使用仓库已有预测，不冒充本次新模型推理。原绘图脚本要求的 `lib2_ALICE_new.csv` 及两列原始计数未提供，因此本次使用现有公开 processed/split_files 数据并标注来源。
5. 本交付采用作者建议的公开输入及已训练权重复现路径，没有从头重训大规模 RoBERTa、SeqGAN、完整 ALICE-X 生成流程或进行湿实验。原仓库 ALICE-X 生成入口还缺 GA/SA 模块，并有 `pd.DtaFrame` 拼写问题；这些未运行模块未擅自重写。

各阶段结果应与根目录 `复现报告.md` 和实际日志一并阅读。论文参考：[Cell Press Blue](https://doi.org/10.1016/j.cpblue.2026.100016)；代码来源：[ALICE](https://github.com/XHW-ZJU/ALICE/tree/e91627ff7fdb9df6637fa1047e8f9f7c9bc51141)。
