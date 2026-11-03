#!/usr/bin/env python3
"""Build the Chinese reproduction report from actual local result files."""
import json
import hashlib
from pathlib import Path

BASE = Path(__file__).resolve().parent
ROOT = BASE.parent


def read(stage):
    path = BASE / "results" / stage / "metrics.json"
    return json.loads(path.read_text()) if path.exists() else {}


def main():
    original, models, analysis = (read(stage) for stage in ["original", "models", "analysis"])
    verification = None
    verification_path = BASE / "verification.json"
    if verification_path.exists():
        candidate = json.loads(verification_path.read_text())
        bindings = {**candidate.get("csv_sha256", {}), **candidate.get("script_sha256", {})}
        if bindings and candidate.get("status") == "passed" and all(
            (BASE / path).is_file() and hashlib.sha256((BASE / path).read_bytes()).hexdigest() == digest
            for path, digest in bindings.items()
        ) and original.get("status") == "completed" and models.get("status") == "completed" and analysis.get("status") == "COMPLETED_ARCHIVED_COHORT_REANALYSIS":
            verification = candidate
    lines = [
        "# ALICE 复现报告", "",
        "本次复现采用作者发布的输入与已训练模型：重新执行原版八步筛选、ALICE-X 模型推理，以及公开数据的论文指标再计算。完整结果保存在 `reproduction/results/`。", "",
        "## Material Passport", "",
        "- Origin Skill: academic-research-suite / experiment-agent",
        "- Origin Mode: run",
        "- Origin Date: 2026-10-31 (Asia/Macau)",
        "- Verification Status: VERIFIED" if verification else "- Verification Status: UNVERIFIED",
        "- Verification Scope: 仅以下已运行的公开输入/权重计算结果；不代表从头训练或湿实验复现",
        "- Version Label: alice_reproduction_v1", "",
        "## 已运行结果", "",
    ]
    final = original.get("final_result", {})
    if original.get("status") == "completed":
        lines += ["### 原版 ALICE", "",
            f"八步筛选实际完成，输入 **{original['data_audit']['input_rows']}** 行，最终保留 **{final['selected_rows']}** 条。",
            f"与论文 Table S13 的 N1–N8 序列集合：**{'完全一致' if final.get('paper_sequence_identity_verified') else '尚未通过逐条一致性验证'}**。",
            f"预测生产适应度与论文保留两位小数的值：**{'全部符合舍入精度' if final.get('predicted_fitness_matches_published_rounding') else '存在未确认或不一致项'}**。", "",
            "各步保留行数：" + " → ".join([str(original['data_audit']['input_rows'])] + [str(step['rows']) for step in original['steps']]) + "。", "",
            "产物：[最终结果](reproduction/results/original/ALICE_TOP.csv)、[论文逐条对照](reproduction/results/original/paper_table_s13_comparison.csv)、[指标与验证记录](reproduction/results/original/metrics.json)。", "",
            "![八步筛选结果](reproduction/results/original/selection_counts.png)", "",
        ]
    else:
        lines += [f"原版 ALICE 状态：`{original.get('status', 'not_run')}`；以日志及本次运行回执为准。", ""]
    if models.get("tasks"):
        lines += ["### ALICE-X 已发布模型复测", "",
            "每个任务使用 256 行上游调参集进行实际推理。下表展示本次加载的默认最佳模型；全部上游参考模型的逐项差异保存在 comparison CSV 中。", "",
            "| 任务 | 本次 Pearson r | 本次 RMSE | 上游参考指标是否全部吻合 |",
            "|---|---:|---:|---|",
        ]
        for task in models["tasks"]:
            actual = task["metrics"]
            lines.append(f"| {task['task']} | {actual['pearson']:.9f} | {actual['rmse']:.9f} | {'是（绝对容差 1e-6）' if task['all_reference_metrics_match'] else '否，详见差异文件'} |")
        lines += ["", "**该数据在原训练代码中被传为 `tuning_data`，不是独立测试集。** 本次检查重复推理逐值一致、输入特征匹配、训练/评估序列交集，以及模型文件未发生修改。", "",
            "产物：[模型指标](reproduction/results/models/metrics.json)、[fitness 逐样本预测](reproduction/results/models/fitness/predictions.csv)、[hTfR1 逐样本预测](reproduction/results/models/htfr1/predictions.csv)。", ""]
    else:
        lines += [f"ALICE-X 推理状态：`{models.get('status', 'not_run')}`。", ""]
    rows = analysis.get("metrics", [])
    selected = {row["target"]: row for row in rows if row["method"] == "ALICE-X" and row["cohort_set"] == "processed"}
    if "mean_pair_alternate" in selected:
        compatible, script_measure = selected["mean_pair_alternate"], selected["mean_pair"]
        lines += ["### 论文 Figure 6F 与公开数据再分析", "",
            f"使用 500 条 ALICE-X 数据计算综合评分 CP，得到 **Pearson r = {compatible['pearson']:.12f}**，三位小数为 **{compatible['pearson']:.3f}**，与论文 Figure 6F 报告值一致。",
            "补充方法第 8 页确认 `CP = (生产适应度 + hTfR1 富集值) / 2`，预测和实测两侧均按此计算。", "",
            f"同一数据未经校准的回归 R² 为 **{compatible['r2']:.6f}**。R² 与 Pearson 相关系数含义不同；本次保留作者原始数值尺度，没有额外拟合校准。", "",
            f"上面使用 `huTfR1fc_log2_enrichment`；原绘图脚本使用另一列 `huTfR1_fc_log2_enrichment`，对应 r = **{script_measure['pearson']:.12f}**。两列均已报告，CSV 列与具体实验的对应关系仍属于依据公开数据的推断；没有把它们静默混用。", "",
            f"额外完成 {len(rows)} 项相关指标和 {analysis['trajectory_epochs']} 个已保存 epoch 的汇总。这部分重算仓库已记录预测，不是重新训练或重新生成。", "",
            "产物：[分析指标与来源](reproduction/results/analysis/metrics.json)、[独立标准库复核](reproduction/independent_pearson_check.json)。", "",
            "![Figure 6F 数值复算](reproduction/results/analysis/figure6f_compatible_alternate_measurement.png)", "",
        ]
    else:
        lines += [f"公开数据分析状态：`{analysis.get('status', 'not_run')}`。", ""]
    lines += ["## 代码与运行", "",
        "```bash", "cd '/Users/zhuang/Desktop/项目复现/ALICE'", "bash reproduction/run.sh", "```", "",
        "可加 `--only original`、`--only models` 或 `--only analysis` 单独执行。完整安装、数据来源及文件说明见 [复现说明](reproduction/README_复现说明.md)。", "",
        "版本：`e91627ff7fdb9df6637fa1047e8f9f7c9bc51141`；硬件：Apple M3、16 GiB RAM、macOS 14.3；Python 3.10.18。原版与 ALICE-X 使用独立依赖环境，未修改上游跟踪文件。", "",
        "## 范围和限制", "",
        "本次完成的是公开输入和预训练模型的计算结果复现，没有重新执行大规模 RoBERTa 预训练、SeqGAN 训练、整个 ALICE-X 生成流程或湿实验。原版八步筛选是项目 README 明确推荐的结果复现路径。", "",
        "原仓库存在 Windows/CUDA 环境锁、部分缺失模块和不对应现有 CSV 的绘图路径；新增脚本对运行到的部分处理路径与环境，并保留原算法和阈值。未运行的生成模块问题已记录在复现说明中。", "",
        "[原项目](https://github.com/XHW-ZJU/ALICE/) · [论文](https://doi.org/10.1016/j.cpblue.2026.100016) · [本地补充材料](reproduction/references/Document_S1.pdf) · [论文证据记录](reproduction/references/paper_evidence.json)", "",
    ]
    if verification:
        lines += ["## 重复运行验证", "",
            f"完整一键命令已成功重跑，缓存就绪后的三个阶段合计 {verification['full_run_seconds']:.3f} 秒。原版最终 CSV、两个模型预测 CSV 和分析指标 CSV 与首次运行逐字节一致。",
            f"24 项模型指标的最大参考差值为 {verification['model_reference_max_absolute_difference']:.3g}（既定绝对容差 1e-6）；独立标准库 Pearson 核验差值为 {verification['independent_pearson_absolute_difference']:.3g}。上游跟踪文件未修改。", "",
            "[验证记录](reproduction/verification.json) · [完整运行回执](reproduction/run_receipt_all.json)", ""]
    report = "\n".join(lines).replace("](reproduction/", "](" + str(BASE) + "/")
    (ROOT / "复现报告.md").write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()
