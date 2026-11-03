#!/usr/bin/env python3
"""Reanalyse ALICE's archived numerical results, without training or inference.

The published analysis script references an absent lib2_ALICE_new.csv and absent
virus3_mean/plasmid_mean columns. This program therefore labels its calculation
as an archived-cohort reanalysis, not an exact reproduction of the missing input.
Correlation definitions and the receptor measurement column follow
code_for_analysis/alice_x_predict_power.py. Fitness is retained from the supplied
CSV and checked against log2(V_mean_P_mean_enrichment), rather than pretending
that this ratio is the absent virus3_mean/plasmid_mean measurement.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import r2_score

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "code_for_analysis" / "data"
METHODS = {
    "ALICE-X": "ALICE_x",
    "ALICE": "ALICE",
    "ST + Ranking": "ST_Ranking",
    "ST": "ST",
    "Random mutation": "Mutate",
}
TARGETS = {
    "fitness": ("fitness prediction", "fitness", "Production fitness"),
    "receptor": (
        "hTFR1 enrichment prediction",
        "huTfR1_fc_log2_enrichment",
        "hTfR1 binding",
    ),
    "mean_pair": ("prediction_mean", "measurement_mean", "Arithmetic mean of both targets"),
}
ALTERNATE_TARGETS = {
    "receptor_alternate": (
        "hTFR1 enrichment prediction", "huTfR1fc_log2_enrichment", "Alternate hTfR1 measurement"
    ),
    "mean_pair_alternate": (
        "prediction_mean", "measurement_mean_alternate", "Mean pair with alternate measurement"
    ),
}


def file_record(path: Path) -> dict:
    return {
        "path": str(path.relative_to(ROOT)),
        "bytes": path.stat().st_size,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def to_float(series: pd.Series) -> pd.Series:
    """Keep the original script's string-to-number conversion semantics."""
    text = series.astype(str).str.strip()
    text = text.str.replace("\u2212", "-", regex=False).str.replace(",", "", regex=False)
    number = text.str.extract(r"([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)", expand=False)
    return pd.to_numeric(number, errors="coerce")


def finite_float(value):
    return float(value) if np.isfinite(value) else None


def corr_stats(y_true, y_pred) -> dict:
    """Pearson, Spearman, and regression R²; finite-pair filtering is upstream's."""
    yt, yp = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
    valid = np.isfinite(yt) & np.isfinite(yp)
    yt, yp = yt[valid], yp[valid]
    result = {"n": int(valid.sum()), "excluded_nonfinite": int((~valid).sum())}
    if len(yt) < 2:
        return {**result, "pearson": None, "spearman": None, "r2": None}
    nonconstant = np.ptp(yt) > 0 and np.ptp(yp) > 0
    return {
        **result,
        "pearson": finite_float(stats.pearsonr(yp, yt).statistic) if nonconstant else None,
        "spearman": finite_float(stats.spearmanr(yp, yt).statistic) if nonconstant else None,
        "r2": finite_float(r2_score(yt, yp)),
    }


def save_plot(fig, stem: Path) -> None:
    fig.savefig(stem.with_suffix(".png"), dpi=200, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def analyse_cohorts(outdir: Path) -> tuple[list, list, list]:
    """Analyse supplied processed and with-control cohorts without new filtering."""
    train_path = DATA / "all_train_and_tst_data.csv"
    train = pd.read_csv(train_path)
    train_seqs = set(train["AA_sequence"].astype(str).str.upper().str.strip())
    sources, cohorts, metrics, point_frames = [file_record(train_path)], [], [], []
    processed_frames = {}
    for cohort_set, folder, suffix in [
        ("processed", "processed", ""),
        ("with_control", "split_files", "_with_Control"),
    ]:
        for method, basename in METHODS.items():
            path = DATA / folder / f"lib2_{basename}{suffix}.csv"
            df = pd.read_csv(path)
            required = ["AA_sequence", "fitness", "fitness prediction",
                        "hTFR1 enrichment prediction", "huTfR1_fc_log2_enrichment",
                        "V_mean_P_mean_enrichment", "huTfR1fc_log2_enrichment"]
            missing = sorted(set(required) - set(df.columns))
            if missing:
                raise ValueError(f"{path}: required columns missing: {missing}")
            df["AA_sequence"] = df["AA_sequence"].astype(str).str.upper().str.strip()
            for column in required[1:]:
                df[column] = to_float(df[column])
            with np.errstate(divide="ignore", invalid="ignore"):
                logratio = np.log2(df["V_mean_P_mean_enrichment"])
            delta = (df["fitness"] - logratio).replace([np.inf, -np.inf], np.nan).dropna()
            df["prediction_mean"] = (df["fitness prediction"] + df["hTFR1 enrichment prediction"]) / 2
            df["measurement_mean"] = (df["fitness"] + df["huTfR1_fc_log2_enrichment"]) / 2
            df["measurement_mean_alternate"] = (df["fitness"] + df["huTfR1fc_log2_enrichment"]) / 2
            duplicated = df["AA_sequence"].duplicated(keep=False)
            cohorts.append({
                "cohort_set": cohort_set,
                "method": method,
                "rows": len(df),
                "unique_sequences": int(df["AA_sequence"].nunique()),
                "duplicate_rows": int(duplicated.sum()),
                "overlap_rows_with_supplied_train_and_test": int(df["AA_sequence"].isin(train_seqs).sum()),
                "fitness_logratio_finite_pairs": len(delta),
                "fitness_logratio_max_absolute_difference": finite_float(delta.abs().max()) if len(delta) else None,
                "source": str(path.relative_to(ROOT)),
                "selection": "All supplied rows, original order; no deduplication or score-based filtering.",
            })
            for target, (prediction, measurement, _) in {**TARGETS, **ALTERNATE_TARGETS}.items():
                metrics.append({"cohort_set": cohort_set, "method": method, "target": target,
                                "prediction_column": prediction, "measurement_column": measurement,
                                **corr_stats(df[measurement], df[prediction])})
            points = df[["AA_sequence", "fitness", "fitness prediction", "huTfR1_fc_log2_enrichment",
                         "huTfR1fc_log2_enrichment", "hTFR1 enrichment prediction", "prediction_mean",
                         "measurement_mean", "measurement_mean_alternate"]].copy()
            points.insert(0, "source_row_1based", np.arange(1, len(points) + 1))
            points.insert(0, "method", method)
            points.insert(0, "cohort_set", cohort_set)
            point_frames.append(points)
            sources.append(file_record(path))
            if cohort_set == "processed":
                processed_frames[method] = df

    pd.DataFrame(metrics).to_csv(outdir / "correlation_metrics.csv", index=False)
    pd.DataFrame(cohorts).to_csv(outdir / "cohort_audit.csv", index=False)
    pd.concat(point_frames, ignore_index=True).to_csv(outdir / "analysed_points.csv", index=False)

    colors = ["#176B87", "#C46A3B", "#668653", "#80639C", "#9A815A"]
    for method, df in processed_frames.items():
        fig, axes = plt.subplots(1, 3, figsize=(14, 4.4), constrained_layout=True)
        for ax, (_, (prediction, measurement, title)) in zip(axes, TARGETS.items()):
            valid = df[[prediction, measurement]].replace([np.inf, -np.inf], np.nan).dropna()
            ax.scatter(valid[prediction], valid[measurement], s=15, alpha=.55, color=colors[0], linewidths=0)
            result = corr_stats(df[measurement], df[prediction])
            text = (f"Pearson r = {result['pearson']:.4f}\nSpearman ρ = {result['spearman']:.4f}"
                    f"\nR² = {result['r2']:.4f}\nn = {result['n']}")
            ax.text(.04, .97, text, transform=ax.transAxes, va="top", fontsize=9,
                    bbox={"facecolor": "white", "alpha": .85, "edgecolor": "none"})
            ax.set(xlabel="Archived prediction (log2)", ylabel="Supplied measurement (log2)", title=title)
            ax.spines[["top", "right"]].set_visible(False)
        fig.suptitle(f"{method} — supplied processed cohort; no retraining", fontsize=14)
        save_plot(fig, outdir / f"scatter_{METHODS[method]}")

    # Report both available measurement channels; do not silently choose a column
    # just because its correlation agrees with the rounded number in the paper.
    alice_x = processed_frames["ALICE-X"]
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.9), constrained_layout=True)
    for ax, measured, title in zip(
        axes, ["measurement_mean", "measurement_mean_alternate"],
        ["Upstream script: huTfR1_fc_log2_enrichment", "Alternative: huTfR1fc_log2_enrichment"],
    ):
        valid = alice_x[["prediction_mean", measured]].replace([np.inf, -np.inf], np.nan).dropna()
        result = corr_stats(valid[measured], valid["prediction_mean"])
        ax.scatter(valid["prediction_mean"], valid[measured], s=16, alpha=.55, color=colors[0], linewidths=0)
        ax.text(.04, .97, f"Pearson r = {result['pearson']:.6f}\nn = {result['n']}",
                transform=ax.transAxes, va="top", bbox={"facecolor": "white", "alpha": .9, "edgecolor": "none"})
        ax.set(title=title, xlabel="Mean of two archived predictions", ylabel="Mean of fitness and selected measurement")
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("ALICE-X measurement-column sensitivity\nCP formula: Methods S1 section 3; CSV column mapping remains an inference", fontsize=12)
    save_plot(fig, outdir / "figure6f_measurement_sensitivity")

    fig, ax = plt.subplots(figsize=(6.2, 5.8), constrained_layout=True)
    alt = corr_stats(alice_x["measurement_mean_alternate"], alice_x["prediction_mean"])
    ax.scatter(alice_x["prediction_mean"], alice_x["measurement_mean_alternate"], s=20, alpha=.6, color=colors[0], linewidths=0)
    ax.text(.04, .97, f"Pearson r = {alt['pearson']:.6f}\nSpearman ρ = {alt['spearman']:.4f}\nR² = {alt['r2']:.4f}\nn = {alt['n']}",
            transform=ax.transAxes, va="top", bbox={"facecolor": "white", "alpha": .9, "edgecolor": "none"})
    ax.set(title="ALICE-X: Figure 6F-compatible numerical reanalysis",
           xlabel="Mean(fitness prediction, hTFR1 prediction)",
           ylabel="Mean(fitness, huTfR1fc_log2_enrichment)")
    ax.spines[["top", "right"]].set_visible(False)
    fig.text(.5, -.025, "Archived data; alternate measurement column, not the original plotting script's column.", ha="center", fontsize=8)
    save_plot(fig, outdir / "figure6f_compatible_alternate_measurement")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
    for ax, column, title in zip(axes, ["fitness", "huTfR1_fc_log2_enrichment"],
                                 ["Production fitness", "hTfR1 binding"]):
        arrays = [frame[column].replace([np.inf, -np.inf], np.nan).dropna() for frame in processed_frames.values()]
        boxes = ax.boxplot(arrays, patch_artist=True, tick_labels=list(processed_frames), showfliers=False)
        for box, color in zip(boxes["boxes"], colors):
            box.set_facecolor(color)
            box.set_alpha(.7)
        ax.tick_params(axis="x", labelrotation=20)
        ax.set(title=title, ylabel="Supplied measurement (log2)")
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Archived cohort distributions — descriptive comparison; outliers hidden only in plot", fontsize=12)
    save_plot(fig, outdir / "archived_cohort_distributions")
    return metrics, cohorts, sources


def analyse_trajectories(outdir: Path) -> tuple[list, list]:
    """Summarise original-order first 500 rows, as in plot_evolve_trajectory.py.

    These curves summarise archived predictions; they do not rerun evolution.
    No confidence intervals or causal claims are inferred from dependent rows.
    """
    summaries, sources = [], []
    for group in ["stage2_sum", "stage2_min", "alice_designed_data"]:
        files = []
        for path in (DATA / group).glob("seq-epoch-*_pred.csv"):
            match = re.fullmatch(r"seq-epoch-(\d+)_pred\.csv", path.name)
            if match:
                files.append((int(match[1]), path))
        for epoch, path in sorted(files):
            df = pd.read_csv(path, nrows=500)
            record = {"group": group, "epoch": epoch, "rows": len(df),
                      "unique_sequences": int(df["seq"].nunique())}
            for column in ["fitness_pred", "htfr1_pred", "ref_dist", "div_prev", "reward_signal", "final_score", "score"]:
                if column not in df.columns:
                    continue
                values = pd.to_numeric(df[column], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
                record[f"{column}_n"] = len(values)
                record[f"{column}_mean"] = finite_float(values.mean())
                record[f"{column}_median"] = finite_float(values.median())
            summaries.append(record)
            sources.append(file_record(path))
    frame = pd.DataFrame(summaries)
    frame.to_csv(outdir / "precomputed_epoch_summary.csv", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    for ax, column, label in zip(axes, ["fitness_pred_mean", "htfr1_pred_mean"],
                                  ["Mean predicted fitness", "Mean predicted hTfR1 binding"]):
        for group, part in frame.groupby("group", sort=False):
            ax.plot(part["epoch"], part[column], marker="o", markersize=3, label=group)
        ax.set(xlabel="Archived epoch", ylabel=label + " (log2)")
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(fontsize=8)
    fig.suptitle("Precomputed trajectories — first 500 rows per archived epoch; not regenerated", fontsize=12)
    save_plot(fig, outdir / "precomputed_trajectories")
    return summaries, sources


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "reproduction/results/analysis")
    args = parser.parse_args()
    outdir = args.output_dir.resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    np.random.seed(2026)  # Analyses are deterministic; no subsampling is performed.
    plt.rcParams.update({"font.family": "DejaVu Sans", "axes.titlesize": 11, "axes.labelsize": 10,
                         "pdf.fonttype": 42, "svg.fonttype": "none"})
    metrics, cohorts, sources = analyse_cohorts(outdir)
    trajectory_summary, trajectory_sources = analyse_trajectories(outdir)
    upstream = ROOT / "code_for_analysis/alice_x_predict_power.py"
    result = {
        "status": "COMPLETED_ARCHIVED_COHORT_REANALYSIS",
        "not_claimed": ["Model training", "Fresh checkpoint inference", "Regenerated evolution", "Exact reproduction of missing original plotting input", "Wet-lab replication"],
        "random_seed": 2026,
        "method": {
            "upstream_script": str(upstream.relative_to(ROOT)),
            "missing_upstream_input": "code_for_analysis/data/split_files/lib2_ALICE_new.csv",
            "missing_raw_columns": ["virus3_mean", "plasmid_mean"],
            "fitness": "Supplied fitness retained; checked against log2(V_mean_P_mean_enrichment). This is not asserted to equal the missing virus3_mean/plasmid_mean ratio.",
            "receptor_measurement": "huTfR1_fc_log2_enrichment (the exact upstream column, distinct from huTfR1fc_log2_enrichment)",
            "alternate_measurement_sensitivity": "Also report huTfR1fc_log2_enrichment and its mean pair separately. The mean-pair correlation on the processed ALICE-X cohort rounds to the paper's Fig. 6F value of 0.523, but matching a rounded number alone does not establish assay correspondence.",
            "mean_pair": "(fitness + receptor)/2 for both observations and archived predictions",
            "correlation": "Pairwise finite filtering; Pearson, Spearman, sklearn.metrics.r2_score(y_true,y_pred). R² is not squared Pearson r.",
            "cohort_selection": "All rows in each supplied file; no removal of controls or duplicates beyond upstream's separate processed cohorts.",
            "trajectory_selection": "First 500 rows in original order per *_pred.csv; not sorted by a score. This matches the row-limit convention in plot_evolve_trajectory.py.",
        },
        "limitations": [
            "All prediction values were provided by the repository; this script performs no new model inference.",
            "The original plotting input is absent, so differences from paper correlations cannot be attributed solely to numerical implementation.",
            "Two distinct receptor measurement columns coexist. Both are analysed, with explicit names. This program has not established which assay maps to which column from experimental metadata; numeric agreement with the paper is not proof of that mapping.",
            "Sequence overlap is checked only against the supplied combined 6387-row training/test file. Zero overlap does not prove independence from every model training or selection dataset.",
            "Cohorts contain different selected sequences. Cross-cohort correlations are descriptive and not a controlled comparison of predictive model quality.",
            "Archived sequence rows and epochs are dependent; no statistical significance or causal improvement is claimed.",
            "R² can be negative even when Pearson correlation is positive; no calibration model is fitted here.",
        ],
        "cohorts": cohorts,
        "metrics": metrics,
        "trajectory_epochs": len(trajectory_summary),
        "sources": sources + trajectory_sources + [file_record(upstream)],
        "software": {"python": platform.python_version(), "platform": platform.platform(),
                     **{name: importlib.metadata.version(name) for name in ["numpy", "pandas", "scipy", "matplotlib", "scikit-learn"]}},
    }
    reference_path = ROOT / "reproduction/references/paper_evidence.json"
    if reference_path.exists():
        reference = json.loads(reference_path.read_text())
        result["paper_formula_evidence"] = reference["figure6f"]
        result["sources"].append(file_record(reference_path))
    (outdir / "metrics.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    for row in metrics:
        if row["cohort_set"] == "processed":
            print(f"{row['method']:16s} {row['target']:10s} n={row['n']:3d} Pearson={row['pearson']:.6f} R2={row['r2']:.6f}")
    print(f"Saved {len(metrics)} correlation summaries and {len(trajectory_summary)} archived epochs to {outdir}")
    print("Scope: archived numerical reanalysis only; no new training or inference.")


if __name__ == "__main__":
    main()
