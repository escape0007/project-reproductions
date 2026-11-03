#!/usr/bin/env python3
"""Re-evaluate the two bundled ALICE-X AutoGluon models on their tuning sets.

This is pretrained-model inference, not retraining and not an independent test.
The upstream training script passes the files named ``test_*`` as tuning_data.
Run with the project-local Python environment; no download or installation occurs.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import platform
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
RANKING = ROOT / "alice_x" / "ranking"
TASKS = (
    ("fitness", "V_mean__P_mean_log2_enr", "fitness"),
    ("htfr1", "hT_mean__V_mean_log2_enr", "hT_mean__V_mean_log2_enr"),
)
METRIC_COLUMNS = {
    "pearson": ("pearsonr", 1),
    "spearman": ("spearmanr", 1),
    "r2": ("r2", 1),
    "rmse": ("root_mean_squared_error", -1),
    "mae": ("mean_absolute_error", -1),
    "median_absolute_error": ("median_absolute_error", -1),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def metrics(y, pred) -> dict:
    import numpy as np
    from scipy.stats import pearsonr, spearmanr
    from sklearn.metrics import mean_absolute_error, median_absolute_error, r2_score

    y, pred = np.asarray(y, dtype=float), np.asarray(pred, dtype=float)
    if y.shape != pred.shape or not np.isfinite(y).all() or not np.isfinite(pred).all():
        raise ValueError("Targets and predictions must be matching finite vectors")
    return {
        "n": int(len(y)),
        "pearson": float(pearsonr(y, pred).statistic),
        "spearman": float(spearmanr(y, pred).statistic),
        "r2": float(r2_score(y, pred)),
        "rmse": float(np.sqrt(np.mean((y - pred) ** 2))),
        "mae": float(mean_absolute_error(y, pred)),
        "median_absolute_error": float(median_absolute_error(y, pred)),
    }


def clean_data(df, label: str):
    """Apply the numeric-column and finite-row rules in upstream trainer.fit."""
    import numpy as np
    import pandas as pd

    numeric = df.drop(columns=["AA_sequence"], errors="ignore").copy()
    numeric[label] = pd.to_numeric(numeric[label], errors="coerce")
    columns = [c for c in numeric if c == label or np.issubdtype(numeric[c].dtype, np.number)]
    return numeric[columns].replace([np.inf, -np.inf], np.nan).dropna()


def model_hashes(model_dir: Path) -> dict:
    return {str(p.relative_to(ROOT)): sha256(p) for p in sorted(model_dir.rglob("*")) if p.is_file()}


def evaluate_task(short_name: str, label: str, file_part: str, output: Path, tolerance: float) -> dict:
    import numpy as np
    import pandas as pd
    from autogluon.tabular import TabularPredictor
    import matplotlib.pyplot as plt

    started = time.monotonic()
    model_dir = RANKING / "gen_model_research" / "aggregate" / "hTfR1_model" / label
    train_path = RANKING / "datapack" / f"train_{file_part}_data.csv"
    test_path = RANKING / "datapack" / f"test_{file_part}_data.csv"
    reference_path = model_dir / f"{label}_leaderboard.csv"
    before = model_hashes(model_dir)
    train_raw, test_raw = pd.read_csv(train_path), pd.read_csv(test_path)
    train, test = clean_data(train_raw, label), clean_data(test_raw, label)
    if len(test) < 2:
        raise ValueError(f"Insufficient finite evaluation rows for {label}")

    # AutoGluon's loader resets stored Windows paths to the supplied local root.
    # Keep version checks enabled and preserve all fitted parameters and weights.
    predictor = TabularPredictor.load(str(model_dir), verbosity=2)
    if predictor.label != label:
        raise ValueError(f"Checkpoint target mismatch: {predictor.label!r} != {label!r}")
    expected = list(predictor.feature_metadata_in.get_features())
    missing = sorted(set(expected) - set(test.columns))
    if missing:
        raise ValueError(f"Missing fitted-model input features: {missing}")
    if label in expected or "AA_sequence" in expected:
        raise ValueError("Target or sequence identifier unexpectedly present in model inputs")
    x = test[expected]
    y = test[label].to_numpy(dtype=float)
    best_model = predictor.model_best
    available_models = predictor.model_names()

    pred = predictor.predict(x).to_numpy(dtype=float)
    repeat = predictor.predict(x).to_numpy(dtype=float)
    deterministic = bool(np.array_equal(pred, repeat))
    repeat_max_abs_diff = float(np.max(np.abs(pred - repeat)))
    if not deterministic:
        raise AssertionError(f"Repeated inference changed ({repeat_max_abs_diff:g})")
    actual = metrics(y, pred)
    references = pd.read_csv(reference_path)
    comparisons, model_rows = [], []
    prediction_table = pd.DataFrame({
        "source_row_index": test.index,
        "AA_sequence": test_raw.loc[test.index, "AA_sequence"].to_numpy(),
        "observed": y,
        "predicted": pred,
        "residual_observed_minus_predicted": y - pred,
    })
    for name in references["model"].unique():
        if name not in available_models:
            comparisons.append({"model": name, "metric": None, "status": "missing_from_loaded_trainer"})
            continue
        model_pred = pred if name == best_model else predictor.predict(x, model=name).to_numpy(dtype=float)
        model_metrics = metrics(y, model_pred)
        model_rows.append({"model": name, **model_metrics})
        prediction_table[f"predicted_{name}"] = model_pred
        reference = references.loc[references["model"] == name].iloc[0]
        for metric, (column, sign) in METRIC_COLUMNS.items():
            value, baseline = model_metrics[metric], float(reference[column]) * sign
            difference = value - baseline
            comparisons.append({
                "model": name, "metric": metric,
                "recomputed": value, "upstream_reference": baseline,
                "signed_difference": difference,
                "absolute_difference": abs(difference),
                "tolerance": tolerance,
                "status": "matched" if abs(difference) <= tolerance else "different",
            })

    task_out = output / short_name
    task_out.mkdir(parents=True, exist_ok=True)
    prediction_table.to_csv(task_out / "predictions.csv", index=False, float_format="%.17g")
    pd.DataFrame(comparisons).to_csv(task_out / "leaderboard_comparison.csv", index=False, float_format="%.17g")
    pd.DataFrame(model_rows).to_csv(task_out / "model_metrics.csv", index=False, float_format="%.17g")

    fig, ax = plt.subplots(figsize=(6.2, 5.4))
    ax.scatter(y, pred, s=18, alpha=.7, color="#267f99", edgecolors="none")
    limits = [min(y.min(), pred.min()), max(y.max(), pred.max())]
    ax.plot(limits, limits, "--", color="#a8a8a8", linewidth=1)
    ax.set(xlabel="Observed log2 enrichment", ylabel="Predicted log2 enrichment",
           title=f"ALICE-X {short_name}: bundled model replay")
    ax.text(.03, .97, f"n = {len(y)}\nPearson r = {actual['pearson']:.6f}\nRMSE = {actual['rmse']:.6f}",
            transform=ax.transAxes, va="top")
    fig.text(.5, .01, "Upstream tuning set; not an independent test", ha="center", fontsize=9)
    fig.tight_layout(rect=(0, .035, 1, 1))
    fig.savefig(task_out / "predictions_scatter.png", dpi=200)
    fig.savefig(task_out / "predictions_scatter.pdf")
    plt.close(fig)

    train_sequences = set(train_raw.loc[train.index, "AA_sequence"].astype(str).str.strip().str.upper())
    test_sequences = set(test_raw.loc[test.index, "AA_sequence"].astype(str).str.strip().str.upper())
    overlap = sorted(train_sequences & test_sequences)
    unchanged = before == model_hashes(model_dir)
    if not unchanged:
        raise AssertionError("The upstream checkpoint files changed during inference")
    return {
        "task": short_name, "label": label, "status": "completed",
        "experiment_type": "pretrained_model_inference_replay",
        "evaluation_role": "upstream_tuning_set",
        "independent_generalization_test": False,
        "caveat": "s2_auto_gb_regressor.py passes test_data as tuning_data during fitting and HPO.",
        "best_model": best_model, "available_models": available_models,
        "metrics": actual,
        "all_reference_metrics_match": all(r["status"] == "matched" for r in comparisons),
        "reference_comparisons": comparisons,
        "data_audit": {
            "train_rows_raw": int(len(train_raw)), "train_rows_finite": int(len(train)),
            "test_rows_raw": int(len(test_raw)), "test_rows_finite": int(len(test)),
            "dropped_train_rows": int(len(train_raw) - len(train)),
            "dropped_test_rows": int(len(test_raw) - len(test)),
            "model_original_feature_count": len(expected),
            "missing_features": missing,
            "ignored_nonmodel_columns": sorted(set(test.columns) - set(expected) - {label}),
            "train_duplicate_sequences": int(len(train) - len(train_sequences)),
            "test_duplicate_sequences": int(len(test) - len(test_sequences)),
            "train_test_sequence_overlap_count": len(overlap),
            "train_test_overlap_sequences": overlap,
        },
        "validation": {
            "repeat_predictions_exactly_equal": deterministic,
            "repeat_predictions_max_abs_diff": repeat_max_abs_diff,
            "checkpoint_files_unchanged": unchanged,
            "reference_metric_absolute_tolerance": tolerance,
        },
        "input_hashes": {
            str(p.relative_to(ROOT)): sha256(p) for p in (train_path, test_path, reference_path)
        },
        "checkpoint_hashes": before,
        "elapsed_seconds": time.monotonic() - started,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "reproduction/results/models")
    parser.add_argument("--tolerance", type=float, default=1e-6)
    args = parser.parse_args()
    if args.tolerance <= 0:
        parser.error("--tolerance must be positive")
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "reproduction/.cache/matplotlib"))
    os.environ.setdefault("OMP_NUM_THREADS", "2")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
    os.environ.setdefault("MKL_NUM_THREADS", "2")
    import matplotlib
    matplotlib.use("Agg")
    import numpy as np
    np.random.seed(10)
    result = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "command": sys.argv,
        "script_sha256": sha256(Path(__file__)),
        "python": sys.version, "platform": platform.platform(),
        "package_versions": {p: version(p) for p in (
            "autogluon.tabular", "numpy", "pandas", "scipy", "scikit-learn", "lightgbm", "matplotlib"
        )},
        "status": "running", "tasks": [],
    }
    (output / "metrics.json").write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    try:
        for task in TASKS:
            print(f"Evaluating {task[0]} on the original tuning set...", flush=True)
            task_result = evaluate_task(*task, output, args.tolerance)
            result["tasks"].append(task_result)
            (output / "metrics.json").write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
            print(json.dumps({"task": task[0], **task_result["metrics"],
                              "matches_upstream": task_result["all_reference_metrics_match"]}), flush=True)
    except Exception:
        result.update(status="failed", validation_status="incomplete", error=traceback.format_exc())
        (output / "metrics.json").write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
        raise
    result["status"] = "completed"
    result["all_reference_metrics_match"] = all(t["all_reference_metrics_match"] for t in result["tasks"])
    result["validation_status"] = "matched" if result["all_reference_metrics_match"] else "reference_mismatch"
    (output / "metrics.json").write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    print(f"Saved {output / 'metrics.json'}", flush=True)


if __name__ == "__main__":
    main()
