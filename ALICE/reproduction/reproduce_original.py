#!/usr/bin/env python3
"""Replay the original ALICE README's eight-step final selection pipeline.

Uses the author's downloaded 1,000-row input and three sklearn 1.0.2 models.
This is pretrained-model inference and final selection, not model retraining,
upstream sequence generation, or wet-lab validation. No downloads occur here.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
from importlib.metadata import PackageNotFoundError, version
import json
import os
from pathlib import Path
import pickle
import platform
import random
import sys
import time
import traceback
import warnings

ROOT = Path(__file__).resolve().parents[1]
ORIGINAL = ROOT / "alice" / "ALICE"
STEP5 = ORIGINAL / "step5_elite_ascendancy"
INPUT_NAME = "RoBERTa_SeqGAN_RankFiltPro_FE_1000.csv"
PREDICTION_STEPS = (
    ("Prod_Fit_model.sav", "Prod_Fit", "Prod_Fit_" + INPUT_NAME),
    ("LY6A_model.sav", "Target_LY6A_Enri", "LY6A_model_Prod_Fit_" + INPUT_NAME),
    ("LY6C1_model.sav", "Target_LY6C1_Enri", "LY6C1_model_LY6A_model_Prod_Fit_" + INPUT_NAME),
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def package_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def load_source(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load upstream module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_json(path: Path, obj: dict) -> None:
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def run(input_dir: Path, output: Path, report: dict) -> None:
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt

    started = time.monotonic()
    input_path = input_dir / INPUT_NAME
    model_paths = [input_dir / name for name, _, _ in PREDICTION_STEPS]
    inputs = [input_path, *model_paths]
    for path in inputs:
        if not path.is_file():
            raise FileNotFoundError(f"Required author-provided artifact is missing: {path}")
    original_sources = sorted(STEP5.glob("*.py")) + [ORIGINAL / "tools" / "parameters.py"]
    original_sources += sorted((ORIGINAL / "tools" / "featured_data_generated").glob("*.py"))
    report["input_sha256"] = {path.name: sha256(path) for path in inputs}
    report["upstream_source_sha256"] = {
        str(path.relative_to(ROOT)): sha256(path) for path in original_sources
    }
    report["input_directory"] = str(input_dir)
    report["output_directory"] = str(output)

    # Import the author's exact descriptor implementation. The path is derived
    # from this file, so execution does not depend on the current working directory.
    sys.path.insert(0, str(ORIGINAL / "tools"))
    parameters = load_source("alice_original_parameters", ORIGINAL / "tools" / "parameters.py")
    ea = load_source("alice_original_elite_ascendancy", STEP5 / "6-EliteAscendancy.py")

    df = pd.read_csv(input_path)
    if not {"AA_sequence", "FE_score"}.issubset(df.columns):
        raise ValueError("The original input must contain AA_sequence and FE_score")
    if df.empty or df["AA_sequence"].isna().any() or not np.isfinite(df["FE_score"]).all():
        raise ValueError("Input contains empty sequences, no rows, or non-finite FE scores")
    # Validate, but never uppercase, truncate, deduplicate or replace input rows.
    alphabet = set("ARNDCEQGHILKMFPSTWYV")
    invalid = [int(i) for i, seq in enumerate(df["AA_sequence"]) if not isinstance(seq, str)
               or len(seq) < 6 or not set(seq).issubset(alphabet)]
    if invalid:
        raise ValueError(f"Input has invalid descriptor sequences at row indices {invalid[:20]}")
    report["data_audit"] = {
        "input_rows": int(len(df)),
        "unique_input_sequences": int(df["AA_sequence"].nunique()),
        "duplicate_input_rows": int(df.duplicated(subset="AA_sequence").sum()),
        "input_sequence_lengths": sorted(int(x) for x in df["AA_sequence"].str.len().unique()),
        "expected_author_input_rows": 1000,
        "author_input_row_count_matches": len(df) == 1000,
    }
    input_copy = output / "00_input.csv"
    df.to_csv(input_copy, index=False)

    feature_started = time.monotonic()
    features = pd.DataFrame([parameters.cal_pep(seq) for seq in df["AA_sequence"].tolist()])
    values = features.values  # Preserve original insertion-order feature columns.
    if not np.isfinite(values).all():
        raise ValueError("Author descriptor function produced non-finite values")
    features.to_csv(output / "features.csv", index=False)
    write_json(output / "feature_columns.json", {"columns": features.columns.tolist()})
    report["features"] = {
        "rows": int(features.shape[0]), "columns": int(features.shape[1]),
        "finite": True, "implementation": "alice/ALICE/tools/parameters.py:cal_pep",
        "column_order": "Unchanged dict insertion order, as in upstream pd.DataFrame(seqs_desc).values",
        "elapsed_seconds": time.monotonic() - feature_started,
    }
    report["steps"] = []

    def record_step(number: int, name: str, path: Path, frame, elapsed: float, **extra) -> None:
        item = {"step": number, "name": name, "rows": int(len(frame)),
                "unique_sequences": int(frame["AA_sequence"].nunique()),
                "output_file": path.name, "sha256": sha256(path),
                "elapsed_seconds": elapsed, **extra}
        report["steps"].append(item)
        write_json(output / "metrics.json", report)
        print(f"Step {number}: {name}: {len(frame)} rows -> {path.name}", flush=True)

    previous = input_path
    for number, (model_name, column, output_name) in enumerate(PREDICTION_STEPS, 1):
        step_started = time.monotonic()
        random.seed(88)
        np.random.seed(88)
        df = pd.read_csv(previous)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            # These are explicitly downloaded author artifacts. Do not load
            # replacement pickle files from an untrusted source.
            with (input_dir / model_name).open("rb") as handle:
                model = pickle.load(handle)
            expected = getattr(model, "n_features_in_", None)
            if expected is not None and int(expected) != values.shape[1]:
                raise ValueError(f"{model_name}: expected {expected} features, got {values.shape[1]}")
            pred = np.asarray(model.predict(values), dtype=float)
            repeated = np.asarray(model.predict(values), dtype=float)
        if pred.shape != (len(df),) or not np.isfinite(pred).all():
            raise ValueError(f"{model_name}: invalid predictions")
        if not np.array_equal(pred, repeated):
            raise AssertionError(f"{model_name}: repeated predictions are not exactly equal")
        df[column] = pred.tolist()
        path = output / output_name
        df.to_csv(path, index=False)
        record_step(number, column, path, df, time.monotonic() - step_started,
                    model_class=f"{type(model).__module__}.{type(model).__name__}",
                    model_file=model_name,
                    model_expected_features=int(expected) if expected is not None else None,
                    repeated_predictions_exactly_equal=True,
                    prediction_min=float(pred.min()), prediction_max=float(pred.max()),
                    prediction_mean=float(pred.mean()),
                    warnings=[str(w.message) for w in caught])
        previous = path
        del model

    step_started = time.monotonic()
    df = pd.read_csv(previous).sort_values(by="FE_score", ascending=False).head(500)
    path = output / "FE_score_Top500.csv"
    df.to_csv(path, index=False)
    record_step(4, "FE score top 500", path, df, time.monotonic() - step_started)

    step_started = time.monotonic()
    df = pd.read_csv(path).sort_values(by="Prod_Fit", ascending=False).head(100)
    path = output / "Prod_Fit_Top100.csv"
    df.to_csv(path, index=False)
    record_step(5, "Production fitness top 100", path, df, time.monotonic() - step_started)

    step_started = time.monotonic()
    ea_path = output / "EA_Top.csv"
    normalization_columns = ["Target_LY6A_Enri", "Target_LY6C1_Enri", "Prod_Fit", "FE_score"]
    if not np.isfinite(df[normalization_columns].values).all() or (df[normalization_columns].max() == df[normalization_columns].min()).any():
        raise ValueError("Original EA min-max normalization requires finite, nonconstant columns")
    # Calling process_data bypasses only pandas display options and path setup.
    # Normalization, the exact score formula, threshold and sorting are upstream.
    ea.process_data(str(path), str(ea_path))
    df = pd.read_csv(ea_path)
    record_step(6, "Original EA score threshold", ea_path, df, time.monotonic() - step_started)

    step_started = time.monotonic()
    df = pd.read_csv(ea_path).drop_duplicates(subset="AA_sequence")
    df["multi_tar"] = df["Target_LY6C1_Enri"] * 0.5 + df["Target_LY6A_Enri"] * 0.5
    df = df.sort_values(by="Target_LY6C1_Enri", ascending=False).reset_index(drop=True)
    df = df[df["multi_tar"] >= -1]
    path = output / "multi_tar_TOP.csv"
    df.to_csv(path, index=False)
    record_step(7, "Original joint-target filter", path, df, time.monotonic() - step_started)

    step_started = time.monotonic()
    df = pd.read_csv(path)
    for excluded in ("IRQGYSS", "LRVGYSS", "LRAGYSS"):
        df = df[~df["AA_sequence"].str.contains(excluded, na=False)]
    path = output / "ALICE_TOP.csv"
    df.to_csv(path, index=False)
    record_step(8, "Original reported-sequence exclusion", path, df, time.monotonic() - step_started)

    counts = [{"stage": "Input", "rows": report["data_audit"]["input_rows"]}]
    counts += [{"stage": f"Step {s['step']}", "rows": s["rows"]} for s in report["steps"]]
    pd.DataFrame(counts).to_csv(output / "selection_counts.csv", index=False)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    bars = ax.bar([c["stage"] for c in counts], [c["rows"] for c in counts], color="#337f91")
    for bar, count in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 10,
                str(count["rows"]), ha="center", fontsize=9)
    ax.set(ylabel="Retained input rows", title="Original ALICE: eight-step selection replay")
    ax.set_ylim(0, max(c["rows"] for c in counts) * 1.13)
    fig.text(.5, .01, "Author input and pretrained models; no retraining or wet-lab validation", ha="center", fontsize=9)
    fig.tight_layout(rect=(0, .04, 1, 1))
    fig.savefig(output / "selection_counts.png", dpi=200)
    fig.savefig(output / "selection_counts.pdf")
    plt.close(fig)

    report["final_result"] = {
        "selected_rows": int(len(df)), "unique_selected_sequences": int(df["AA_sequence"].nunique()),
        "readme_reported_sequence_count": 8,
        "readme_sequence_count_matches": len(df) == 8,
        "paper_sequence_identity_verified": False,
        "paper_sequence_identity_note": "No independent paper sequence ground-truth comparison was performed.",
        "selected_file": "ALICE_TOP.csv",
    }
    reference_path = ROOT / "reproduction/references/paper_evidence.json"
    if reference_path.exists():
        reference = json.loads(reference_path.read_text())
        published = pd.DataFrame(reference["original_eight"]["records"])
        expected_sequences = set(published["sequence"])
        observed_sequences = set(df["AA_sequence"])
        matched = expected_sequences == observed_sequences and len(df) == len(published)
        comparison = published.merge(df, left_on="sequence", right_on="AA_sequence", how="left", validate="one_to_one")
        comparison["selected_in_replay"] = comparison["AA_sequence"].notna()
        comparison["fitness_absolute_difference"] = (comparison["Prod_Fit"] - comparison["predicted_production_fitness_rounded"]).abs()
        comparison["fitness_matches_published_rounding"] = comparison["fitness_absolute_difference"] <= 0.00500001
        comparison.to_csv(output / "paper_table_s13_comparison.csv", index=False)
        report["final_result"].update({
            "paper_sequence_identity_verified": matched,
            "paper_sequence_identity_note": "Exact set comparison with AAV.ALICE-N1–N8 in supplementary Table S13, PDF page 61.",
            "paper_reference_sha256": sha256(reference_path),
            "paper_source_sha256": reference["sha256"],
            "missing_paper_sequences": sorted(expected_sequences - observed_sequences),
            "extra_selected_sequences": sorted(observed_sequences - expected_sequences),
            "predicted_fitness_matches_published_rounding": bool(comparison["fitness_matches_published_rounding"].all()),
        })
    after = {path.name: sha256(path) for path in inputs}
    report["validation"] = {"input_and_model_files_unchanged": report["input_sha256"] == after}
    if not report["validation"]["input_and_model_files_unchanged"]:
        raise AssertionError("An author input or model file changed during inference")
    report["elapsed_seconds"] = time.monotonic() - started
    report["status"] = "completed"
    write_json(output / "metrics.json", report)
    print(json.dumps(report["final_result"], ensure_ascii=False), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=ROOT / "reproduction/external/alice_original")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "reproduction/results/original_alice")
    args = parser.parse_args()
    output, input_dir = args.output_dir.resolve(), args.input_dir.resolve()
    if output == input_dir:
        parser.error("Input and output directories must differ")
    output.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "reproduction/.cache/matplotlib"))
    os.environ.setdefault("OMP_NUM_THREADS", "2")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
    os.environ.setdefault("MKL_NUM_THREADS", "2")
    report = {
        "status": "running", "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "command": sys.argv, "script_sha256": sha256(Path(__file__)),
        "python": sys.version, "platform": platform.platform(),
        "package_versions": {p: package_version(p) for p in (
            "numpy", "pandas", "scipy", "scikit-learn", "matplotlib"
        )},
        "experiment_type": "original_alice_pretrained_inference_and_eight_step_selection",
        "expected_model_sklearn_version": "1.0.2",
        "algorithm_changes": [],
        "execution_adaptations": [
            "Resolve all input/output paths from the wrapper location.",
            "Reuse the deterministic upstream descriptor matrix across three prediction tasks.",
            "Bypass unused metrics imports, CUDA environment settings, temporary tar_seq.pkl and display options.",
            "Use original EA process_data directly; reproduce the other step statements unchanged.",
            "Add hashes, finite-value validation, exact repeated inference check, counts and a selection plot.",
        ],
        "scope_exclusions": ["RoBERTa/SeqGAN training", "upstream evolution", "wet-lab validation"],
    }
    try:
        import matplotlib
        matplotlib.use("Agg")
        run(input_dir, output, report)
    except Exception as exc:
        report["status"] = "failed"
        report["error"] = {"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()}
        write_json(output / "metrics.json", report)
        raise


if __name__ == "__main__":
    main()
