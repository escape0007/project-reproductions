#!/usr/bin/env python3
"""Run the local reproduction stages with logs, timeouts and provenance."""
from __future__ import annotations

import argparse
from datetime import datetime
from zoneinfo import ZoneInfo
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "reproduction"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", choices=["all", "analysis", "models", "original"], default="all")
    parser.add_argument("--timeout", type=int, default=1800, help="Hard timeout per stage, seconds")
    args = parser.parse_args()
    logs = BASE / "logs"
    logs.mkdir(exist_ok=True)
    env = os.environ.copy()
    env.update({"MPLCONFIGDIR": str(BASE / ".mplconfig"), "MPLBACKEND": "Agg",
                "OMP_NUM_THREADS": "4", "OPENBLAS_NUM_THREADS": "4",
                "MKL_NUM_THREADS": "4", "PYTHONHASHSEED": "88"})
    omp_dir = BASE / "lib"
    if (omp_dir / "libomp.dylib").exists():
        env["DYLD_LIBRARY_PATH"] = str(omp_dir) + (":" + env["DYLD_LIBRARY_PATH"] if env.get("DYLD_LIBRARY_PATH") else "")
    stages = [("analysis", ".venv"), ("models", ".venv"), ("original", ".venv-original")]
    stages = [(name, venv) for name, venv in stages if args.only in ("all", name)]
    now = datetime.now(ZoneInfo("Asia/Macau"))
    stamp = now.isoformat()
    run_id = now.strftime("%Y%m%dT%H%M%S%f")
    receipt = {"run_id": run_id, "started_at": stamp, "timezone": "Asia/Macau", "platform": platform.platform(),
               "machine": platform.machine(), "stages": [], "timeout_seconds_per_stage": args.timeout,
               "scope": "Published-input replay and archived-data analysis; no de novo pretraining or wet-lab reproduction"}
    receipt["upstream_commit"] = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    receipt["tracked_changes"] = subprocess.check_output(["git", "diff", "--stat"], cwd=ROOT, text=True).strip()
    for name, venv in stages:
        python = ROOT / venv / "bin/python"
        script = BASE / f"reproduce_{name}.py"
        out = BASE / "results" / name
        if out.exists() and any(out.iterdir()):
            history = BASE / "history" / run_id
            history.mkdir(parents=True, exist_ok=True)
            out.rename(history / name)
            previous_log = logs / f"{name}.log"
            if previous_log.exists():
                previous_log.rename(history / f"{name}.log")
        out.mkdir(parents=True, exist_ok=True)
        command = [str(python), "-u", str(script), "--output-dir", str(out)]
        record = {"stage": name, "command": command, "log": str((logs / f"{name}.log").relative_to(ROOT)),
                  "script_sha256": hashlib.sha256(script.read_bytes()).hexdigest()}
        start = time.monotonic()
        print(f"[START] {name}", flush=True)
        with (logs / f"{name}.log").open("w") as log:
            try:
                process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
                record["pid"] = process.pid
                while True:
                    remaining = args.timeout - (time.monotonic() - start)
                    if remaining <= 0:
                        print(f"[TIMEOUT] {name}: terminating after {args.timeout}s", flush=True)
                        process.terminate()
                        try:
                            process.wait(timeout=10)
                        except subprocess.TimeoutExpired:
                            process.kill()
                            process.wait()
                        record["status"] = "timeout"
                        break
                    try:
                        process.wait(timeout=min(30, remaining))
                        record["status"] = "completed" if process.returncode == 0 else "failed"
                        break
                    except subprocess.TimeoutExpired:
                        print(f"[RUNNING] {name}: {time.monotonic() - start:.0f}s; log={record['log']}", flush=True)
                record["exit_code"] = process.returncode
            except OSError as error:
                record.update(status="failed", exit_code=None, error=str(error))
        record["elapsed_seconds"] = round(time.monotonic() - start, 3)
        metrics_file = out / "metrics.json"
        if metrics_file.exists():
            metrics = json.loads(metrics_file.read_text())
            if name == "models":
                record["validation_status"] = metrics.get("validation_status", "incomplete")
            elif name == "original":
                final = metrics.get("final_result", {})
                record["validation_status"] = "paper_sequences_matched" if final.get("paper_sequence_identity_verified") else ("count_matches_identity_unverified" if final.get("readme_sequence_count_matches") else "count_mismatch_or_incomplete")
        record["artifacts"] = [{"path": str(p.relative_to(ROOT)), "bytes": p.stat().st_size,
                                 "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
                                for p in sorted(out.rglob("*")) if p.is_file()]
        receipt["stages"].append(record)
        print(f"[{record['status'].upper()}] {name}: {record['elapsed_seconds']}s", flush=True)
    receipt["finished_at"] = datetime.now(ZoneInfo("Asia/Macau")).isoformat()
    receipt["status"] = "completed" if all(r["status"] == "completed" for r in receipt["stages"]) else "partial"
    (BASE / f"run_receipt_{args.only}.json").write_text(json.dumps(receipt, indent=2, ensure_ascii=False) + "\n")
    subprocess.run([sys.executable, str(BASE / "summarize.py")], cwd=ROOT, check=True)
    if receipt["status"] != "completed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
