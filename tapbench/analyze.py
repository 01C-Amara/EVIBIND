from __future__ import annotations

import csv
import json
import os
import shutil
import subprocess
from pathlib import Path

from .config import REPO_ROOT
from .io import read_jsonl


def export_scores_csv(scores_path: str | Path, output_dir: str | Path) -> Path:
    rows = read_jsonl(scores_path)
    target_dir = Path(output_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / "scores.csv"
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with target.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return target


def run_lme4(scores_path: str | Path, output_dir: str | Path) -> Path:
    csv_path = export_scores_csv(scores_path, output_dir)
    target_dir = Path(output_dir)
    script = REPO_ROOT / "analysis" / "glmm_fit.R"
    local_rscript = REPO_ROOT / "work" / "conda_r" / "bin" / "Rscript"
    rscript = os.environ.get("TAPBENCH_RSCRIPT")
    if not rscript and local_rscript.exists():
        rscript = str(local_rscript)
    if not rscript:
        rscript = shutil.which("Rscript")
    if rscript is None:
        status = {
            "engine": "lme4",
            "engine_available": False,
            "error": "Rscript executable was not found",
        }
        (target_dir / "glmm_fit_status.json").write_text(json.dumps([status], indent=2, sort_keys=True) + "\n", encoding="utf-8")
        with (target_dir / "glmm_fit_status.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(status.keys()))
            writer.writeheader()
            writer.writerow(status)
        raise RuntimeError("Rscript executable was not found; install R and lme4 to run primary GLMM fitting")
    try:
        subprocess.run([rscript, str(script), str(csv_path), str(output_dir)], check=True)
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"lme4 fitting failed with exit status {exc.returncode}") from exc
    return target_dir / "glmm_coefficients.csv"
