#!/usr/bin/env python
# coding=utf-8

"""
GAIA runner wrapper for K2 auto-evolve.

Fixes missing --judge_model in MemEvolve.utils.run_provider (defaults to gpt-4.1-mini,
which fails on DeepSeek API and yields judgement=error for every task).
"""

import os
import subprocess
import sys
from pathlib import Path
from typing import List, Optional

from MemEvolve.config import DEFAULT_RUNNERS


def run_provider_k2(
    dataset_name: str,
    provider_name: str,
    task_indices: List[int],
    dataset_file: Path,
    output_dir: Path,
    judge_model: Optional[str] = None,
    max_steps: int = 40,
) -> Path:
    """Invoke dataset runner with judge_model and max_steps for fair evaluation."""
    output_dir.mkdir(parents=True, exist_ok=True)
    task_arg = ",".join(str(i + 1) for i in task_indices)
    runner = DEFAULT_RUNNERS.get(dataset_name)
    if runner is None:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    if judge_model is None:
        judge_model = os.getenv(
            "DEFAULT_JUDGE_MODEL",
            os.getenv("DEFAULT_MODEL", "deepseek-v4-flash"),
        )

    runner_path = Path(__file__).resolve().parent.parent.parent / runner
    if not runner_path.exists():
        raise FileNotFoundError(f"Runner script not found: {runner_path}")

    outfile = output_dir / "results.jsonl"
    cmd = [
        "python",
        str(runner_path),
        "--infile",
        str(dataset_file),
        "--outfile",
        str(outfile),
        "--task_indices",
        task_arg,
        "--memory_provider",
        provider_name,
        "--max_steps",
        str(max_steps),
        "--judge_model",
        judge_model,
        "--concurrency",
        "1",
        "--direct_output_dir",
        str(output_dir),
    ]

    print(f"\n[Runner-K2] judge_model={judge_model}, max_steps={max_steps}")
    print(f"[Runner] Executing: {' '.join(cmd[:2])} ...")
    print(f"[Runner] Working directory: {Path.cwd()}")
    print(f"[Runner] Output will be displayed below:\n")
    print("-" * 60)

    process = subprocess.Popen(
        cmd,
        stdout=sys.stdout,
        stderr=sys.stderr,
        text=True,
        bufsize=1,
    )
    process.wait()
    print("-" * 60)

    if process.returncode != 0:
        print(f"\n[Runner] Error occurred (exit code: {process.returncode})")
        raise RuntimeError(
            f"Runner failed ({dataset_name}) with exit code {process.returncode}"
        )

    print(f"\n[Runner] Execution completed successfully")
    json_files = list(output_dir.glob("*.json"))
    if json_files:
        print(f"[Runner] Found {len(json_files)} task logs in output_dir")
    return output_dir
