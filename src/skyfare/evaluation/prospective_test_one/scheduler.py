#!/usr/bin/env python3
"""Bounded parallel scheduler for 24 immutable Test 1 scoring jobs."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from skyfare.evaluation.prospective_test_one.contract import RUNTIME, registry
from skyfare.evaluation.prospective_test_one.runtime import OUTPUT_ROOT, artifact_complete, job_root, write_json_atomic


@dataclass
class Running:
    job: dict[str, object]
    process: subprocess.Popen[bytes]
    log_handle: object
    started: float
    gpu: int | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--python", default=sys.executable)
    return parser.parse_args()


def _launch(job: dict[str, object], python: str, gpu: int | None) -> Running:
    root = job_root(str(job["job_id"]))
    root.mkdir(parents=True, exist_ok=True)
    handle = (root / "job.log").open("ab", buffering=0)
    env = os.environ.copy()
    env["PYTHONHASHSEED"] = "0"
    env["OMP_NUM_THREADS"] = str(RUNTIME.cpu_threads_per_worker)
    env["MKL_NUM_THREADS"] = str(RUNTIME.cpu_threads_per_worker)
    env["OPENBLAS_NUM_THREADS"] = str(RUNTIME.cpu_threads_per_worker)
    env["NUMEXPR_NUM_THREADS"] = str(RUNTIME.cpu_threads_per_worker)
    env["SKYFARE_V24_CPU_THREADS"] = str(RUNTIME.cpu_threads_per_worker)
    if gpu is None:
        env["CUDA_VISIBLE_DEVICES"] = ""
    else:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        env["SKYFARE_V24_GPU_DEVICE"] = "0"
        env["SKYFARE_V24_XGB_DEVICE"] = "cuda:0"
        env["SKYFARE_V24_TF_MEMORY_LIMIT_MB"] = str(RUNTIME.tensorflow_memory_limit_mib)
    command = [
        python,
        "-m",
        "skyfare.evaluation.prospective_test_one.score_job",
        "--job-json",
        json.dumps(job, separators=(",", ":")),
    ]
    process = subprocess.Popen(command, stdout=handle, stderr=subprocess.STDOUT, env=env)
    return Running(job, process, handle, time.monotonic(), gpu)


def main() -> None:
    cli = parse_args()
    jobs = registry()
    pending = [job for job in jobs if not artifact_complete(str(job["job_id"]))]
    completed = len(jobs) - len(pending)
    active: list[Running] = []
    gpu_counter = 0
    started = time.monotonic()
    print(f"[TEST1 SCHEDULER START] complete={completed}/24 pending={len(pending)}", flush=True)
    while pending or active:
        cpu_active = sum(item.gpu is None for item in active)
        gpu_active = sum(item.gpu is not None for item in active)
        launched = True
        while pending and launched:
            launched = False
            for index, job in enumerate(pending):
                resource = str(job["resource"])
                if resource == "CPU" and cpu_active < RUNTIME.cpu_workers:
                    item = _launch(job, cli.python, None)
                    cpu_active += 1
                elif resource == "GPU" and gpu_active < RUNTIME.gpu_workers:
                    gpu = gpu_counter % RUNTIME.minimum_gpus
                    gpu_counter += 1
                    item = _launch(job, cli.python, gpu)
                    gpu_active += 1
                else:
                    continue
                active.append(item)
                pending.pop(index)
                print(
                    f"[TEST1 RUNNING] {completed + len(active)}/24 job={job['job_id']} "
                    f"lane={'CPU' if item.gpu is None else f'GPU{item.gpu}'}",
                    flush=True,
                )
                launched = True
                break
        now = time.monotonic()
        for item in list(active):
            code = item.process.poll()
            elapsed = now - item.started
            if code is None and elapsed > RUNTIME.job_timeout_minutes * 60:
                item.process.terminate()
                try:
                    item.process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    item.process.kill()
                code = 124
            if code is None:
                continue
            item.log_handle.close()
            active.remove(item)
            if code != 0 or not artifact_complete(str(item.job["job_id"])):
                log = job_root(str(item.job["job_id"])) / "job.log"
                tail = log.read_text(encoding="utf-8", errors="replace").splitlines()[-30:]
                print("\n".join(tail), file=sys.stderr, flush=True)
                raise RuntimeError(f"Test 1 job failed: {item.job['job_id']} code={code}")
            completed += 1
            print(
                f"[TEST1 PASS] {completed}/24 remaining={24 - completed} "
                f"job={item.job['job_id']} elapsed={elapsed / 60:.1f}m",
                flush=True,
            )
        if pending or active:
            time.sleep(2)
    report = {
        "status": "PASS",
        "jobs_complete": completed,
        "jobs_expected": len(jobs),
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "cpu_workers": RUNTIME.cpu_workers,
        "gpu_workers": RUNTIME.gpu_workers,
        "gpu_workers_per_device": RUNTIME.gpu_workers_per_device,
    }
    write_json_atomic(OUTPUT_ROOT / "runtime" / "scheduler_test1.json", report)
    print(f"[TEST1 ALL JOBS PASS] {completed}/24", flush=True)


if __name__ == "__main__":
    main()
