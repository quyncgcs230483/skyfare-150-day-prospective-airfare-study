#!/usr/bin/env python3
"""Concurrent CPU/GPU job scheduler with timeout, heartbeat and atomic resume."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

from skyfare.models.selection_contract import RUNTIME
from skyfare.models.temporal_runtime import (
    OUTPUT_ROOT,
    artifact_complete,
    write_json_atomic,
)

MODULE_DIR = Path(__file__).resolve().parent
RUN_JOB_MODULE = "skyfare.models.train_candidate"
PRINT_LOCK = threading.Lock()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--gpu-slots", type=int, choices=(2, 4), required=True)
    parser.add_argument("--deadline-epoch", type=float, required=True)
    parser.add_argument("--require-complete", action="store_true")
    return parser.parse_args()


def _job_environment(job: dict[str, object], slot: int, gpu_slots: int) -> dict[str, str]:
    env = os.environ.copy()
    if job["resource"] == "GPU":
        physical = slot % 2
        env.update(
            CUDA_VISIBLE_DEVICES=str(physical),
            SKYFARE_V24_GPU_DEVICE="0",
            SKYFARE_V24_XGB_DEVICE="cuda:0",
            SKYFARE_V24_GPU_HOST_THREADS="4",
            SKYFARE_V24_TF_MEMORY_LIMIT_MB="10500" if gpu_slots == 4 else "0",
            SKYFARE_V24_CAT_GPU_RAM_PART="0.42" if gpu_slots == 4 else "0.80",
            OMP_NUM_THREADS="4",
            OPENBLAS_NUM_THREADS="4",
            MKL_NUM_THREADS="4",
            NUMEXPR_NUM_THREADS="4",
        )
    else:
        threads = str(RUNTIME.cpu_threads_per_worker)
        env.update(
            CUDA_VISIBLE_DEVICES="",
            SKYFARE_V24_CPU_THREADS=threads,
            OMP_NUM_THREADS=threads,
            OPENBLAS_NUM_THREADS=threads,
            MKL_NUM_THREADS=threads,
            NUMEXPR_NUM_THREADS=threads,
        )
    return env


def run_one(job: dict[str, object], slot: int, gpu_slots: int, python: str, deadline: float) -> tuple[str, float]:
    identifier = str(job["job_id"])
    if artifact_complete(identifier):
        return identifier, 0.0
    remaining = deadline - time.time()
    timeout = min(RUNTIME.job_timeout_minutes * 60.0, remaining)
    if timeout <= 0:
        raise TimeoutError("V24 scheduler deadline reached")
    command = [
        python,
        "-m",
        RUN_JOB_MODULE,
        "--job-json",
        json.dumps(job, separators=(",", ":")),
    ]
    started = time.monotonic()
    process = subprocess.Popen(
        command,
        env=_job_environment(job, slot, gpu_slots),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        start_new_session=True,
    )
    output: list[str] = []
    try:
        stdout, _ = process.communicate(timeout=timeout)
        output = stdout.splitlines()
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            stdout, _ = process.communicate(timeout=15)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            stdout, _ = process.communicate()
        output = stdout.splitlines()
        raise TimeoutError(
            f"job timeout {identifier} after {timeout / 60.0:.1f}m"
        ) from None
    finally:
        with PRINT_LOCK:
            for line in output[-40:]:
                print(line, flush=True)
    if process.returncode != 0 or not artifact_complete(identifier):
        raise RuntimeError(f"job failed or artifact incomplete: {identifier} code={process.returncode}")
    return identifier, time.monotonic() - started


def execute_lane(
    lane: str,
    jobs: list[dict[str, object]],
    workers: int,
    python: str,
    deadline: float,
    progress: dict[str, int],
    total: int,
) -> None:
    if not jobs:
        return
    print(f"[V24 LANE START] lane={lane} workers={workers} jobs={len(jobs)}", flush=True)
    slot_locks = [threading.Lock() for _ in range(workers)]

    def run_on_slot(job: dict[str, object], slot: int) -> tuple[str, float]:
        with slot_locks[slot]:
            return run_one(job, slot, workers, python, deadline)

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers, thread_name_prefix=f"v24-{lane}") as executor:
        pending = {
            executor.submit(run_on_slot, job, index % workers): job
            for index, job in enumerate(jobs)
        }
        for future in concurrent.futures.as_completed(pending):
            pending[future]
            try:
                identifier, elapsed = future.result()
            except Exception:
                for peer in pending:
                    peer.cancel()
                raise
            with PRINT_LOCK:
                progress["complete"] += 1
                print(
                    f"[V24 PROGRESS] {progress['complete']}/{total} remaining={total - progress['complete']} "
                    f"lane={lane} job={identifier} elapsed={elapsed:.1f}s",
                    flush=True,
                )


def main() -> None:
    cli = parse_args()
    payload = json.loads(cli.registry.read_text(encoding="utf-8"))
    jobs = list(payload["jobs"])
    if int(payload["job_count"]) != len(jobs):
        raise RuntimeError("registry count mismatch")
    outstanding = [job for job in jobs if not artifact_complete(str(job["job_id"]))]
    cpu = [job for job in outstanding if job["resource"] == "CPU"]
    gpu = [job for job in outstanding if job["resource"] == "GPU"]
    progress = {"complete": len(jobs) - len(outstanding)}
    print(
        f"[V24 SCHEDULER START] phase={payload['phase']} total={len(jobs)} resumed={progress['complete']} "
        f"cpu={len(cpu)} gpu={len(gpu)} cpu_workers={RUNTIME.cpu_workers} gpu_slots={cli.gpu_slots}",
        flush=True,
    )
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as lanes:
        futures = [
            lanes.submit(execute_lane, "CPU", cpu, RUNTIME.cpu_workers, cli.python, cli.deadline_epoch, progress, len(jobs)),
            lanes.submit(execute_lane, "GPU", gpu, cli.gpu_slots, cli.python, cli.deadline_epoch, progress, len(jobs)),
        ]
        for future in futures:
            future.result()
    complete = [job for job in jobs if artifact_complete(str(job["job_id"]))]
    if cli.require_complete and len(complete) != len(jobs):
        raise RuntimeError(f"registry incomplete {len(complete)}/{len(jobs)}")
    status = {
        "status": "PASS" if len(complete) == len(jobs) else "PARTIAL",
        "phase": payload["phase"],
        "complete": len(complete),
        "total": len(jobs),
        "gpu_slots": cli.gpu_slots,
        "cpu_workers": RUNTIME.cpu_workers,
    }
    write_json_atomic(OUTPUT_ROOT / "runtime" / f"scheduler_{str(payload['phase']).lower()}.json", status)
    print(f"[V24 SCHEDULER PASS] phase={payload['phase']} coverage={len(complete)}/{len(jobs)}", flush=True)


if __name__ == "__main__":
    main()
