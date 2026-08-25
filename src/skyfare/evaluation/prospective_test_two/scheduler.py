#!/usr/bin/env python3
"""Restartable bounded-parallel scheduler for 24 immutable Test 2 jobs."""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass

from skyfare.evaluation.prospective_test_two.contract import RUNTIME, registry
from skyfare.evaluation.prospective_test_two.runtime import (
    OUTPUT_ROOT,
    artifact_complete,
    job_root,
    write_json_atomic,
)


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
    env.update(
        {
            "PYTHONHASHSEED": "0",
            "OMP_NUM_THREADS": str(RUNTIME.cpu_threads_per_worker),
            "MKL_NUM_THREADS": str(RUNTIME.cpu_threads_per_worker),
            "OPENBLAS_NUM_THREADS": str(RUNTIME.cpu_threads_per_worker),
            "NUMEXPR_NUM_THREADS": str(RUNTIME.cpu_threads_per_worker),
            "SKYFARE_V24_CPU_THREADS": str(RUNTIME.cpu_threads_per_worker),
        }
    )
    if gpu is None:
        env["CUDA_VISIBLE_DEVICES"] = ""
    else:
        env.update(
            {
                "CUDA_VISIBLE_DEVICES": str(gpu),
                "SKYFARE_V24_GPU_DEVICE": "0",
                "SKYFARE_V24_XGB_DEVICE": "cuda:0",
                "SKYFARE_V24_GPU_HOST_THREADS": str(RUNTIME.gpu_host_threads),
                "SKYFARE_V24_TF_MEMORY_LIMIT_MB": str(RUNTIME.tensorflow_memory_limit_mib),
                "SKYFARE_V24_CAT_GPU_RAM_PART": "0.40",
            }
        )
    command = [
        python,
        "-m",
        "skyfare.evaluation.prospective_test_two.score_job",
        "--job-json",
        json.dumps(job, separators=(",", ":")),
    ]
    process = subprocess.Popen(
        command,
        stdout=handle,
        stderr=subprocess.STDOUT,
        env=env,
        start_new_session=True,
    )
    return Running(job, process, handle, time.monotonic(), gpu)


def _stop(item: Running) -> None:
    if item.process.poll() is not None:
        return
    try:
        os.killpg(item.process.pid, signal.SIGTERM)
        item.process.wait(timeout=15)
    except (ProcessLookupError, subprocess.TimeoutExpired):
        try:
            os.killpg(item.process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


def _heartbeat_age(job_id: str) -> float | None:
    path = job_root(job_id) / "heartbeat.json"
    if not path.is_file():
        return None
    return max(0.0, time.time() - path.stat().st_mtime)


def _choose_gpu(active: list[Running]) -> int | None:
    counts = {
        gpu: sum(item.gpu == gpu for item in active)
        for gpu in range(RUNTIME.minimum_gpus)
    }
    available = [
        gpu for gpu, count in counts.items() if count < RUNTIME.gpu_workers_per_device
    ]
    return min(available, key=lambda gpu: (counts[gpu], gpu)) if available else None


def main() -> None:
    cli = parse_args()
    jobs = registry()
    total = len(jobs)
    pending = [job for job in jobs if not artifact_complete(str(job["job_id"]))]
    completed = total - len(pending)
    active: list[Running] = []
    started = time.monotonic()
    print(
        f"[TEST2 SCHEDULER START] complete={completed}/{total} pending={len(pending)} "
        f"cpu_workers={RUNTIME.cpu_workers} gpu_workers={RUNTIME.gpu_workers}",
        flush=True,
    )
    try:
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
                        gpu = _choose_gpu(active)
                        if gpu is None:
                            continue
                        item = _launch(job, cli.python, gpu)
                        gpu_active += 1
                    else:
                        continue
                    active.append(item)
                    pending.pop(index)
                    lane = "CPU" if item.gpu is None else f"GPU{item.gpu}"
                    print(
                        f"[TEST2 RUNNING] complete={completed}/{total} active={len(active)} "
                        f"pending={len(pending)} lane={lane} job={job['job_id']}",
                        flush=True,
                    )
                    launched = True
                    break

            now = time.monotonic()
            for item in list(active):
                identifier = str(item.job["job_id"])
                code = item.process.poll()
                elapsed = now - item.started
                heartbeat_age = _heartbeat_age(identifier)
                timed_out = elapsed > RUNTIME.job_timeout_minutes * 60
                stale = (
                    heartbeat_age is not None
                    and heartbeat_age > RUNTIME.heartbeat_stale_minutes * 60
                )
                if code is None and (timed_out or stale):
                    reason = "hard-timeout" if timed_out else "stale-heartbeat"
                    print(
                        f"[TEST2 WATCHDOG STOP] job={identifier} reason={reason} "
                        f"elapsed={elapsed / 60:.1f}m heartbeat_age="
                        f"{'none' if heartbeat_age is None else f'{heartbeat_age / 60:.1f}m'}",
                        file=sys.stderr,
                        flush=True,
                    )
                    _stop(item)
                    code = 124
                if code is None:
                    continue
                item.log_handle.close()
                active.remove(item)
                if code != 0 or not artifact_complete(identifier):
                    log = job_root(identifier) / "job.log"
                    tail = log.read_text(encoding="utf-8", errors="replace").splitlines()[-60:]
                    print("\n".join(tail), file=sys.stderr, flush=True)
                    raise RuntimeError(f"Test 2 job failed: {identifier} code={code}")
                completed += 1
                print(
                    f"[TEST2 PASS] {completed}/{total} remaining={total - completed} "
                    f"job={identifier} elapsed={elapsed / 60:.1f}m",
                    flush=True,
                )
            if pending or active:
                time.sleep(2)
    except BaseException:
        for item in active:
            _stop(item)
            item.log_handle.close()
        raise

    report = {
        "status": "PASS",
        "jobs_complete": completed,
        "jobs_expected": total,
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "cpu_workers": RUNTIME.cpu_workers,
        "gpu_workers": RUNTIME.gpu_workers,
        "gpu_workers_per_device": RUNTIME.gpu_workers_per_device,
        "watchdog": {
            "heartbeat_stale_minutes": RUNTIME.heartbeat_stale_minutes,
            "job_timeout_minutes": RUNTIME.job_timeout_minutes,
            "process_group_termination": True,
        },
    }
    write_json_atomic(OUTPUT_ROOT / "runtime" / "scheduler_test2.json", report)
    print(f"[TEST2 ALL JOBS PASS] {completed}/{total}", flush=True)


if __name__ == "__main__":
    main()
