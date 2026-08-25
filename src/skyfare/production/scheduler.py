#!/usr/bin/env python3
"""Resource-aware restartable scheduler for 2x RTX 4090 and 64 CPU host."""

from __future__ import annotations

import concurrent.futures
import json
import os
import queue
import subprocess
import sys
import threading
import time

from skyfare.production.contract import RUNTIME, registry
from skyfare.production.runtime import OUTPUT_ROOT, artifact_complete, write_json_atomic

LOG_ROOT = OUTPUT_ROOT / "logs"
STATE_PATH = OUTPUT_ROOT / "runtime/scheduler_state.json"
LOCK = threading.Lock()


def _state() -> dict[str, object]:
    jobs = registry()
    complete = sum(artifact_complete(str(job["job_id"])) for job in jobs)
    return {"status": "RUNNING" if complete < len(jobs) else "COMPLETE", "complete": complete, "total": len(jobs)}


def _run(job: dict[str, object], gpu_slots: queue.Queue[int], cpu_slots: threading.Semaphore) -> str:
    job_id = str(job["job_id"])
    if artifact_complete(job_id):
        return job_id
    resource = str(job["resource"])
    gpu: int | None = None
    if resource == "GPU":
        gpu = gpu_slots.get()
    else:
        cpu_slots.acquire()
    try:
        env = os.environ.copy()
        env.update(
            {
                "PYTHONHASHSEED": "0",
                "OMP_NUM_THREADS": str(RUNTIME.gpu_host_threads if resource == "GPU" else RUNTIME.cpu_threads_per_worker),
                "MKL_NUM_THREADS": str(RUNTIME.gpu_host_threads if resource == "GPU" else RUNTIME.cpu_threads_per_worker),
                "OPENBLAS_NUM_THREADS": str(RUNTIME.gpu_host_threads if resource == "GPU" else RUNTIME.cpu_threads_per_worker),
                "SKYFARE_V24_CPU_THREADS": str(RUNTIME.cpu_threads_per_worker),
                "SKYFARE_V24_GPU_HOST_THREADS": str(RUNTIME.gpu_host_threads),
                "SKYFARE_V24_TF_MEMORY_LIMIT_MB": str(RUNTIME.tensorflow_memory_limit_mib),
            }
        )
        if gpu is None:
            env["CUDA_VISIBLE_DEVICES"] = ""
        else:
            env["CUDA_VISIBLE_DEVICES"] = str(gpu)
            env["SKYFARE_V24_XGB_DEVICE"] = "cuda:0"
        LOG_ROOT.mkdir(parents=True, exist_ok=True)
        command = [
            sys.executable,
            "-m",
            "skyfare.production.train_job",
            "--job-json",
            json.dumps(job, sort_keys=True),
        ]
        with (LOG_ROOT / f"{job_id}.log").open("a", encoding="utf-8") as handle:
            handle.write(f"[SCHEDULER CLAIM] resource={resource} gpu={gpu}\n")
            handle.flush()
            result = subprocess.run(
                command,
                env=env,
                stdout=handle,
                stderr=subprocess.STDOUT,
                timeout=RUNTIME.job_timeout_minutes * 60,
                check=False,
            )
        if result.returncode != 0 or not artifact_complete(job_id):
            raise RuntimeError(f"production job failed: {job_id} code={result.returncode}")
        with LOCK:
            write_json_atomic(STATE_PATH, _state())
            state = _state()
            print(f"[PRODUCTION PROGRESS] {state['complete']}/{state['total']} job={job_id}", flush=True)
        return job_id
    finally:
        if gpu is None:
            cpu_slots.release()
        else:
            gpu_slots.put(gpu)


def main() -> None:
    started = time.monotonic()
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    jobs = [job for job in registry() if not artifact_complete(str(job["job_id"]))]
    gpu_slots: queue.Queue[int] = queue.Queue()
    for device in range(2):
        for _ in range(RUNTIME.gpu_workers_per_device):
            gpu_slots.put(device)
    cpu_slots = threading.Semaphore(RUNTIME.cpu_workers)
    write_json_atomic(STATE_PATH, _state())
    print(
        f"[PRODUCTION SCHEDULER START] pending={len(jobs)} gpu_slots={RUNTIME.gpu_workers} "
        f"cpu_workers={RUNTIME.cpu_workers}",
        flush=True,
    )
    failures: list[str] = []
    gpu_jobs = [job for job in jobs if job["resource"] == "GPU"]
    cpu_jobs = [job for job in jobs if job["resource"] == "CPU"]
    with (
        concurrent.futures.ThreadPoolExecutor(max_workers=RUNTIME.gpu_workers) as gpu_executor,
        concurrent.futures.ThreadPoolExecutor(max_workers=RUNTIME.cpu_workers) as cpu_executor,
    ):
        futures = {
            **{gpu_executor.submit(_run, job, gpu_slots, cpu_slots): str(job["job_id"]) for job in gpu_jobs},
            **{cpu_executor.submit(_run, job, gpu_slots, cpu_slots): str(job["job_id"]) for job in cpu_jobs},
        }
        for future in concurrent.futures.as_completed(futures):
            try:
                future.result()
            except Exception as error:
                failures.append(f"{futures[future]}: {error}")
    if failures:
        write_json_atomic(STATE_PATH, {**_state(), "status": "FAILED", "failures": failures})
        raise RuntimeError("; ".join(failures))
    subprocess.run([sys.executable, "-m", "skyfare.production.finalize"], check=True)
    subprocess.run([sys.executable, "-m", "skyfare.production.verify"], check=True)
    subprocess.run([sys.executable, "-m", "skyfare.production.archive"], check=True)
    write_json_atomic(
        STATE_PATH,
        {**_state(), "elapsed_seconds": round(time.monotonic() - started, 3), "result_archive_ready": True},
    )
    print("[PRODUCTION PIPELINE COMPLETE] verified, serialized, archived", flush=True)


if __name__ == "__main__":
    main()
