#!/usr/bin/env python3
"""Run ojp-core N times against a model, report mean/stdev of peak RSS
and wall time. Uses resource.getrusage (RUSAGE_CHILDREN), no external
dependency needed (GNU time isn't installed on this machine).
"""
import argparse
import resource
import statistics
import subprocess
import sys
import time


def run_once(binary: str, model_path: str, gpu_layers: int, env_extra: dict):
    env = {**__import__("os").environ, "OJP_GPU_LAYERS": str(gpu_layers), **env_extra}
    start = time.monotonic()
    proc = subprocess.run([binary, model_path], capture_output=True, text=True, env=env)
    elapsed = time.monotonic() - start
    if proc.returncode != 0:
        print(proc.stderr[-2000:], file=sys.stderr)
        raise RuntimeError(f"run failed, exit code {proc.returncode}")
    usage = resource.getrusage(resource.RUSAGE_CHILDREN)
    return elapsed, usage.ru_maxrss


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("binary")
    parser.add_argument("model_path")
    parser.add_argument("--gpu-layers", type=int, default=0)
    parser.add_argument("--reps", type=int, default=5)
    parser.add_argument("--vk-icd", default="")
    args = parser.parse_args()

    env_extra = {}
    if args.vk_icd:
        env_extra["VK_ICD_FILENAMES"] = args.vk_icd

    times, rss = [], []
    for i in range(args.reps):
        t, r = run_once(args.binary, args.model_path, args.gpu_layers, env_extra)
        times.append(t)
        rss.append(r)
        print(f"  rep {i + 1}/{args.reps}: elapsed={t:.3f}s peak_rss={r}KB")

    print(
        f"RESULT model={args.model_path} gpu_layers={args.gpu_layers} "
        f"n={args.reps} "
        f"time_mean={statistics.mean(times):.3f}s time_stdev={statistics.stdev(times):.3f}s "
        f"rss_mean={statistics.mean(rss):.0f}KB rss_stdev={statistics.stdev(rss):.0f}KB"
    )


if __name__ == "__main__":
    main()
