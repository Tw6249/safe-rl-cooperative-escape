import os

for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[variable] = "1"

import argparse
import hashlib
import json
import multiprocessing as mp
import runpy
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
WORKER = {}


def initialize():
    scope = runpy.run_path(str(HERE.parent / "experiments_sota_rl" / "run_sota_baselines.py"))
    from exact_arithmetic import enable
    enable(short_circuit=True)
    WORKER["scope"] = scope


def audit_job(job):
    method, seed_index, scene = job
    source = HERE / f"external_{method}_s{seed_index}"
    output = HERE / f"external_audit_v2_{method}_s{seed_index}"
    suffix = "_egp_final" if method == "egp" else ""
    trajectory = source / f"{scene['name']}{suffix}.npz"
    row = json.loads(trajectory.with_suffix(".json").read_text(encoding="utf-8"))
    segments, failed, distance, connectivity = WORKER["scope"]["audit"](trajectory, scene["lam"])
    with np.load(trajectory) as archive:
        states, times, goal = archive["states"], archive["times"], archive["goal"]
    left, right = np.triu_indices(scene["n"], 1)
    collision_segments = 0
    for index in range(len(times) - 1):
        relative = states[index, left] - states[index, right]
        variation = states[index + 1, left] - states[index + 1, right] - relative
        denominator = np.sum(variation * variation, axis=1)
        fraction = np.divide(-np.sum(relative * variation, axis=1), denominator, out=np.zeros_like(denominator), where=denominator > 0)
        fraction = np.clip(fraction, 0, 1)
        collision_segments += int(np.linalg.norm(relative + fraction[:, None] * variation, axis=1).min() < 1 - 1e-9)
    row.update(audit_segments=segments, audit_failed=len(failed), audit_failed_indices=failed,
               audit_min_distance=distance, audit_min_lambda2=connectivity,
               safe_arrived=row["status"] == "arrived" and not failed, audit_status="completed",
               numeric_collision_segments=collision_segments,
               final_max_err=float(np.linalg.norm(states[-1] - goal, axis=1).max()),
               audit_arithmetic="Exact GMP rational backend, unchanged frozen certificate formulas",
               trajectory_sha256=hashlib.sha256(trajectory.read_bytes()).hexdigest())
    (output / f"{scene['name']}.json").write_text(json.dumps(row, indent=2), encoding="utf-8")
    print(f"{method}/{seed_index} {scene['name']}: {row['status']}; audit={len(failed)}", flush=True)
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=("egp", "dgppo"), required=True)
    parser.add_argument("--seed-index", type=int, required=True)
    arguments = parser.parse_args()
    output = HERE / f"external_audit_v2_{arguments.method}_s{arguments.seed_index}"
    output.mkdir(exist_ok=False)
    scenes = json.loads((HERE / "external_test" / "scenes.json").read_text(encoding="utf-8"))
    with mp.get_context("spawn").Pool(8, initializer=initialize) as pool:
        rows = list(pool.imap_unordered(audit_job, [(arguments.method, arguments.seed_index, scene) for scene in scenes]))
    rows.sort(key=lambda row: row["scene"])
    (output / "summary_audited.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
