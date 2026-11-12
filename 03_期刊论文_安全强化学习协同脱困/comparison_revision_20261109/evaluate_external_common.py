import os

for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[variable] = "1"
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"

import argparse
import hashlib
import json
import multiprocessing as mp
import runpy
import sys
from functools import partial
from pathlib import Path

import numpy as np


sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
WORKER = {}


def initialize(checkpoint):
    scope = runpy.run_path(str(HERE.parent / "experiments_sota_rl" / "run_sota_baselines.py"))
    from exact_arithmetic import enable
    enable()
    scope["torch"].set_num_threads(1)
    WORKER["scope"] = scope
    if checkpoint:
        state = scope["torch"].load(checkpoint, map_location="cpu", weights_only=False)
        actor = scope["P"].Actor(state["variant"])
        actor.load_state_dict(state["actor"])
        actor.eval()
        WORKER["actor"] = actor


def run_job(job):
    scene, output, audit_only = job
    scope = WORKER["scope"]
    scene = dict(scene, p=np.asarray(scene["p"], dtype=float), goal=np.asarray(scene["goal"], dtype=float))
    output = Path(output)
    if audit_only:
        trajectory = output / f"{scene['name']}.npz"
        row_path = output / f"{scene['name']}.json"
        row = json.loads(row_path.read_text(encoding="utf-8"))
    else:
        environment, bridge, evaluator, enveloped = scope["E"], scope["pa_bridge"], scope["EV"], scope["EE"]
        bridge.PAOracle = partial(evaluator.budget_oracle(environment), settle="local", diagnostic=False)
        enveloped.EnvelopedResidualFlow.gate_kind = "mu"
        enveloped.EnvelopedResidualFlow.eps_g = 0.05
        enveloped.EnvelopedResidualFlow.policy = scope["P"].NumpyPolicy(WORKER["actor"], True)
        bridge.BraidFlow = enveloped.EnvelopedResidualFlow
        row = bridge.run_pa(scene, output, 0.0, 8, environment.RADIUS, 60.0, "egp_final", clip_level=True)
        if row["status"] == "solver_failure" and "StepBudget" in str(row.get("failure")):
            row["status"] = "step_budget"
        trajectory = output / f"{scene['name']}_egp_final.npz"
        row_path = output / f"{scene['name']}_audited.json"
    segments, failed, distance, connectivity = scope["audit"](trajectory, scene["lam"])
    with np.load(trajectory) as archive:
        states, times, goals = archive["states"], archive["times"], archive["goal"]
    left, right = np.triu_indices(scene["n"], 1)
    collision_segments = 0
    numeric_min_distance = float("inf")
    for index in range(len(times) - 1):
        relative = states[index, left] - states[index, right]
        variation = states[index + 1, left] - states[index + 1, right] - relative
        denominator = np.sum(variation * variation, axis=1)
        fraction = np.divide(-np.sum(relative * variation, axis=1), denominator, out=np.zeros_like(denominator), where=denominator > 0)
        fraction = np.clip(fraction, 0, 1)
        closest = np.linalg.norm(relative + fraction[:, None] * variation, axis=1).min()
        numeric_min_distance = min(numeric_min_distance, float(closest))
        collision_segments += int(closest < 1 - 1e-9)
    observed_speeds = np.linalg.norm(np.diff(states, axis=0), axis=2) / np.diff(times)[:, None] if len(times) > 1 else np.zeros((0, scene["n"]))
    row.update(audit_segments=segments, audit_failed=len(failed), audit_failed_indices=failed,
               audit_min_distance=distance, audit_min_lambda2=connectivity,
               safe_arrived=row["status"] == "arrived" and not failed, audit_status="completed",
               audit_arithmetic="Exact rational arithmetic with GMP backend; unchanged certificate formulas",
               numeric_collision_segments=collision_segments, numeric_min_distance=numeric_min_distance,
               numeric_speed_violation_segments=int(np.count_nonzero(observed_speeds.max(axis=1) > 0.5 + 1e-9)) if len(observed_speeds) else 0,
               final_max_err=float(np.linalg.norm(states[-1] - goals, axis=1).max()),
               trajectory_sha256=hashlib.sha256(trajectory.read_bytes()).hexdigest())
    row_path.write_text(json.dumps(row, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"{scene['name']}: {row['status']}; audited={len(failed)}", flush=True)
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint")
    parser.add_argument("--output", required=True)
    parser.add_argument("--audit-only", action="store_true")
    arguments = parser.parse_args()
    output = Path(arguments.output).resolve()
    if not arguments.audit_only:
        output.mkdir(exist_ok=False)
        checkpoint = Path(arguments.checkpoint).resolve()
        (output / "evaluation_protocol.json").write_text(json.dumps(dict(checkpoint=str(checkpoint),
            checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            selection="Pre-specified final training checkpoint; no new test selection",
            physical_horizon=60.0, hold_seconds=1.0, controller_call_budget=3600,
            controller="Original EGP-NPA envelope, dynamic allocation B=8 and adaptive segment certification"), indent=2), encoding="utf-8")
    scenes = json.loads((HERE / "external_test" / "scenes.json").read_text(encoding="utf-8"))
    with mp.get_context("spawn").Pool(8, initializer=initialize, initargs=(arguments.checkpoint,)) as pool:
        rows = list(pool.imap_unordered(run_job, [(scene, str(output), arguments.audit_only) for scene in scenes]))
    rows.sort(key=lambda row: row["scene"])
    (output / "summary_audited.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
