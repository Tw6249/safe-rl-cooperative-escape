import os

for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[variable] = "1"
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"

import hashlib
import json
import multiprocessing as mp
import runpy
import sys
from pathlib import Path

import numpy as np

from evaluation import evaluate


sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
OUT = HERE / "results_exploratory_corrected"
WORKER = {}


def initialize():
    scope = runpy.run_path(str(HERE.parent / "experiments_sota_rl" / "run_sota_baselines.py"))
    scope["torch"].set_num_threads(1)
    checkpoint = scope["torch"].load(scope["LAG_CKPT"], map_location="cpu", weights_only=False)
    actor = scope["P"].Actor(checkpoint["variant"])
    actor.load_state_dict(checkpoint["actor"])
    actor.eval()
    WORKER.update(scope=scope, actor=actor, scenes=scope["all_96_scenes"]())


def run_case(name):
    scope = WORKER["scope"]
    scene = WORKER["scenes"][name]
    policy = scope["P"].NumpyPolicy(WORKER["actor"], True)
    previous_velocity = np.zeros_like(scene["p"])
    memory = np.zeros(scene["n"])
    environment = scope["E"]

    def controller(position):
        nonlocal previous_velocity, memory
        observation, geometry, memory = scope["observe_geom_fast"](position, scene["goal"], scene["lam"], previous_velocity, memory)
        node_action, edge_action = policy(observation)
        error = geometry["e"]
        source, target = geometry["I"], geometry["J"]
        rotation = np.clip(node_action[:, 0], -1.0, 1.0)
        radial = np.clip(node_action[:, 1], -1.0, 1.0)
        edge_weight = np.clip(edge_action, -1.0, 1.0)
        perpendicular = np.stack([-error[:, 1], error[:, 0]], axis=1)
        repulsion = np.zeros_like(position)
        if len(source):
            away = (position[source] - position[target]) / geometry["d"][:, None]
            np.add.at(repulsion, source, edge_weight[:, None] * away)
        repulsion /= np.maximum(geometry["deg"], 1)[:, None]
        residual = environment.A_ROT * rotation[:, None] * perpendicular + environment.A_RAD * radial[:, None] * error + environment.A_PSI * environment.U * repulsion
        previous_velocity = environment.clip_ball(-error + observation["gate"][:, None] * residual)
        return previous_velocity

    trajectory = OUT / f"{name}.npz"
    row = evaluate(scene, controller, trajectory)
    segments, failed, distance, connectivity = scope["audit"](trajectory, scene["lam"])
    row.update(bench=scene["bench"], n=scene["n"], lambda_required=scene["lam"],
               audit_segments=segments, audit_failed=len(failed), audit_failed_indices=failed,
               audit_min_distance=distance, audit_min_lambda2=connectivity,
               safe_arrived=row["status"] == "arrived" and len(failed) == 0)
    (OUT / f"{name}.json").write_text(json.dumps(row, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"{name}: {row['status']}, held={row['holding_duration']}, audit_failed={len(failed)}", flush=True)
    return row


def main():
    OUT.mkdir(exist_ok=False)
    initialize()
    checkpoint = WORKER["scope"]["LAG_CKPT"]
    protocol = dict(identity="Historical exploratory Lagrangian PPO; not a published-method reproduction",
                    checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                    changes=["Elapsed 1.0 s holding criterion", "No workspace position clipping"],
                    training="unchanged; does not satisfy its training cost limit")
    (OUT / "protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    names = sorted(WORKER["scenes"])
    with mp.get_context("spawn").Pool(8, initializer=initialize) as pool:
        rows = list(pool.imap_unordered(run_case, names))
    rows.sort(key=lambda row: row["scene"])
    (OUT / "summary.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(dict(runs=len(rows), safe_arrivals=sum(row["safe_arrived"] for row in rows),
                         audit_failed=sum(row["audit_failed"] for row in rows)), indent=2))


if __name__ == "__main__":
    main()
