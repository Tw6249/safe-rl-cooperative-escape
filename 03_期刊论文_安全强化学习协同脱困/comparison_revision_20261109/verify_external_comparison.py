import hashlib
import json
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def main():
    scenes = {row["name"]: row for row in json.loads((HERE / "external_test" / "scenes.json").read_text(encoding="utf-8"))}
    checks = []
    for method in ("egp", "dgppo"):
        for seed_index in (1, 2):
            output = HERE / f"external_{method}_s{seed_index}"
            candidates = [HERE / f"external_audit_v2_{method}_s{seed_index}", HERE / f"external_audit_{method}_s{seed_index}", output]
            audited = next(folder for folder in candidates if (folder / "summary_audited.json").exists())
            rows = json.loads((audited / "summary_audited.json").read_text(encoding="utf-8"))
            if len(rows) != 96 or len({row["scene"] for row in rows}) != 96:
                raise ValueError(f"Incomplete or duplicate rows: {output}")
            for row in rows:
                scene = scenes[row["scene"]]
                suffix = "_egp_final" if method == "egp" else ""
                trajectory = output / f"{row['scene']}{suffix}.npz"
                with np.load(trajectory) as archive:
                    states, times, goal = archive["states"], archive["times"], archive["goal"]
                np.testing.assert_array_equal(states[0], np.array(scene["p"]))
                np.testing.assert_array_equal(goal, np.array(scene["goal"]))
                if len(times) - 1 != row["audit_segments"]:
                    raise ValueError("Audit does not cover every executed segment")
                if row["status"] == "arrived":
                    error = np.linalg.norm(states - goal, axis=2).max(axis=1)
                    start = len(times) - 1
                    while start > 0 and error[start - 1] <= .01:
                        start -= 1
                    if error[-1] > .01 or times[-1] - times[start] < 1 - 1e-9:
                        raise ValueError("Arrival holding interval is incomplete")
                checks.append(dict(method=method, seed_index=seed_index, scene=row["scene"],
                                   sha256=hashlib.sha256(trajectory.read_bytes()).hexdigest()))
    for seed_index in (1, 2):
        training = HERE / f"official_dgppo_h500_margin_s{seed_index}"
        protocol = json.loads((training / "protocol.json").read_text(encoding="utf-8"))
        done = json.loads((training / "done.json").read_text(encoding="utf-8"))
        if done["smoke"] or done["environment_transitions"] != 480000 or protocol["horizon_steps"] != 500:
            raise ValueError("Wrong external training budget or horizon")
        logs = [json.loads(line) for line in (training / "training.jsonl").read_text(encoding="utf-8").splitlines()]
        if len(logs) != 120 or logs[-1]["environment_transitions"] != 480000:
            raise ValueError("Training trajectory accounting failed")
        hashes = json.loads((training / "source_sha256.json").read_text(encoding="utf-8"))
        checked_sources = 0
        for name, digest in hashes.items():
            if Path(name).parts[0] == "vendor":
                if hashlib.sha256((HERE / name).read_bytes()).hexdigest() != digest:
                    raise ValueError("Upstream algorithm changed after training")
                checked_sources += 1
        if checked_sources < 20:
            raise ValueError("Insufficient upstream source coverage")
    (HERE / "external_verification.json").write_text(json.dumps(dict(
        trajectories_checked=len(checks), shared_scene_inputs_verified=True, elapsed_goal_holding_verified=True,
        audit_segment_coverage_verified=True, dgppo_original_update_sources_verified=True,
        training_replicates=2, training_transitions_per_dgppo_seed=480000, trajectories=checks), indent=2), encoding="utf-8")
    print("Verified 384 trajectories, matched scene inputs, holding intervals, audit coverage and upstream sources")


if __name__ == "__main__":
    main()
