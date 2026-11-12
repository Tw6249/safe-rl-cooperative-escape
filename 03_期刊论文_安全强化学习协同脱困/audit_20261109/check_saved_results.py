import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
PAPER = HERE.parent
RESULTS = PAPER / "experiments_sota_rl" / "results"


def main():
    rows = [json.loads(path.read_text(encoding="utf-8")) for path in sorted((RESULTS / "eval_rows").glob("*.json"))]
    summary = json.loads((RESULTS / "summary_sota_rl.json").read_text(encoding="utf-8"))
    summary_map = {(row["scene"], row["tag"]): row for row in summary}
    groups = defaultdict(list)
    scene_inputs = {}
    input_mismatches = []
    numeric_checks = []
    for row in rows:
        path = RESULTS / "eval_runs" / f"{row['scene']}_{row['tag']}.npz"
        with np.load(path) as archive:
            states, times, goal = archive["states"], archive["times"], archive["goal"]
        inputs = (states[0], goal)
        if row["scene"] in scene_inputs:
            if not all(np.array_equal(left, right) for left, right in zip(inputs, scene_inputs[row["scene"]])):
                input_mismatches.append([row["scene"], row["tag"]])
        else:
            scene_inputs[row["scene"]] = inputs
        pair_left, pair_right = np.triu_indices(states.shape[1], 1)
        collision_segments = 0
        minimum_distance = float("inf")
        for start in range(0, len(times) - 1, 256):
            previous = states[start:min(start + 256, len(times) - 1)]
            following = states[start + 1:min(start + 257, len(times))]
            relative = previous[:, pair_left] - previous[:, pair_right]
            change = following[:, pair_left] - following[:, pair_right] - relative
            denominator = np.sum(change * change, axis=2)
            numerator = -np.sum(relative * change, axis=2)
            fraction = np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator > 0)
            fraction = np.clip(fraction, 0, 1)
            distance = np.linalg.norm(relative + fraction[:, :, None] * change, axis=2)
            minimum_distance = min(minimum_distance, float(distance.min()))
            collision_segments += int(np.count_nonzero(distance.min(axis=1) < 1 - 1e-9))
        speeds = np.linalg.norm(np.diff(states, axis=0), axis=2) / np.diff(times)[:, None]
        errors = np.linalg.norm(states - goal, axis=2).max(axis=1)
        tail_start = len(times) - 1
        while tail_start > 0 and errors[tail_start - 1] <= 0.01:
            tail_start -= 1
        holding_duration = float(times[-1] - times[tail_start]) if errors[-1] <= 0.01 else 0.0
        check = dict(
            scene=row["scene"], tag=row["tag"],
            sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            saved_steps_match=len(times) - 1 == row["steps"],
            min_distance=minimum_distance,
            saved_min_distance_difference=minimum_distance - row["audit_min_distance"],
            collision_segments_float_tolerance_1e_9=collision_segments,
            max_speed=float(speeds.max()),
            speed_segments_above_0_5_plus_1e_9=int(np.count_nonzero(speeds.max(axis=1) > 0.5 + 1e-9)),
            max_abs_position=float(np.abs(states).max()),
            final_error=float(errors[-1]),
            saved_final_error_difference=float(errors[-1]) - row["final_max_err"],
            final_holding_duration=holding_duration,
            duration=float(times[-1]),
            status=row["status"],
            audit_failed_saved=row["audit_failed"],
        )
        numeric_checks.append(check)
        groups[(row["tag"], row["bench"])].append((row, check))
    grouped = []
    for (tag, bench), pairs in sorted(groups.items()):
        arrivals = [row["arrival"] for row, check in pairs if row["status"] == "arrived"]
        grouped.append(dict(
            tag=tag, bench=bench, runs=len(pairs),
            statuses=dict(Counter(row["status"] for row, check in pairs)),
            arrivals=len(arrivals), median_arrival=float(np.median(arrivals)) if arrivals else None,
            saved_safe_arrivals=sum(row["status"] == "arrived" and row["audit_failed"] == 0 for row, check in pairs),
            saved_audit_failed_segments=sum(row["audit_failed"] for row, check in pairs),
            numeric_collision_segments=sum(check["collision_segments_float_tolerance_1e_9"] for row, check in pairs),
            numeric_collision_runs=sum(check["collision_segments_float_tolerance_1e_9"] > 0 for row, check in pairs),
            speed_violation_segments=sum(check["speed_segments_above_0_5_plus_1e_9"] for row, check in pairs),
            step_cuts=sum(row["step_cuts"] for row, check in pairs),
            arrived_hold_below_1s=sum(row["status"] == "arrived" and check["final_holding_duration"] < 1 - 1e-8 for row, check in pairs),
            max_abs_position=max(check["max_abs_position"] for row, check in pairs),
        ))
    report = dict(
        scope="Read-only recomputation from saved trajectories; not a training rerun or exact rational spectral re-certification.",
        collision_method="Analytic closest point on each affine segment, computed in floating point with 1e-9 m reporting tolerance.",
        rows=len(rows), unique_keys=len({(row['scene'], row['tag']) for row in rows}),
        summary_rows=len(summary), unique_scenes=len(scene_inputs),
        summary_status_mismatches=sum(summary_map[(row['scene'], row['tag'])]['status'] != row['status'] for row in rows),
        input_mismatches=input_mismatches, groups=grouped, trajectories=numeric_checks,
    )
    (HERE / "saved_results_audit.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "trajectories"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
