from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class EvaluationConfig:
    dt: float = 0.05
    horizon: float = 60.0
    goal_tolerance: float = 0.01
    hold_seconds: float = 1.0
    stalled_speed: float = 1e-6
    stalled_seconds: float = 5.0


def evaluate(scene, controller, output: Path, config=EvaluationConfig()):
    position = np.array(scene["p"], dtype=float, copy=True)
    goal = np.asarray(scene["goal"], dtype=float)
    states, times, actions = [position.copy()], [0.0], []
    arrival = None
    quiet_since = None
    status = "horizon_reached"
    failure = None
    for step in range(int(np.ceil(config.horizon / config.dt))):
        elapsed = times[-1]
        interval = min(config.dt, config.horizon - elapsed)
        try:
            velocity = np.asarray(controller(position.copy()), dtype=float)
            if velocity.shape != position.shape or not np.isfinite(velocity).all():
                raise ValueError("Controller output is nonfinite or has an invalid shape")
        except Exception as error:
            status, failure = "controller_failure", repr(error)
            break
        position = position + interval * velocity
        now = min((step + 1) * config.dt, config.horizon)
        states.append(position.copy())
        actions.append(velocity.copy())
        times.append(now)
        if np.linalg.norm(position - goal, axis=1).max() <= config.goal_tolerance:
            if arrival is None:
                arrival = now
            if now - arrival >= config.hold_seconds - 1e-12:
                status = "arrived"
                break
        else:
            arrival = None
        if np.linalg.norm(velocity) < config.stalled_speed:
            if quiet_since is None:
                quiet_since = elapsed
            if now - quiet_since >= config.stalled_seconds - 1e-12:
                status = "stalled"
                break
        else:
            quiet_since = None
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite trajectory: {output}")
    np.savez_compressed(output, states=np.array(states), times=np.array(times), actions=np.array(actions), goal=goal)
    return dict(scene=scene["name"], status=status, failure=failure, steps=len(actions), duration=times[-1],
                arrival=arrival if status == "arrived" else None,
                holding_duration=times[-1] - arrival if status == "arrived" else None,
                final_max_err=float(np.linalg.norm(position - goal, axis=1).max()), config=asdict(config))
