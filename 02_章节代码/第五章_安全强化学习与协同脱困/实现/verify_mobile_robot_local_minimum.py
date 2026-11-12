"""Exact reachable local minimum with seven fully movable planar robots.

The safe path is a reachability witness, never a controller input. All local
certificates use rational coordinates after scaling y by sqrt(3). No numerical
optimizer is needed to verify the critical-cone optimum or the growth bound.
"""
from pathlib import Path
import hashlib
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import sympy as sp

HERE = Path(__file__).resolve().parent
OUT = HERE / "results" / "mobile_robot_local_minimum"
R = sp.Rational


def exact_certificate():
    axial = [(q, r) for q in range(-1, 2) for r in range(-1, 2)
             if max(abs(q), abs(r), abs(q + r)) <= 1]
    p = sp.Matrix([[R(2 * q + r, 2), R(r, 2)] for q, r in axial])
    center, right = axial.index((0, 0)), axial.index((1, 0))
    qmat = sp.diag(*([1, 3] * 7))
    edges, rows = [], []
    for i in range(7):
        for j in range(i + 1, 7):
            x, y = p[i, 0] - p[j, 0], p[i, 1] - p[j, 1]
            if x * x + 3 * y * y != 1:
                continue
            row = sp.zeros(1, 14)
            row[0, 2 * i], row[0, 2 * j] = x, -x
            row[0, 2 * i + 1], row[0, 2 * j + 1] = 3 * y, -3 * y
            edges.append((i, j))
            rows.append(row)
    g = sp.Matrix.vstack(*rows)
    eq = sp.zeros(2, 14)
    eq[0, 2 * center], eq[0, 2 * right] = 1, -1
    eq[1, 2 * center + 1], eq[1, 2 * right + 1] = 1, -1
    primal = sp.Matrix([
        -R(15, 496), R(15, 496), -R(33, 124), R(33, 124),
        -R(105, 496), -R(15, 496), R(45, 248), R(103, 248),
        R(177, 496), R(177, 496), -R(105, 496), -R(225, 496),
        R(45, 248), -R(145, 248),
    ])
    mu = sp.zeros(len(edges), 1)
    for edge, value in {
        (0, 2): R(15, 248), (1, 3): R(33, 62), (2, 5): R(15, 62),
        (3, 4): R(177, 248), (5, 6): R(225, 248),
    }.items():
        mu[edges.index(edge)] = value
    nu = sp.Matrix([R(135, 496), R(1545, 496)])
    slack = g * primal
    assert all(value >= 0 for value in slack)
    assert eq * primal == sp.Matrix([0, 1])
    assert qmat * primal == g.T * mu + eq.T * nu
    assert all(mu[i] >= 0 and mu[i] * slack[i] == 0 for i in range(len(edges)))
    optimum = (primal.T * qmat * primal)[0]
    assert optimum == R(1545, 496)

    # Reflect across the horizontal axis to certify the opposite transverse sign.
    permutation = [axial.index((q + r, -r)) for q, r in axial]
    reflected = sp.Matrix([primal[2 * permutation[i] + c] * (1 if c == 0 else -1)
                           for i in range(7) for c in range(2)])
    reflected_mu = sp.Matrix([
        mu[edges.index(tuple(sorted((permutation[i], permutation[j]))))]
        for i, j in edges])
    reflected_nu = sp.Matrix([nu[0], -nu[1]])
    assert all(value >= 0 for value in g * reflected)
    assert eq * reflected == sp.Matrix([0, -1])
    assert qmat * reflected == g.T * reflected_mu + eq.T * reflected_nu

    # The task KKT multiplier is 1/2 on the exchanged pair, zero elsewhere.
    goal = p.copy()
    goal[center, :], goal[right, :] = p[right, :], p[center, :]
    task_gradient = sp.Matrix([
        (p[i, c] - goal[i, c]) * (1 if c == 0 else 3)
        for i in range(7) for c in range(2)])
    assert task_gradient == g[edges.index((center, right)), :].T
    assert sum((p[i, 0] - goal[i, 0]) ** 2
               + 3 * (p[i, 1] - goal[i, 1]) ** 2 for i in range(7)) / 2 == 1
    assert all((goal[i, 0] - goal[j, 0]) ** 2
               + 3 * (goal[i, 1] - goal[j, 1]) ** 2 >= 1
               for i in range(7) for j in range(i + 1, 7))

    physical_ratio = optimum / 3
    coercivity = 1 - 1 / physical_ratio
    assert physical_ratio == R(515, 496) and coercivity == R(19, 515)
    dual_sum = sum(mu)
    assert dual_sum == R(609, 248)
    assert dual_sum ** 2 <= R(9, 4) * optimum
    local_radius, growth = R(1, 200), R(1, 100)
    growth_lower_bound = (1 - R(496, 515) * (1 + R(3, 2) * local_radius) ** 2
                          - 4 * local_radius ** 2) / 2
    assert growth_lower_bound >= growth

    # The obstruction persists with strictly interior starts and goals.
    # The goal perturbation changes the local task difference by -tau*goal.T*delta.
    initial_expansion, goal_expansion = R(1, 10**8), R(1, 10**6)
    position_norm_sq = sum(p[i, 0] ** 2 + 3 * p[i, 1] ** 2 for i in range(7))
    position_goal_inner = sum(p[i, 0] * goal[i, 0]
                              + 3 * p[i, 1] * goal[i, 1] for i in range(7))
    assert position_norm_sq == 6 and position_goal_inner == 5
    assert position_norm_sq < R(5, 2) ** 2
    sphere_barrier = growth * local_radius ** 2 - goal_expansion * R(5, 2) * local_radius
    entry_excess = (initial_expansion + 3 * initial_expansion ** 2
                    - 5 * initial_expansion * goal_expansion)
    assert 0 < entry_excess < sphere_barrier
    assert position_norm_sq * initial_expansion ** 2 < local_radius ** 2
    goal_offset_sq = 2 + 2 * goal_expansion + 6 * goal_expansion ** 2
    assert goal_offset_sq > local_radius ** 2
    assert 1 + initial_expansion < 3 and 1 + goal_expansion < 3
    initial_margin = 2 * initial_expansion + initial_expansion ** 2
    goal_margin = 2 * goal_expansion + goal_expansion ** 2
    assert initial_margin > 0 and goal_margin > 0
    open_radius = initial_expansion / 4
    initial_task = 1 + goal_expansion + 3 * goal_expansion ** 2 + entry_excess
    assert 2 * initial_task < 4  # Task gradient norm is below two at the interior start.
    assert R(5, 2) * initial_expansion + open_radius < local_radius
    assert entry_excess + 2 * open_radius + open_radius ** 2 / 2 < sphere_barrier
    assert 2 * open_radius < initial_expansion
    assert 1 + initial_expansion + open_radius < 3

    # Expand by three, rotate only the exchanged pair, then contract.
    # Every point stays in the radius-three ball. The rotating pair has radius
    # 3/2; every stationary robot is strictly more than radius+1 from its center.
    midpoint = sp.Matrix([[R(3, 2), 0]])
    rotation_radius = R(3, 2)
    center_distances_sq = []
    for i in range(7):
        if i in (center, right):
            continue
        x, y = 3 * p[i, 0] - midpoint[0, 0], 3 * p[i, 1]
        distance_sq = x * x + 3 * y * y
        assert distance_sq > (rotation_radius + 1) ** 2
        center_distances_sq.append(distance_sq)
    max_distance_sq = R(36)
    minimum_weight = (1 - max_distance_sq / 100) ** 2
    spectral_lower_bound = 7 * minimum_weight
    assert spectral_lower_bound == R(1792, 625) > R(5, 2)

    return dict(
        scope="Exact reachable strict task local minimum; not a deadlock-escape controller",
        scaled_positions=[[str(p[i, c]) for c in range(2)] for i in range(7)],
        physical_y="sqrt(3) times scaled y", exchanged_indices=[center, right],
        collision_distance="1", goal_is_safe=True, task_at_initial="1",
        zero_control_KKT_multiplier="1/2 on the exchanged pair; all others zero",
        critical_cone="G*v>=0, vx_center=vx_right",
        normalized_transverse_difference="physical vy_center-vy_right=sqrt(3)",
        contact_edges=edges, scaled_primal=[str(v) for v in primal],
        exact_dual=[str(v) for v in mu], equality_dual=[str(v) for v in nu],
        exact_KKT_and_reflection=True, minimum_norm_sq=str(optimum),
        minimum_norm_sq_per_physical_transverse_sq=str(physical_ratio),
        lagrangian_coercivity=str(coercivity),
        finite_local_growth=dict(radius=str(local_radius), coefficient=str(growth),
                                 bound="W(p+delta)-1 >= coefficient*||delta||^2",
                                 conservative_error_barrier=str(growth * local_radius ** 2)),
        strictly_interior_trap=dict(
            initial="(1+epsilon)*p_initial", goal="(1+tau)*p_exchanged",
            epsilon=str(initial_expansion), tau=str(goal_expansion),
            initial_minimum_collision_h=str(initial_margin),
            goal_minimum_collision_h=str(goal_margin),
            sphere_task_increase_lower_bound=str(sphere_barrier),
            initial_task_increase=str(entry_excess),
            initial_inside_ball=True, goal_outside_ball=True,
            strictly_below_exit_barrier=True,
            reachable_with_same_expand_exchange_contract_witness=True,
            scope="Every continuous safe trajectory with nonincreasing perturbed task stays in the ball; not a sampled-QP claim",
            certified_open_initial_ball_radius=str(open_radius),
            open_initial_set="Every start in this ball is strictly safe, below the exit barrier, and can first travel safely to the center of the ball"),
        reachable_witness=dict(
            stages=["uniformly expand by 3", "half-circle exchange of two labels", "uniformly contract"],
            note="Witness only; never used as input to any proposed controller",
            circle_to_stationary_center_distances_sq=[str(v) for v in center_distances_sq],
            all_path_collisions_safe=True, communication_radius="10",
            weight="(1-distance_squared/100)^2", all_path_spectral_lower_bound=str(spectral_lower_bound),
            required_spectrum="5/2", bounded_input="arbitrary positive bound by slowing the path",
            CBF_note="Contraction may approach contact goals asymptotically; no finite-time CBF boundary arrival asserted"),
    )


def plot_witness(report):
    p = np.array([[float(sp.Rational(x)), float(sp.Rational(y)) * np.sqrt(3)]
                  for x, y in report["scaled_positions"]])
    i, j = report["exchanged_indices"]
    goal = p.copy()
    goal[[i, j]] = goal[[j, i]]
    expanded = np.array([scale * p for scale in np.linspace(1, 3, 121)])
    rotated = []
    for theta in np.linspace(0, np.pi, 121)[1:]:
        state = 3 * p.copy()
        delta = 1.5 * np.array([np.cos(theta), -np.sin(theta)])
        state[i], state[j] = [1.5, 0] - delta, [1.5, 0] + delta
        rotated.append(state)
    contracted = np.array([scale * goal for scale in np.linspace(3, 1, 121)[1:]])
    states = np.concatenate([expanded, rotated, contracted])
    w = .5 * np.sum((states - goal) ** 2, axis=(1, 2))
    np.savez_compressed(OUT / "reachability_witness.npz", states=states, goal=goal,
                        W=w, scope="Prescribed reachability witness, not controller execution")
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), constrained_layout=True)
    colors = ["#74787d"] * 7
    colors[i], colors[j] = "#1976a3", "#c74b59"
    for k, color in enumerate(colors):
        axes[0].add_patch(plt.Circle(p[k], .5, facecolor=color, edgecolor="white", alpha=.8))
        axes[0].text(*p[k], str(k + 1), ha="center", va="center", color="white")
        axes[1].plot(states[:, k, 0], states[:, k, 1], color=color, alpha=.85)
    axes[0].set(xlim=(-1.65, 1.65), ylim=(-1.7, 1.55), title="All seven robots can move")
    axes[0].text(0, -1.56, f"Only labels {i + 1} and {j + 1} exchange goals", ha="center", fontsize=10)
    axes[1].set(title="Safe reachability witness (prescribed)", xlabel="x", ylabel="y")
    for ax in axes[:2]:
        ax.set_aspect("equal")
    axes[2].plot(np.linspace(0, 3, len(w)), w, color="#1976a3")
    axes[2].axhline(1, color="#c74b59", linestyle="--", label="Initial W = 1: strict local min")
    axes[2].set(title="A safe path exists, but W first rises", xlabel="Witness path parameter", ylabel="Task error W")
    axes[2].legend(fontsize=8)
    fig.savefig(OUT / "mobile_deadlock_counterexample.png", dpi=180)
    plt.close(fig)


def main():
    report = exact_certificate()
    OUT.mkdir(parents=True, exist_ok=True)
    plot_witness(report)
    report["source_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    (OUT / "audit.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("PASS: exact critical-cone KKT and opposite-sign reflection")
    print("PASS: ||v||^2 >= (515/496)*|vy_center-vy_right|^2 on the critical cone")
    print("PASS: W(p+delta)-1 >= ||delta||^2/100 for every safe 0<||delta||<=1/200")
    print("PASS: strictly interior initial/goal pair has a certified barrier for every nonincreasing-task safe trajectory")
    print("PASS: prescribed safe path exists; lambda2 >= 1792/625 > 5/2 throughout")


if __name__ == "__main__":
    main()
