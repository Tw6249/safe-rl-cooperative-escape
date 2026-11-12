import argparse
import hashlib
import json
from pathlib import Path

from dgppo_adapter import HERE, LatticeEnv, LatticeState, jax, jnp, np
from run_official_dgppo import create_algorithm
from evaluation import evaluate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--training", required=True)
    parser.add_argument("--output", required=True)
    arguments = parser.parse_args()
    training = Path(arguments.training).resolve()
    output = Path(arguments.output).resolve()
    output.mkdir(exist_ok=False)
    training_protocol = json.loads((training / "protocol.json").read_text(encoding="utf-8"))
    completed = json.loads((training / "done.json").read_text(encoding="utf-8"))
    if completed["smoke"]:
        raise ValueError("Smoke models cannot be used for performance evaluation")
    model_files = {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in
                   (training / "models" / str(completed["environment_transitions"])).glob("*.pkl")}
    (output / "evaluation_protocol.json").write_text(json.dumps(dict(training=str(training),
        model_sha256=model_files, training_protocol=training_protocol,
        selection="Pre-specified final budget model; no selection on the new test",
        physical_horizon=60.0, hold_seconds=1.0, independent_audit="performed separately"), indent=2), encoding="utf-8")
    rows = []
    for count in (8, 12, 20, 25):
        bank_path = HERE / "external_test" / f"n{count}.npz"
        environment = LatticeEnv(bank_path)
        algorithm = create_algorithm(environment, 0, 1, 128)
        algorithm.load(str(training / "models"), completed["environment_transitions"])
        parameters = algorithm.params
        if count == 8:
            parameter_counts = {name: int(sum(np.asarray(leaf).size for leaf in jax.tree_util.tree_leaves(tree))) for name, tree in parameters.items()}
            (output / "parameter_counts.json").write_text(json.dumps(parameter_counts, indent=2), encoding="utf-8")

        @jax.jit
        def command(position, goal, threshold, initial_distance):
            state = LatticeState(position, goal, threshold, initial_distance)
            graph = environment.get_graph(state)
            return algorithm.act(graph, algorithm.init_rnn_state, params=parameters)[0]

        with np.load(bank_path) as bank:
            positions, goals, thresholds, names = bank["positions"], bank["goals"], bank["thresholds"], bank["names"]
        for index, name in enumerate(names):
            scene = dict(name=str(name), p=positions[index], goal=goals[index], lam=float(thresholds[index]))
            initial_distance = max(float(np.linalg.norm(scene["p"] - scene["goal"], axis=1).sum()), 1e-9)

            def controller(position):
                raw_action = np.asarray(command(jnp.asarray(position, dtype=jnp.float32), jnp.asarray(scene["goal"], dtype=jnp.float32),
                                                jnp.asarray(scene["lam"], dtype=jnp.float32), jnp.asarray(initial_distance, dtype=jnp.float32)), dtype=np.float64)
                return 0.5 * raw_action / np.maximum(1.0, np.linalg.norm(raw_action, axis=1, keepdims=True))

            row = evaluate(scene, controller, output / f"{name}.npz")
            row.update(n=count, lambda_required=scene["lam"], method="official_dgppo_adapted", audit_status="pending_independent_audit")
            (output / f"{name}.json").write_text(json.dumps(row, indent=2), encoding="utf-8")
            rows.append(row)
            print(f"DGPPO: {name}: {row['status']}", flush=True)
    (output / "summary_unaudited.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
