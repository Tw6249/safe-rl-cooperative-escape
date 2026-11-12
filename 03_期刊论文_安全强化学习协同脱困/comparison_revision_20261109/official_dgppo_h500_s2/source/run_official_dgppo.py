import argparse
import hashlib
import json
import pickle
import time
from pathlib import Path

from dgppo_adapter import HERE, LatticeEnv, jax, jnp, np
from dgppo.algo.dgppo import DGPPO


def create_algorithm(environment, seed, updates, batch_size):
    return DGPPO(environment, environment.node_dim, environment.edge_dim, environment.state_dim,
                 environment.action_dim, environment.num_agents, seed=seed, use_rnn=False,
                 rnn_step=20 if environment.max_episode_steps % 20 == 0 else 16, batch_size=batch_size, epoch_ppo=1, train_steps=updates,
                 actor_gnn_layers=2, Vl_gnn_layers=2, Vh_gnn_layers=1)


def train(arguments):
    output = Path(arguments.output).resolve()
    output.mkdir(exist_ok=False)
    counts = list(range(6, 11)) if arguments.mixed else [8]
    environments = {count: LatticeEnv(HERE / "mixed_scene_banks" / f"train_n{count}.npz" if arguments.mixed else HERE / "dgppo_scene_banks" / "train.npz", max_step=arguments.horizon_steps) for count in counts}
    transitions_per_update = 2 * arguments.environments * arguments.horizon_steps
    updates = max(1, int(np.ceil(arguments.budget / transitions_per_update)))
    algorithms = {count: create_algorithm(environment, arguments.seed, updates, arguments.environments * arguments.horizon_steps) for count, environment in environments.items()}
    algorithm = algorithms[8]
    np.random.seed(arguments.seed)
    generator = jax.random.PRNGKey(arguments.seed + 1)
    metadata = dict(identity="Official DGPPO update, adapted single-integrator lattice environment",
                    upstream_commit="bf9ff80ebc7f135121b9fe1658800bdb106b8c20",
                    mode="smoke_only" if arguments.smoke else "adaptation_training",
                    seed=arguments.seed, requested_budget=arguments.budget,
                    horizon_steps=arguments.horizon_steps, environments=arguments.environments,
                    transitions_per_update=transitions_per_update, updates=updates,
                    budget_counting="Both stochastic collection and deterministic collection inside DGPPO.update",
                    training_agents=counts, training_distribution="Finite banks sampled from frozen original train_scene; mixed-size updates" if arguments.mixed else "Bank sampled from original train_scene conditioned on n=8",
                    observations="6D nodes and 5D edges; includes exact spectral margin and workspace clearance",
                    reward="Normalized progress minus 0.001 per step; not the EGP-NPA dual-dependent reward",
                    physical_model="p_next=p+0.05*u; radial speed bound 0.5; no state clipping",
                    limitations="Task adaptation; bank sampling and size-homogeneous batches differ from online scene generation. Native policy architecture and objectives are retained and disclosed.",
                    devices=[str(device) for device in jax.devices()])
    (output / "protocol.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    steps = 0
    size_generator = np.random.default_rng(arguments.seed + 2)
    if arguments.mixed:
        size_counts = json.loads((HERE / "mixed_scene_banks" / "manifest.json").read_text(encoding="utf-8"))["accepted_counts"]
        size_probabilities = np.array([size_counts[str(count)] for count in counts], dtype=float)
        size_probabilities /= size_probabilities.sum()
    source_hashes = {}
    for source in (HERE / "vendor" / "dgppo" / "dgppo").rglob("*.py"):
        source_hashes[str(source.relative_to(HERE))] = hashlib.sha256(source.read_bytes()).hexdigest()
    for source in (Path(__file__), HERE / "dgppo_adapter.py"):
        source_hashes[source.name] = hashlib.sha256(source.read_bytes()).hexdigest()
    (output / "source_sha256.json").write_text(json.dumps(source_hashes, indent=2), encoding="utf-8")
    started = time.perf_counter()
    with (output / "training.jsonl").open("w", encoding="utf-8") as logfile:
        for iteration in range(updates):
            count = (counts[iteration % len(counts)] if arguments.smoke else int(size_generator.choice(counts, p=size_probabilities))) if arguments.mixed else 8
            selected = algorithms[count]
            for name in ("policy_train_state", "Vl_train_state", "Vh_train_state"):
                old, new = getattr(algorithm, name), getattr(selected, name)
                setattr(selected, name, new.replace(params=old.params, opt_state=old.opt_state, step=old.step))
            selected.key = algorithm.key
            algorithm = selected
            generator, subkey = jax.random.split(generator)
            keys = jax.random.split(subkey, arguments.environments)
            rollout = algorithm.collect(algorithm.params, keys)
            info = algorithm.update(rollout, iteration)
            jax.block_until_ready(algorithm.policy_train_state.params)
            steps += transitions_per_update
            values = {key: float(np.asarray(value)) for key, value in info.items()}
            record = dict(iteration=iteration, agents=count, environment_transitions=steps,
                          elapsed_seconds=time.perf_counter() - started,
                          stochastic_mean_reward=float(np.asarray(rollout.rewards).mean()),
                          stochastic_unsafe_fraction=float(np.asarray(rollout.costs).max(axis=3).__gt__(0).mean()),
                          metrics=values)
            if not all(np.isfinite(value) for value in values.values()):
                raise FloatingPointError("Official update returned a nonfinite metric")
            logfile.write(json.dumps(record) + "\n")
            logfile.flush()
            print(json.dumps({key: value for key, value in record.items() if key != "metrics"}), flush=True)
    algorithm.save(str(output / "models"), steps)
    (output / "done.json").write_text(json.dumps(dict(environment_transitions=steps,
                                                          elapsed_seconds=time.perf_counter() - started,
                                                          smoke=arguments.smoke), indent=2), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=110201)
    parser.add_argument("--budget", type=int, default=960000)
    parser.add_argument("--environments", type=int, default=4)
    parser.add_argument("--horizon-steps", type=int, default=500)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--mixed", action="store_true")
    parser.add_argument("--smoke-updates", type=int, default=1)
    arguments = parser.parse_args()
    if arguments.smoke:
        arguments.environments = 2
        arguments.horizon_steps = 32
        arguments.budget = 128 * arguments.smoke_updates
    train(arguments)


if __name__ == "__main__":
    main()
