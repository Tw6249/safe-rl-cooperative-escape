import os
import sys
from pathlib import Path
from typing import NamedTuple

os.environ["WANDB_MODE"] = "disabled"
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "vendor" / "dgppo"))

import jax
import jax.numpy as jnp
import numpy as np

from dgppo.env.base import MultiAgentEnv, StepResult
from dgppo.utils.graph import EdgeBlock, GetGraph


class LatticeState(NamedTuple):
    position: object
    goal: object
    threshold: object
    initial_distance: object


def physical_velocity(action):
    norm = jnp.linalg.norm(action, axis=1, keepdims=True)
    return 0.5 * action / jnp.maximum(1.0, norm)


def spectral_connectivity(position):
    relative = position[:, None] - position[None, :]
    squared = jnp.sum(relative * relative, axis=-1)
    weights = jnp.maximum(1.0 - squared / 100.0, 0.0) ** 2
    weights = weights * (1.0 - jnp.eye(len(position)))
    laplacian = jnp.diag(weights.sum(axis=1)) - weights
    return jnp.linalg.eigvalsh(laplacian)[1]


class LatticeEnv(MultiAgentEnv):
    def __init__(self, bank_path, max_step=128):
        with np.load(bank_path) as bank:
            self.positions = jnp.asarray(bank["positions"], dtype=jnp.float32)
            self.goals = jnp.asarray(bank["goals"], dtype=jnp.float32)
            self.thresholds = jnp.asarray(bank["thresholds"], dtype=jnp.float32)
        super().__init__(self.positions.shape[1], 8.0, max_step=max_step, dt=0.05)

    @property
    def state_dim(self):
        return 2

    @property
    def node_dim(self):
        return 6

    @property
    def edge_dim(self):
        return 5

    @property
    def action_dim(self):
        return 2

    @property
    def n_cost(self):
        return 3

    @property
    def cost_components(self):
        return ("collision", "spectral connectivity", "workspace")

    def reset(self, key):
        index = jax.random.randint(key, (), 0, len(self.positions))
        position, goal, threshold = self.positions[index], self.goals[index], self.thresholds[index]
        initial_distance = jnp.maximum(jnp.linalg.norm(position - goal, axis=1).sum(), 1e-9)
        return self.get_graph(LatticeState(position, goal, threshold, initial_distance))

    def step(self, graph, action, get_eval_info=False):
        state = graph.env_states
        velocity = physical_velocity(action)
        position = state.position + self.dt * velocity
        next_state = LatticeState(position, state.goal, state.threshold, state.initial_distance)
        next_graph = self.get_graph(next_state)
        old_distance = jnp.linalg.norm(state.position - state.goal, axis=1).sum()
        new_distance = jnp.linalg.norm(position - state.goal, axis=1).sum()
        reward = (old_distance - new_distance) / state.initial_distance - 1e-3
        cost = self.get_cost(graph)
        return StepResult(next_graph, reward, cost, jnp.array(False), {})

    def get_cost(self, graph):
        state = graph.env_states
        relative = state.position[:, None] - state.position[None, :]
        distance = jnp.sqrt(jnp.sum(relative * relative, axis=2) + jnp.eye(self.num_agents))
        distance = distance + jnp.eye(self.num_agents) * 1e6
        collision = 1.0 - distance.min(axis=1)
        connectivity = (state.threshold - spectral_connectivity(state.position)) / jnp.maximum(state.threshold, 1e-6)
        workspace = (jnp.abs(state.position).max(axis=1) - 4.0) / 4.0
        raw_cost = jnp.stack([collision, jnp.full(self.num_agents, connectivity), workspace], axis=1)
        shaped_cost = jnp.where(raw_cost <= 0.0, raw_cost - 0.5, raw_cost + 0.5)
        return jnp.maximum(shaped_cost, -1.0)

    def state_lim(self, state=None):
        return jnp.full(2, -jnp.inf), jnp.full(2, jnp.inf)

    def action_lim(self):
        return jnp.full(2, -1.0), jnp.full(2, 1.0)

    def get_graph(self, state):
        position, goal = state.position, state.goal
        error = goal - position
        error_norm = jnp.linalg.norm(error, axis=1)
        lambda2 = spectral_connectivity(position)
        nodes = jnp.concatenate([error, error_norm[:, None],
                                 jnp.full((self.num_agents, 1), (lambda2 - state.threshold) / state.threshold),
                                 (4.0 - jnp.abs(position)),], axis=1)
        relative = position[:, None] - position[None, :]
        distance = jnp.linalg.norm(relative, axis=2)
        error_difference = error[:, None] - error[None, :]
        edge = jnp.concatenate([relative, distance[:, :, None], error_difference], axis=2)
        mask = (distance <= 2.5) & ~jnp.eye(self.num_agents, dtype=bool)
        indices = jnp.arange(self.num_agents)
        block = EdgeBlock(edge, mask, indices, indices)
        return GetGraph(nodes=nodes, node_type=jnp.zeros(self.num_agents, dtype=jnp.int32),
                        edge_blocks=[block], env_states=state, states=position).to_padded()

    def render_video(self, *args, **kwargs):
        raise NotImplementedError("Use the shared trajectory renderer")
