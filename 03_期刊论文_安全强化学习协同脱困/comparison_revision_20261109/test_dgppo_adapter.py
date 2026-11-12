import unittest

from dgppo_adapter import HERE, LatticeEnv, LatticeState, jax, jnp, np, physical_velocity, spectral_connectivity


class AdapterChecks(unittest.TestCase):
    def setUp(self):
        self.environment = LatticeEnv(HERE / "dgppo_scene_banks" / "train.npz", max_step=32)
        self.graph = self.environment.reset(jax.random.PRNGKey(1))

    def test_exact_same_polynomial_spectrum_as_numpy(self):
        position = np.asarray(self.graph.env_states.position, dtype=np.float64)
        squared = np.sum((position[:, None] - position[None, :]) ** 2, axis=2)
        weights = np.maximum(1 - squared / 100, 0) ** 2 * (1 - np.eye(len(position)))
        expected = np.linalg.eigvalsh(np.diag(weights.sum(axis=1)) - weights)[1]
        self.assertAlmostEqual(float(spectral_connectivity(jnp.asarray(position))), expected, places=4)

    def test_command_is_integrated_without_workspace_clipping(self):
        state = self.graph.env_states
        position = state.position.at[0, 0].set(3.99)
        graph = self.environment.get_graph(LatticeState(position, state.goal, state.threshold, state.initial_distance))
        action = jnp.zeros((8, 2)).at[0, 0].set(1.0)
        result = self.environment.step(graph, action)
        self.assertAlmostEqual(float(result.graph.env_states.position[0, 0]), 4.015, places=5)
        np.testing.assert_allclose(np.asarray(result.graph.env_states.position), np.asarray(position + 0.05 * physical_velocity(action)), atol=1e-7)

    def test_cost_signs_and_input_shapes(self):
        self.assertEqual(self.graph.nodes.shape, (9, 6))
        self.assertEqual(self.graph.edges.shape, (64, 5))
        self.assertEqual(self.environment.get_cost(self.graph).shape, (8, 3))
        state = self.graph.env_states
        position = state.position.at[1].set(state.position[0] + jnp.array([0.25, 0.0]))
        graph = self.environment.get_graph(LatticeState(position, state.goal, state.threshold, state.initial_distance))
        self.assertGreater(float(self.environment.get_cost(graph)[0, 0]), 0.7)
        speeds = np.linalg.norm(np.asarray(physical_velocity(jnp.ones((8, 2)))), axis=1)
        np.testing.assert_allclose(speeds, 0.5, atol=1e-7)


if __name__ == "__main__":
    unittest.main()
