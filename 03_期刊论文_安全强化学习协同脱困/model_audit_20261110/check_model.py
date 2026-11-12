import json
from fractions import Fraction
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent


def laplacian(position, local=False):
    squared = np.sum((position[:, None] - position[None, :]) ** 2, axis=2)
    weights = ((squared <= 2.5**2).astype(float) if local else np.maximum(1 - squared / 100, 0) ** 2)
    np.fill_diagonal(weights, 0)
    return np.diag(weights.sum(axis=1)) - weights


def spectral_matrix_example():
    generator = np.random.default_rng(111010)
    for attempt in range(20000):
        position = generator.uniform(-2.0, 2.0, (3, 2))
        left, right = np.triu_indices(3, 1)
        relative = position[left] - position[right]
        squared = np.sum(relative**2, axis=1)
        if squared.min() < 1.3**2:
            continue
        matrix = laplacian(position)
        values, vectors = np.linalg.eigh(matrix)
        gradient_weight = -0.04 * (1 - squared / 100)[:, None] * relative
        gradient = np.zeros_like(position)
        contribution = (vectors[left, 1] - vectors[right, 1])[:, None] ** 2 * gradient_weight
        np.add.at(gradient, left, contribution)
        np.add.at(gradient, right, -contribution)
        velocity = generator.normal(size=position.shape)
        velocity -= gradient * (np.sum(gradient * velocity) / np.sum(gradient**2))
        velocity += 1e-5 * gradient
        velocity *= 0.49 / np.linalg.norm(velocity, axis=1).max()
        derivative = np.zeros((3, 3))
        for index, (source, target) in enumerate(zip(left, right)):
            incidence = np.eye(3)[source] - np.eye(3)[target]
            rate = gradient_weight[index] @ (velocity[source] - velocity[target])
            derivative += rate * np.outer(incidence, incidence)
        threshold = values[1] - 0.001
        complement = vectors[:, 1:]
        residual = complement.T @ derivative @ complement + 2 * np.diag(values[1:] - threshold)
        collision_residual = 2 * np.sum(relative * (velocity[left] - velocity[right]), axis=1) + 2 * (squared - 1)
        scalar_residual = float(np.sum(gradient * velocity))
        eigenvalue = float(np.linalg.eigvalsh(residual)[0])
        if eigenvalue < -1e-3 and collision_residual.min() > 0 and scalar_residual > 0:
            return dict(position=position.tolist(), velocity=velocity.tolist(), threshold=float(threshold),
                        scalar_fiedler_residual_with_clipped_margin=scalar_residual,
                        full_matrix_cbf_min_eigenvalue=eigenvalue,
                        minimum_collision_cbf_residual=float(collision_residual.min()),
                        max_speed=float(np.linalg.norm(velocity, axis=1).max()),
                        lambda2=float(values[1]), eigengap=float(values[2] - values[1]), attempt=attempt)
    raise RuntimeError("No scalar/matrix separation witness found")


def exact_kkt():
    coordinates = [(-2, 0), (-1, 1), (-1, -1), (0, 0), (1, 1), (1, -1), (2, 0)]
    points = [[Fraction(horizontal, 2), Fraction(vertical, 2)] for horizontal, vertical in coordinates]
    witness = [[Fraction(value, 496) for value in pair] for pair in [(-15,15),(-132,132),(-105,-15),(90,206),(177,177),(-105,-225),(90,-290)]]
    contacts = [(0,1),(0,2),(0,3),(1,3),(1,4),(2,3),(2,5),(3,4),(3,5),(3,6),(4,6),(5,6)]
    multipliers = {(0,2):Fraction(15,248), (1,3):Fraction(33,62), (2,5):Fraction(15,62), (3,4):Fraction(177,248), (5,6):Fraction(225,248)}
    gradient = [[Fraction(0),Fraction(0)] for unused in points]
    slacks = []
    for source, target in contacts:
        direction = [points[source][0]-points[target][0], 3*(points[source][1]-points[target][1])]
        slack = sum(direction[axis]*(witness[source][axis]-witness[target][axis]) for axis in (0,1))
        slacks.append(slack)
        multiplier = multipliers.get((source,target),Fraction(0))
        assert multiplier*slack == 0
        for axis in (0,1):
            gradient[source][axis] += multiplier*direction[axis]
            gradient[target][axis] -= multiplier*direction[axis]
    for axis, multiplier in enumerate((Fraction(135,496),Fraction(1545,496))):
        gradient[3][axis] += multiplier
        gradient[6][axis] -= multiplier
    stationary = all(gradient[index][axis] == witness[index][axis]*(1 if axis == 0 else 3) for index in range(7) for axis in (0,1))
    objective = sum(horizontal**2+3*vertical**2 for horizontal,vertical in witness)
    assert stationary and min(slacks) >= 0 and objective == Fraction(1545,496)
    return dict(stationarity_exact=stationary, feasible=True, complementarity=True,
                scaled_objective=str(objective), physical_normalized_objective=str(objective/3),
                scope="Checks the stated rational QP witness only; not the entire finite-radius or reachability proof")


def main():
    positions = np.array([[-3.0,0.0],[-1.9,0.0],[1.9,0.0],[3.0,0.0]])
    physical_spectrum = np.linalg.eigvalsh(laplacian(positions))
    communication_spectrum = np.linalg.eigvalsh(laplacian(positions,local=True))
    lambda2 = Fraction(2)*(1-Fraction(4,100))**2
    threshold = lambda2 - Fraction(1,100)
    effective_threshold = threshold + Fraction(2,100)
    paper_b = -(lambda2-effective_threshold)
    code_b = -max(lambda2-effective_threshold,Fraction(0))
    report = dict(
        kkt_witness=exact_kkt(),
        graph_separation=dict(positions=positions.tolist(), physical_lambda2=float(physical_spectrum[1]),
                              communication_lambda2=float(communication_spectrum[1]),
                              implication="Maintaining the 10 m physical graph need not maintain the 2.5 m coordination graph"),
        zero_speed_feasibility=dict(two_positions=[[0,0],[2,0]],lambda2=str(lambda2),lambda_required=str(threshold),
                                    margin="1/50", paper_spectral_b_per_agent=str(paper_b),
                                    code_spectral_b_per_agent=str(code_b),
                                    implication="A state strictly inside the stated spectral safe set need not satisfy the un-clipped positive-margin zero-input condition"),
        scalar_vs_matrix=spectral_matrix_example(),
        kernel_cutoff=dict(radius=10, second_derivative_inside_limit=0.08, second_derivative_outside=0,
                           smoothness="C1 and locally Lipschitz gradient, but not C2 at the cutoff"),
        envelope_counterexample=dict(dynamics="e_dot=-e+2e=e",residual="r(e)=2e=O(|e|)",
                                      implication="First-order vanishing and equilibrium restoration alone do not imply asymptotic stability"))
    (HERE / "model_checks.json").write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps(report,ensure_ascii=False,indent=2))


if __name__ == "__main__":
    main()
