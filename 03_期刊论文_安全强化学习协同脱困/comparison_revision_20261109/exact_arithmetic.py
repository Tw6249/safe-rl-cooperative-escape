import sys
from fractions import Fraction as PythonFraction
from pathlib import Path


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "gmp_runtime"))
import gmpy2


class RationalType(type):
    def __instancecheck__(cls, value):
        return isinstance(value, gmpy2.mpq)


class ExactRational(metaclass=RationalType):
    def __new__(cls, *arguments):
        fraction = PythonFraction(*arguments)
        return gmpy2.mpq(fraction.numerator, fraction.denominator)


def enable(short_circuit=False):
    import safety_checks
    safety_checks.Fraction = ExactRational
    if short_circuit:
        import audit_trajectories
        original = safety_checks.exact_affine_segment_certificate

        def certificate(start, end, radius, minimum_distance, lambda_required):
            first = [[ExactRational(str(value)) for value in row] for row in start]
            last = [[ExactRational(str(value)) for value in row] for row in end]
            required = ExactRational(str(minimum_distance)) ** 2
            for left in range(len(first)):
                for right in range(left + 1, len(first)):
                    relative = [first[left][axis] - first[right][axis] for axis in range(2)]
                    change = [last[left][axis] - last[right][axis] - relative[axis] for axis in range(2)]
                    denominator = sum(value * value for value in change)
                    fraction = max(0, min(1, -sum(relative[axis] * change[axis] for axis in range(2)) / denominator)) if denominator else 0
                    if sum((relative[axis] + fraction * change[axis]) ** 2 for axis in range(2)) < required:
                        return {"certified": False, "collision_safe": False}
            return original(start, end, radius, minimum_distance, lambda_required)

        audit_trajectories.exact_affine_segment_certificate = certificate
