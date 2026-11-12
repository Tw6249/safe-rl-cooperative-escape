"""Independent metrics and whole-segment bounds for first-order robot motion.

The segment bounds are analytic, evaluated in floating-point arithmetic. They
are sufficient tests, not an exact characterization of spectral-safe paths.
They apply only to the specified affine interpolation (e.g. zero-order hold).
"""
from fractions import Fraction
import numpy as np


def pair_indices(n):
    return np.triu_indices(n, 1)


def weights_from_squared_distance(distance2, radius):
    return np.maximum(0.0, 1.0 - np.asarray(distance2) / radius**2)**2


def laplacian_from_weights(n, weights):
    i, j = pair_indices(n)
    w = np.zeros((n, n))
    w[i, j] = weights
    w[j, i] = weights
    return np.diag(w.sum(axis=1)) - w


def configuration_metrics(p, radius):
    p = np.asarray(p, dtype=float)
    i, j = pair_indices(len(p))
    distance2 = np.sum((p[i] - p[j])**2, axis=1)
    laplacian = laplacian_from_weights(len(p), weights_from_squared_distance(distance2, radius))
    return {
        'min_distance': float(np.sqrt(distance2.min())),
        'lambda2': float(np.linalg.eigvalsh(laplacian)[1]),
    }


def affine_segment_bounds(p0, p1, radius):
    """Bound all p(s)=(1-s)p0+s*p1, 0<=s<=1, without time sampling.

    Pairwise squared distance is convex quadratic. Its maximum is at an end
    and minimum at an end or the projected quadratic minimizer. Each weight
    is decreasing in squared distance, so its endpoint-minimum is a lower
    bound on the entire segment. The resulting Laplacian is a Loewner lower
    bound; its second eigenvalue is a conservative connectivity certificate.
    """
    p0, p1 = np.asarray(p0, dtype=float), np.asarray(p1, dtype=float)
    if p0.shape != p1.shape or p0.ndim != 2 or p0.shape[1] != 2 or len(p0) < 2:
        raise ValueError('Matching (N,2) configurations, N>=2, are required')
    if radius <= 0 or not np.isfinite(p0).all() or not np.isfinite(p1).all():
        raise ValueError('Finite positions and a positive radius are required')
    i, j = pair_indices(len(p0))
    a = p0[i] - p0[j]
    b = (p1[i] - p1[j]) - a
    b2 = np.sum(b*b, axis=1)
    s = np.clip(np.divide(-np.sum(a*b, axis=1), b2,
                         out=np.zeros_like(b2), where=b2>0), 0, 1)
    pair_min2 = np.sum((a+s[:, None]*b)**2, axis=1)
    pair_max2 = np.maximum(np.sum(a*a, axis=1), np.sum((a+b)**2, axis=1))
    lower = laplacian_from_weights(len(p0), weights_from_squared_distance(pair_max2, radius))
    return {
        'min_distance': float(np.sqrt(pair_min2.min())),
        'lambda2_lower': float(np.linalg.eigvalsh(lower)[1]),
        'pair_min_distance': np.sqrt(pair_min2),
        'pair_max_distance': np.sqrt(pair_max2),
    }


def exact_affine_segment_certificate(p0, p1, radius, minimum_distance, lambda_required):
    """Exact rational certificate for the supplied decimal endpoint values.

    All geometric tests and a reduced-Laplacian LDL factorization use Fraction.
    Strict spectral positivity is required; False means not certified by this
    sufficient bound, not proof that the actual segment violates connectivity.
    The certified path is the specified straight segment, not an ODE interpolant.
    """
    def rational(x):
        return x if isinstance(x, Fraction) else Fraction(str(x))

    a = [[rational(x) for x in row] for row in p0]
    z = [[rational(x) for x in row] for row in p1]
    n = len(a)
    if n < 2 or len(z) != n or any(len(row) != 2 for row in a+z):
        raise ValueError('Matching (N,2) configurations, N>=2, are required')
    r2, d2, lam = rational(radius)**2, rational(minimum_distance)**2, rational(lambda_required)
    if rational(radius) <= 0 or rational(minimum_distance) < 0 or lam <= 0:
        raise ValueError('Positive radius/spectral threshold and nonnegative clearance required')
    lap = [[Fraction(0) for _ in range(n)] for _ in range(n)]
    collision_safe = True
    smallest_margin = None
    for i in range(n):
        for j in range(i+1, n):
            x = [a[i][k]-a[j][k] for k in range(2)]
            y = [z[i][k]-z[j][k] for k in range(2)]
            b = [y[k]-x[k] for k in range(2)]
            b2 = sum(t*t for t in b)
            s = max(Fraction(0), min(Fraction(1), -sum(x[k]*b[k] for k in range(2))/b2)) if b2 else Fraction(0)
            lo = sum((x[k]+s*b[k])**2 for k in range(2))
            hi = max(sum(t*t for t in x), sum(t*t for t in y))
            collision_safe &= lo >= d2
            smallest_margin = lo-d2 if smallest_margin is None else min(smallest_margin, lo-d2)
            w = max(Fraction(0), 1-hi/r2)**2
            lap[i][i] += w; lap[j][j] += w
            lap[i][j] -= w; lap[j][i] -= w
    # Columns of B are e_i-e_N. B spans 1-perp but is not orthonormal:
    # test B' L_lower B - lambda B' B > 0, retaining the Gram matrix.
    m = n-1
    reduced = [[lap[i][j]-lap[i][-1]-lap[-1][j]+lap[-1][-1]
                -lam*(1+int(i==j)) for j in range(m)] for i in range(m)]
    pivots = []
    for k in range(m):
        pivot = reduced[k][k]
        pivots.append(pivot)
        if pivot <= 0:
            break
        for i in range(k+1, m):
            for j in range(k+1, m):
                reduced[i][j] -= reduced[i][k]*reduced[k][j]/pivot
    spectral_safe = len(pivots) == m and all(x > 0 for x in pivots)
    return {'certified': bool(collision_safe and spectral_safe),
            'collision_safe': bool(collision_safe),
            'strict_spectral_certificate': spectral_safe,
            'minimum_squared_distance_margin': str(smallest_margin),
            'arithmetic': 'exact rational interpretation of supplied decimal values'}


def exact_taylor_segment_certificate(p0, u, h, radius, minimum_distance, lambda_required):
    """Exact rational certificate of the segment p0+s*u, 0<=s<=h, from the second-order Taylor bound
    (Proposition 6, lemma):  L(p0+s u) >= M(s) := L(p0) + s DL(p0)[u] - (c s^2/2) |u|^2 Pi,  c = 8N/R^2,
    Pi = I - 11^T/N.  M(s) - lambda Pi is concave in s (Loewner order), so it is >= 0 on [0,h] if it is >= 0 at
    s=0 and s=h; both are tested with the exact LDL used above (columns e_i-e_N, Gram matrix retained).
    Collisions use the exact quadratic minimum along the segment. This is the test the curvature row is built to
    pass, so it does not lose the first-order gains of shortening edges (the max-distance bound above does).
    """
    def q(x):
        return x if isinstance(x, Fraction) else Fraction(str(x))
    n = len(p0)
    a = [[q(x) for x in row] for row in p0]
    v = [[q(x) for x in row] for row in u]
    hh, R2, d2, lam = q(h), q(radius)**2, q(minimum_distance)**2, q(lambda_required)
    z = [[a[i][k]+hh*v[i][k] for k in range(2)] for i in range(n)]
    c = Fraction(8*n)/R2
    unorm2 = sum(v[i][k]**2 for i in range(n) for k in range(2))
    L0 = [[Fraction(0)]*n for _ in range(n)]
    D = [[Fraction(0)]*n for _ in range(n)]
    collision_safe = True
    for i in range(n):
        for j in range(i+1, n):
            x = [a[i][k]-a[j][k] for k in range(2)]
            y = [z[i][k]-z[j][k] for k in range(2)]
            b = [y[k]-x[k] for k in range(2)]
            b2 = sum(t*t for t in b)
            s = max(Fraction(0), min(Fraction(1), -sum(x[k]*b[k] for k in range(2))/b2)) if b2 else Fraction(0)
            collision_safe &= sum((x[k]+s*b[k])**2 for k in range(2)) >= d2
            zz = sum(t*t for t in x)
            g = max(Fraction(0), 1-zz/R2)
            w = g*g
            dw = -4*g/R2*sum(x[k]*(v[i][k]-v[j][k]) for k in range(2))   # d/ds of w along u
            for M, val in ((L0, w), (D, dw)):
                M[i][i] += val; M[j][j] += val; M[i][j] -= val; M[j][i] -= val

    def pd(sv):
        shift = lam + c*sv*sv/2*unorm2
        A = [[L0[i][j]+sv*D[i][j] - shift*((1 if i == j else 0) - Fraction(1, n)) for j in range(n)] for i in range(n)]
        m = n-1
        red = [[A[i][j]-A[i][-1]-A[-1][j]+A[-1][-1] for j in range(m)] for i in range(m)]
        for k in range(m):
            piv = red[k][k]
            if piv <= 0:
                return False
            for i in range(k+1, m):
                for j in range(k+1, m):
                    red[i][j] -= red[i][k]*red[k][j]/piv
        return True
    spectral_safe = pd(Fraction(0)) and pd(hh)
    return {'certified': bool(collision_safe and spectral_safe), 'collision_safe': bool(collision_safe),
            'strict_spectral_certificate': bool(spectral_safe), 'endpoint': z,
            'arithmetic': 'exact rational; second-order Taylor bound of Proposition 6'}
