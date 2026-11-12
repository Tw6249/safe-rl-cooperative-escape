"""Gaussian/C2-cutoff channel, graph derivatives and consistently signed CBF rows.

Distances: m; positions/controls: shape (N, d); weights: dimensionless.
No solver, simulation, file writes, or original-project imports at module import.
"""
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import numpy as np


@dataclass(frozen=True)
class ChannelConfig:
    sigma_w: float
    r_tap: float
    r_com: float
    w_max: float = 1.0

    def __post_init__(self):
        values = (self.sigma_w, self.r_tap, self.r_com, self.w_max)
        if not all(np.isfinite(v) and v > 0 for v in values):
            raise ValueError("Channel parameters must be positive finite scalars")
        if self.r_tap >= self.r_com:
            raise ValueError("Require 0 < r_tap < r_com")

    @classmethod
    def from_settings(cls, path=None):
        path = Path(path) if path else Path(__file__).resolve().parents[1] / "settings.json"
        settings = json.loads(path.read_text(encoding="utf-8"))
        if settings.get("schema_version") != 1 or settings["channel"]["model"] != "gaussian_quintic_c2":
            raise ValueError("Unsupported settings version or channel model")
        return cls(**settings["channel"]["parameters"])


def settings_digest(path=None):
    """SHA-256 of the exact settings file, for new experiment provenance."""
    path = Path(path) if path else Path(__file__).resolve().parents[1] / "settings.json"
    return hashlib.sha256(path.read_bytes()).hexdigest()


def channel_weight(distance, config):
    """Return w(d), dw/dd, d2w/dd2, with the same shape as distance.

    Scalars yield scalar floats. At and beyond r_com all three are exactly zero.
    Factored taper avoids negative weights caused by cancellation near cutoff.
    """
    d = np.asarray(distance, dtype=float)
    if not np.all(np.isfinite(d)) or np.any(d < 0):
        raise ValueError("Distances must be finite and nonnegative")
    flat = d.reshape(-1)
    w = np.zeros_like(flat); w1 = np.zeros_like(flat); w2 = np.zeros_like(flat)
    inside = flat < config.r_com
    x = flat[inside]
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        a = x / config.sigma_w
        exponential = np.exp(-(a * a))
        a_exp = np.where(exponential == 0, 0.0, a * exponential)
        a2_exp = np.where(exponential == 0, 0.0, a * a * exponential)
        base = config.w_max * exponential
        base1 = -2 * config.w_max * a_exp / config.sigma_w
        base2 = config.w_max * (4 * a2_exp - 2 * exponential) / config.sigma_w**2
    taper = np.ones_like(x); taper1 = np.zeros_like(x); taper2 = np.zeros_like(x)
    transition = x > config.r_tap
    width = config.r_com - config.r_tap
    z = (x[transition] - config.r_tap) / width
    taper[transition] = (1-z)**3 * (1+3*z+6*z*z)
    taper1[transition] = -30*z*z*(1-z)**2 / width
    taper2[transition] = -60*z*(1-z)*(1-2*z) / width**2
    w[inside] = base*taper
    w1[inside] = base1*taper+base*taper1
    w2[inside] = base2*taper+2*base1*taper1+base*taper2
    if np.any(w[inside] <= 0):
        raise ValueError("Weight underflow inside support; rescale channel parameters instead of dropping an edge")
    out = (w.reshape(d.shape), w1.reshape(d.shape), w2.reshape(d.shape))
    if not all(np.all(np.isfinite(v)) for v in out):
        raise ValueError("Parameter scale exceeds floating-point derivative range")
    return tuple(float(v) for v in out) if d.ndim == 0 else out


@dataclass(frozen=True)
class GraphData:
    adjacency: np.ndarray
    laplacian: np.ndarray
    # gradient_i[i,j,:] = partial w_ij / partial p_i (column in formulas).
    gradient_i: np.ndarray
    candidate_edges: tuple


def build_graph(positions, config, candidate_edges=None):
    """Build an undirected physical graph, preserving zero-weight candidate pairs.

    candidate_edges defaults to all pairs. Explicit edges are 0-based and must
    satisfy 0 <= i < j < N, without duplicates. Self loops are never allowed.
    """
    p = np.asarray(positions, dtype=float)
    if p.ndim != 2 or p.shape[0] < 2 or p.shape[1] < 1 or not np.all(np.isfinite(p)):
        raise ValueError("positions must be a finite (N,d) array, N >= 2")
    n, dim = p.shape
    edges = tuple((i,j) for i in range(n) for j in range(i+1,n)) if candidate_edges is None else tuple(tuple(e) for e in candidate_edges)
    if any(len(e)!=2 or any(not isinstance(v,(int,np.integer)) or isinstance(v,(bool,np.bool_)) for v in e) or not 0<=e[0]<e[1]<n for e in edges):
        raise ValueError("Each edge must contain integer indices 0 <= i < j < N")
    if len(set(edges)) != len(edges):
        raise ValueError("Duplicate candidate edges")
    adjacency = np.zeros((n,n)); gradients = np.zeros((n,n,dim))
    for i,j in edges:
        displacement = p[i]-p[j]
        d = float(np.linalg.norm(displacement))
        w, dw, _ = channel_weight(d, config)
        adjacency[i,j] = adjacency[j,i] = w
        gradient = dw*(displacement/d) if d > 0 else np.zeros(dim)
        gradients[i,j] = gradient
        gradients[j,i] = -gradient
    laplacian = np.diag(adjacency.sum(axis=1))-adjacency
    return GraphData(adjacency,laplacian,gradients,edges)


def laplacian_rate(graph, velocity):
    """Exact instantaneous dot L for dot p=u; not a sampled-data guarantee."""
    u = np.asarray(velocity,dtype=float)
    n,_,dim = graph.gradient_i.shape
    if u.shape != (n,dim) or not np.all(np.isfinite(u)):
        raise ValueError("velocity must match the finite (N,d) graph shape")
    wdot = np.zeros((n,n))
    for i,j in graph.candidate_edges:
        wdot[i,j] = wdot[j,i] = graph.gradient_i[i,j]@(u[i]-u[j])
    return np.diag(wdot.sum(axis=1))-wdot


def simple_lambda2_gradient(graph, *, gap_tolerance):
    """Return lambda_2 and its N-by-d position gradient only for a simple root.

    gap_tolerance is an explicit absolute numerical test in edge-weight units.
    It is not a certified spectral margin. Repeated or disconnected cases raise.
    """
    if not np.isfinite(gap_tolerance) or gap_tolerance < 0:
        raise ValueError("gap_tolerance must be finite and nonnegative")
    values,vectors = np.linalg.eigh(graph.laplacian)
    if values[1]-values[0] <= gap_tolerance or (len(values)>2 and values[2]-values[1] <= gap_tolerance):
        raise ValueError("lambda_2 is disconnected, repeated, or insufficiently separated")
    phi = vectors[:,1]  # unit Euclidean norm, not the SCL sqrt(N) state.
    gradient = np.zeros_like(graph.gradient_i[:,0,:])
    for i,j in graph.candidate_edges:
        contribution = (phi[i]-phi[j])**2 * graph.gradient_i[i,j]
        gradient[i] += contribution;gradient[j] -= contribution
    return float(values[1]),gradient


def single_integrator_cbf_rows(gradient_h, *, h, gain, partial_t_h=0.0,
                               tightening=0.0, shares=None):
    """Return a_i,b_i for sum(a_i.T u_i+b_i)<=0.

    Equivalent to dot h + gain*h >= tightening, for dot p_i=u_i.
    Adding zero-sum s_i to local left sides preserves the global inequality.
    Rows represent participating agents. When passing all agents for a sparse
    constraint, explicitly give nonparticipants zero shares.
    tightening has units [h]/s; it is not a spectral-value margin.
    """
    grad = np.asarray(gradient_h,dtype=float)
    if grad.ndim!=2 or grad.shape[0]<1 or not np.all(np.isfinite(grad)):
        raise ValueError("gradient_h must be a finite (N,d) array")
    if not all(np.isfinite(v) for v in (h,gain,partial_t_h,tightening)) or gain<=0 or tightening<0:
        raise ValueError("Require finite data, gain > 0 and tightening >= 0")
    n = len(grad)
    omega = np.full(n,1/n) if shares is None else np.asarray(shares,dtype=float)
    if omega.shape!=(n,) or not np.all(np.isfinite(omega)) or np.any(omega<0) or not np.isclose(omega.sum(),1,rtol=0,atol=1e-12):
        raise ValueError("Nonnegative shares must sum to one")
    return -grad,omega*(-partial_t_h-gain*h+tightening)


def row_capacity(a, limit, *, input_set):
    """max(-a.T u) for an origin-centred ball or box; never conflate their norms."""
    a = np.asarray(a,dtype=float)
    bound = np.asarray(limit,dtype=float)
    if a.ndim!=1 or not np.all(np.isfinite(a)) or not np.all(np.isfinite(bound)) or np.any(bound<0):
        raise ValueError("Finite vector a and nonnegative finite limits required")
    if input_set=="ball_l2" and bound.ndim==0:
        return float(bound*np.linalg.norm(a))
    if input_set=="box_linf" and (bound.ndim==0 or bound.shape==a.shape):
        return float(np.sum(bound*np.abs(a)))
    raise ValueError("Use ball_l2 with scalar radius or box_linf with scalar/vector limits")
