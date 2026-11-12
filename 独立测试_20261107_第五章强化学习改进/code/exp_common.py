"""Shared pieces of this folder's evaluation (README §3, §4).

ExtFlow     = the thesis deployment nominal (20261103_终段包络验证 EnvelopedResidualFlow: gated residual x completion
              envelope rho_i), unchanged in every computation; it only records extra per-sample diagnostics
              (intervention, nominal size, raw action saturation, lateral components) into the run_pa trace.
O2Policy    = idea 6 of 独立测试.md: inference-time O(2) symmetrisation of an SO(2)-equivariant actor,
              omega_sym = (omega(x) - omega(Px)) / 2,  rho_sym = (rho(x) + rho(Px)) / 2,  psi_sym = (psi(x) + psi(Px)) / 2,
              P negates the pseudoscalar (reflection-odd) features. Indices verified by test_o2.py.
mirror_case = reflect positions and goals through the x axis (y -> -y).
"""
import paths  # noqa: F401
import numpy as np
import sarl_env as E

import importlib.util as _ilu
_spec = _ilu.spec_from_file_location('eval_env', str(paths.C5 / '独立验证' / '20261103_终段包络验证' / 'code' / 'eval_env.py'))
EE = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(EE)                  # frozen envelope flow (imports sarl_env / evaluate from SARL, unchanged)

# reflection-odd columns of the geometric features (sarl_env.ResidualFlow.observe); every algebraic column is even
PSEUDO_NODE = [5, 8]            # v.J e_hat, delta.J e_hat
PSEUDO_EDGE = [3, 5, 8, 10]     # r.J e_hat_i, r.J e_hat_j, (J e_hat_i).e_hat_j, (v_j - v_i).J r
U = E.U


class RecPolicy:
    """Pass-through wrapper that keeps the last raw (unclipped) actor output for diagnostics."""
    def __init__(self, base):
        self.base = base
        self.last = None

    def __call__(self, obs):
        an, ae = self.base(obs)
        self.last = (an, ae)
        return an, ae


class O2Policy:
    """Inference-time O(2) symmetrisation (README §3.3). base: obs -> (node (n,2), edge (E,))."""
    def __init__(self, base):
        self.base = base

    def __call__(self, obs):
        an1, ae1 = self.base(obs)
        ob2 = dict(obs)
        x = obs['x'].copy(); x[:, PSEUDO_NODE] *= -1
        ea = obs['ea'].copy(); ea[:, PSEUDO_EDGE] *= -1
        ob2['x'], ob2['ea'] = x, ea
        an2, ae2 = self.base(ob2)
        an = np.column_stack([0.5 * (an1[:, 0] - an2[:, 0]), 0.5 * (an1[:, 1] + an2[:, 1])])
        return an, 0.5 * (ae1 + ae2)


class ExtFlow(EE.EnvelopedResidualFlow):
    """Thesis deployment nominal with extra recorded diagnostics. use_env=False drops the envelope (plain
    ResidualFlow.compose). No decision of the controller depends on the recorded quantities."""
    use_env = True

    def compose(self, p, geo, gate, node_a, edge_a):
        if self.use_env:
            nominal, res = super().compose(p, geo, gate, node_a, edge_a)
        else:
            nominal, res = E.ResidualFlow.compose(self, p, geo, gate, node_a, edge_a)
        self._nom, self._geo = nominal, geo
        return nominal, res

    def control(self, p):
        u, info = super().control(p)
        nom, geo = self._nom, self._geo
        e, ne = geo['e'], geo['ne']
        nomc = E.clip_ball(nom)
        dl = u - nomc
        eh = e / np.maximum(ne, 1e-9)[:, None]
        Jeh = np.stack([-eh[:, 1], eh[:, 0]], 1)
        mov = ne > 0.05
        pol = type(self).policy
        ext = dict(interv_mean=float(np.linalg.norm(dl, axis=1).mean()),
                   interv_max=float(np.linalg.norm(dl, axis=1).max()),
                   nom_mean=float(np.linalg.norm(nom, axis=1).mean()),
                   nomc_mean=float(np.linalg.norm(nomc, axis=1).mean()),
                   n_moving=int(mov.sum()))
        if mov.any():
            ext.update(lat_nom_moving=float(np.abs(np.sum(nom[mov] * Jeh[mov], 1)).mean()),
                       lat_u_moving=float(np.abs(np.sum(u[mov] * Jeh[mov], 1)).mean()))
        last = getattr(pol, 'last', None)
        if last is not None:
            an, ae = last
            ext.update(sat_w=float(np.mean(np.abs(an[:, 0]) >= 1.0)), sat_r=float(np.mean(np.abs(an[:, 1]) >= 1.0)),
                       sat_psi=float(np.mean(np.abs(ae) >= 1.0)) if len(ae) else 0.0,
                       absw_mean=float(np.abs(np.clip(an[:, 0], -1, 1)).mean()),
                       w_mean=float(np.clip(an[:, 0], -1, 1).mean()))
        info.update(ext)
        return u, info


def mirror_case(case):
    c = dict(case)
    M = np.array([1.0, -1.0])
    c['p'] = np.asarray(case['p'], float) * M
    c['goal'] = np.asarray(case['goal'], float) * M
    c['name'] = case['name'] + '_mir'
    return c


def smoothness(npz, a_lim=0.8):
    """Command-jump proxy from the executed trajectory: per robot and sample, |v_k - v_{k-1}| / dt_k with
    v_k = (p_{k+1} - p_k) / dt_k (constant over a certified segment). a_lim = 0.8 m/s^2 is the Chapter-6 chassis
    limit, used only as an indicative threshold (Chapter 6 runs at 10 Hz on a second-order plant)."""
    z = np.load(npz)
    st, ts = z['states'], np.array(z['times'])
    if len(st) < 3:
        return dict(acc_mean=None, acc_p99=None, acc_frac_over=None)
    dt = np.diff(ts)
    v = np.diff(st, axis=0) / dt[:, None, None]
    acc = np.linalg.norm(np.diff(v, axis=0), axis=2) / dt[1:, None]
    return dict(acc_mean=float(acc.mean()), acc_p99=float(np.quantile(acc, .99)),
                acc_frac_over=float(np.mean(acc > a_lim)), acc_max=float(acc.max()))
