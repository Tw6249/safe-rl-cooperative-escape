"""Shared-parameter graph policy and centralised critic (README §4). Inputs are rotation-invariant scalars; the
outputs are coefficients of the equivariant vectors J e_i, e_i and (p_i - p_j)/d_ij used in sarl_env.ResidualFlow.

Variants: 'full' actor reads the geometric and the algebraic features (shares, reserves, local duals, edge ledger,
Fiedler/subspace edge terms); 'geom' and 'static' actors read only the geometric features. The critic always reads
everything (centralised training, decentralised execution)."""
import math
import numpy as np
import torch
import torch.nn as nn

from sarl_env import NODE_GEOM, EDGE_GEOM, NODE_ALG, EDGE_ALG

H = 64


def mlp(i, o, last_zero=False):
    m = nn.Sequential(nn.Linear(i, H), nn.SiLU(), nn.Linear(H, H), nn.SiLU(), nn.Linear(H, o))
    if last_zero:
        nn.init.zeros_(m[-1].weight)
        nn.init.zeros_(m[-1].bias)
    return m


def scatter_mean_max(m, idx, n):
    s = torch.zeros(n, m.shape[1], device=m.device, dtype=m.dtype).index_add_(0, idx, m)
    c = torch.zeros(n, device=m.device, dtype=m.dtype).index_add_(0, idx, torch.ones_like(idx, dtype=m.dtype))
    mean = s / c.clamp(min=1.0)[:, None]
    mx = torch.full((n, m.shape[1]), -1e9, device=m.device, dtype=m.dtype)
    mx = mx.scatter_reduce(0, idx[:, None].expand_as(m), m, reduce='amax', include_self=True)
    mx = torch.where(c[:, None] > 0, mx, torch.zeros_like(mx))
    return mean, mx


class GNN(nn.Module):
    def __init__(self, fn, fe, layers=2):
        super().__init__()
        self.enc = mlp(fn, H)
        self.msg = nn.ModuleList([mlp(2 * H + fe, H) for _ in range(layers)])
        self.upd = nn.ModuleList([mlp(3 * H, H) for _ in range(layers)])
        self.norm = nn.ModuleList([nn.LayerNorm(H) for _ in range(layers)])

    def forward(self, x, ei, ea):
        n = x.shape[0]
        h = self.enc(x)
        src, dst = ei[0], ei[1]
        for msg, upd, norm in zip(self.msg, self.upd, self.norm):
            m = msg(torch.cat([h[src], h[dst], ea], 1))
            mean, mx = scatter_mean_max(m, src, n)
            h = norm(h + upd(torch.cat([h, mean, mx], 1)))
        return h


ALG_VARIANTS = ('full', 'full_int')          # actors that read the algebraic features ('full_int': added 2026-11-02)


def actor_dims(variant):
    return (NODE_GEOM + NODE_ALG, EDGE_GEOM + EDGE_ALG) if variant in ALG_VARIANTS else (NODE_GEOM, EDGE_GEOM)


def actor_inputs(obs, variant):
    if variant in ALG_VARIANTS:
        return np.concatenate([obs['x'], obs['xa']], 1), np.concatenate([obs['ea'], obs['eaa']], 1)
    return obs['x'], obs['ea']


class Actor(nn.Module):
    def __init__(self, variant, std0=0.4):
        super().__init__()
        fn, fe = actor_dims(variant)
        self.variant = variant
        self.gnn = GNN(fn, fe)
        self.node_head = mlp(H, 2, last_zero=True)
        self.edge_head = mlp(2 * H + fe, 1, last_zero=True)
        self.log_std_node = nn.Parameter(torch.full((2,), math.log(std0)))
        self.log_std_edge = nn.Parameter(torch.full((1,), math.log(std0)))

    def forward(self, x, ei, ea):
        h = self.gnn(x, ei, ea)
        mu_n = self.node_head(h)
        mu_e = self.edge_head(torch.cat([h[ei[0]], h[ei[1]], ea], 1)).squeeze(-1)
        return mu_n, mu_e

    def stds(self):
        return self.log_std_node.clamp(-3.0, 0.5).exp(), self.log_std_edge.clamp(-3.0, 0.5).exp()

    def log_prob(self, x, ei, ea, an, ae):
        """Per-robot log-probability: own node action + all outgoing edge actions."""
        mu_n, mu_e = self(x, ei, ea)
        sn, se = self.stds()
        lpn = torch.distributions.Normal(mu_n, sn).log_prob(an).sum(-1)
        lpe = torch.distributions.Normal(mu_e, se[0]).log_prob(ae)
        return lpn.index_add(0, ei[0], lpe)


class Critic(nn.Module):
    def __init__(self):
        super().__init__()
        fn, fe = NODE_GEOM + NODE_ALG, EDGE_GEOM + EDGE_ALG
        self.gnn = GNN(fn, fe)
        self.head = nn.Sequential(nn.Linear(2 * H + 4, H), nn.SiLU(), nn.Linear(H, H), nn.SiLU(), nn.Linear(H, 1))

    def forward(self, x, ei, ea, batch, glob, n_graphs):
        h = self.gnn(x, ei, ea)
        mean, mx = scatter_mean_max(h, batch, n_graphs)
        return self.head(torch.cat([mean, mx, glob], 1)).squeeze(-1)


class NumpyPolicy:
    """CPU wrapper used inside a closed loop: obs -> (node action, edge action); records log-probabilities."""

    def __init__(self, actor, deterministic, rng=None):
        self.actor, self.det, self.rng = actor, deterministic, rng
        self.record = []

    @torch.no_grad()
    def __call__(self, obs):
        xn, xe = actor_inputs(obs, self.actor.variant)
        x, ea = torch.from_numpy(xn), torch.from_numpy(xe)
        ei = torch.from_numpy(obs['ei'])
        mu_n, mu_e = self.actor(x, ei, ea)
        mu_n, mu_e = mu_n.numpy().astype(np.float64), mu_e.numpy().astype(np.float64)
        if self.det:
            an, ae = mu_n, mu_e
        else:
            sn, se = (t.numpy().astype(np.float64) for t in self.actor.stds())
            an = mu_n + sn * self.rng.standard_normal(mu_n.shape)
            ae = mu_e + se[0] * self.rng.standard_normal(mu_e.shape)
            lp = self.actor.log_prob(x, ei, ea, torch.from_numpy(an.astype(np.float32)),
                                     torch.from_numpy(ae.astype(np.float32))).numpy()
            self.record.append(dict(an=an.astype(np.float32), ae=ae.astype(np.float32), logp=lp))
        return an, ae
