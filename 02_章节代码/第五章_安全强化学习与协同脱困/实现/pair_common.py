"""Minimal shared infrastructure for pair circulation and its one baseline.

First-order planar dynamics; central full-matrix spectral CBF. No planner,
reference chain, ratchet, nonlinear potential family or distributed claim.
"""
from pathlib import Path
from dataclasses import dataclass
import json,time
import numpy as np
import cvxpy as cp
from scipy.linalg import null_space
from safety_checks import (pair_indices,laplacian_from_weights,weights_from_squared_distance,
                           affine_segment_bounds,exact_affine_segment_certificate)

U=.5-1e-7

def dump(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    payload=json.dumps(data,ensure_ascii=False,indent=2,
        default=lambda v:v.tolist() if isinstance(v,np.ndarray) else v.item()
        if isinstance(v,np.generic) else str(v))
    tmp=path.with_suffix(path.suffix+'.tmp')
    for retry in range(30):
        try:tmp.write_text(payload,encoding='utf8');tmp.replace(path);return
        except PermissionError:
            if retry==29:raise
            time.sleep(.1)

def cases():
    data=json.loads((Path(__file__).parent/'scenarios.json').read_text(encoding='utf8'))
    return [dict(name=c['name'],p=np.array(c['p']),goal=np.array(c['goal']),lam=c['lam']) for c in data['cases']]

@dataclass(frozen=True)
class Parameters:
    radius:float=10.
    clearance:float=1.
    lambda_required:float=2.5
    speed:float=U
    step:float=.05
    alpha_collision:float=2.
    alpha_spectral:float=2.

def frozen_matrices(p,par,qbasis):
    n=len(p);i,j=pair_indices(n);rel=p[i]-p[j];d2=np.sum(rel*rel,axis=1)
    coll=np.zeros((len(i),n,2));dl=np.zeros((n-1,n-1,n,2))
    grad=-4/par.radius**2*np.maximum(0,1-d2/par.radius**2)[:,None]*rel
    for z,(a,b) in enumerate(zip(i,j)):
        coll[z,a]=2*rel[z];coll[z,b]=-2*rel[z]
        v=qbasis[a]-qbasis[b];mat=np.outer(v,v)
        dl[:,:,a,:]+=mat[:,:,None]*grad[z];dl[:,:,b,:]-=mat[:,:,None]*grad[z]
    lap=laplacian_from_weights(n,weights_from_squared_distance(d2,par.radius))
    h=qbasis.T@lap@qbasis-par.lambda_required*np.eye(n-1)
    return coll.reshape(len(i),2*n),par.alpha_collision*(d2-par.clearance**2),dl.reshape((n-1)**2,2*n),par.alpha_spectral*h

class CentralController:
    def __init__(self,n,par):
        self.n=n;self.par=par;self.qbasis=null_space(np.ones((1,n)))
        self.u=cp.Variable((n,2));self.nominal=cp.Parameter((n,2))
        self.coll=cp.Parameter((n*(n-1)//2,2*n));self.rhs=cp.Parameter(n*(n-1)//2)
        self.dl=cp.Parameter(((n-1)**2,2*n));self.h=cp.Parameter((n-1,n-1),symmetric=True)
        flat=cp.reshape(self.u,(2*n,),order='C')
        self.constraints=[self.coll@flat+self.rhs>=0,cp.norm(self.u,axis=1)<=par.speed,
                          cp.reshape(self.dl@flat,(n-1,n-1),order='C')+self.h>>0]
        self.problem=cp.Problem(cp.Minimize(.5*cp.sum_squares(self.u-self.nominal)),self.constraints)

class Oracle:
    def __init__(self,n,lam,dt):
        self.c=CentralController(n,Parameters(lambda_required=lam,step=dt))
        self.pos=cp.Parameter((n,2))
        self.problem=cp.Problem(self.c.problem.objective,self.c.constraints+
                               [self.c.u<=2*(4-self.pos),self.c.u>=-2*(4+self.pos)])
        self.count=0;self.inaccurate=0;self.max_residual=0.
    def solve(self,p,v):
        c=self.c;A,b,D,H=frozen_matrices(p,c.par,c.qbasis)
        c.coll.value=A;c.rhs.value=b;c.dl.value=D;c.h.value=(H+H.T)/2
        c.nominal.value=v;self.pos.value=p
        self.problem.solve(solver='CLARABEL',tol_feas=1e-10,tol_gap_abs=1e-10,tol_gap_rel=1e-10,max_iter=200)
        self.count+=1
        if self.problem.status not in ('optimal','optimal_inaccurate') or c.u.value is None:raise RuntimeError(self.problem.status)
        residual=max(float(np.max(x.violation())) for x in self.problem.constraints)
        self.max_residual=max(self.max_residual,residual)
        if self.problem.status=='optimal_inaccurate':self.inaccurate+=1
        if not np.isfinite(residual) or residual>1e-8:raise RuntimeError('primal_residual_'+str(residual))
        return c.u.value.copy()

def field(p,ids,kind,sign):
    """Only the retained affine COMPARISON baseline, not the main controller."""
    assert kind=='affine'
    x=p[ids]-p[ids].mean(axis=0)
    v=.6*x+.8*sign*np.column_stack((-x[:,1],x[:,0]))
    return U*v/max(np.linalg.norm(v,axis=1).max(),1e-12)

class Supervisor:
    """One retained baseline: mixed affine reference with fixed chirality."""
    recovery_threshold=.02
    def __init__(self,case,method,dt):
        assert method=='mixed_fixed_two_stage'
        self.goal=case['goal'];self.oracle=Oracle(len(self.goal),case['lam'],dt)
        self.mode='G';self.B=None;self.low=0.;self.sign=1
        self.events=[];self.episode=0;self.last_dt=0.
    def control(self,p,t):
        e=p-self.goal;W=float(.5*np.sum(e*e));base=self.oracle.solve(p,-e)
        D=float(-np.sum(e*base));rho=D/max(W,1e-30)
        goal=np.linalg.norm(e,axis=1).max()<=.01
        self.low=self.low+self.last_dt if rho<=.001 and not goal else 0.
        def event(name,**kw):self.events.append(dict(t=t,event=name,episode=self.episode,W=W,B=self.B,rho=rho,**kw))
        if self.mode=='E' and rho>=self.recovery_threshold:
            event('recover');self.mode='G';self.B=None;self.low=0.
        if self.mode=='G' and self.low>=.15-1e-10:
            self.episode+=1;self.B=W
            x=p-p.mean(axis=0);torque=float(np.sum(x[:,0]*(-e[:,1])-x[:,1]*(-e[:,0])))
            if self.episode==1:self.sign=1 if abs(torque)<1e-10 else int(np.sign(torque))
            event('begin',sign=self.sign);self.mode='E';self.low=0.
        ref=-e
        if self.mode=='E' and not goal:
            ref=field(p,list(range(len(p))),'affine',self.sign)-e;u=self.oracle.solve(p,ref)
        else:u=base
        return u,ref,dict(mode=self.mode,B=self.B,episode=self.episode,W=W,D=D,rho=rho,
            projection_residual=float(np.sum(e*base)+np.sum(base*base)),filter_change=float(np.linalg.norm(u-ref)))

def certified_step(p,u,lam,dt):
    from fractions import Fraction as F
    cuts=0
    while cuts<=20:
        z=p+dt*u;b=affine_segment_bounds(p,z,10.)
        if b['min_distance']>=1. and b['lambda2_lower']>=lam and np.abs(z).max()<=4.:
            cert=exact_affine_segment_certificate(p,z,10.,1.,lam)
            if cert['certified']:
                limit=(F(str(.5))*F(str(dt)))**2
                input_ok=all(sum((F(str(z[i,j]))-F(str(p[i,j])))**2 for j in range(2))<=limit for i in range(len(p)))
                if input_ok:return z,dt,cuts,b
        cuts+=1;dt*=.5
    raise RuntimeError('no certified positive step above nominal_dt/2**20')
