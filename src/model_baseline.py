"""Original Unit 2 numerical routines, preserved as a reference for regression tests."""
import numpy as np
from scipy.special import logsumexp
from scipy.stats import norm

def generate(n=400,h=.5,scenario='two_point',rng=None):
    rng=np.random.default_rng() if rng is None else rng
    if scenario=='two_point':
        alpha=np.where(rng.random(n)<2/3,-h,2*h)
    elif scenario=='uniform':
        alpha=rng.uniform(-h,2*h,n)
    else: raise ValueError(scenario)
    y=alpha+rng.normal(size=n)
    return y,alpha

def fit_kw(y, grid=None, max_iter=500, tol=1e-8):
    y=np.asarray(y,dtype=float)
    if grid is None: grid=np.linspace(float(np.min(y))-0.5,float(np.max(y))+0.5,161)
    grid=np.asarray(grid,dtype=float)
    log_kernel=norm.logpdf(y[:,None]-grid[None,:])
    weights=np.full(len(grid),1/len(grid));llprev=-np.inf
    for iteration in range(max_iter):
        lw=log_kernel+np.log(np.maximum(weights,1e-300))[None,:]
        den=logsumexp(lw,axis=1)
        weights=np.exp(lw-den[:,None]).mean(axis=0)
        ll=float(den.sum())
        if abs(ll-llprev)<tol:break
        llprev=ll
    return grid,weights,iteration+1

def posterior_mean(y,grid,weights):
    y=np.asarray(y)
    logp=norm.logpdf(y[...,None]-grid)+np.log(np.maximum(weights,1e-300))
    post=np.exp(logp-logsumexp(logp,axis=-1,keepdims=True))
    return np.sum(post*grid,axis=-1)

def oracle(y,scenario,h=.5):
    if scenario=='two_point':
        grid=np.array([-h,2*h]);w=np.array([2/3,1/3]);return posterior_mean(y,grid,w)
    grid=np.linspace(-h,2*h,1201);w=np.full(grid.size,1/grid.size)
    return posterior_mean(y,grid,w)

def density(x,grid,w):
    return np.exp(logsumexp(norm.logpdf(np.asarray(x)[:,None]-grid)+np.log(np.maximum(w,1e-300)),axis=1))
