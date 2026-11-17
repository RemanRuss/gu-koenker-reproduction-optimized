"""Gaussian location mixture; optimized finite-grid Kiefer-Wolfowitz EM.

Optimizations: cache a row-scaled normal-density matrix once, then use BLAS
matrix-vector products instead of repeatedly forming large log-responsibility
matrices and calling logsumexp for each EM iteration.
"""
import numpy as np
from scipy.special import logsumexp
from scipy.stats import norm
from .model_baseline import generate, oracle, density

def fit_kw(y, grid=None, max_iter=500, tol=1e-8):
    y=np.asarray(y,dtype=float)
    if y.ndim != 1 or y.size == 0 or not np.isfinite(y).all():
        raise ValueError('y must be a nonempty finite 1D array')
    if grid is None:
        grid=np.linspace(float(y.min())-.5,float(y.max())+.5,161)
    grid=np.asarray(grid,dtype=float)
    if grid.ndim != 1 or grid.size == 0 or not np.isfinite(grid).all():
        raise ValueError('grid must be a nonempty finite 1D array')
    if max_iter < 1: raise ValueError('max_iter must be positive')
    # The normalizing constants cancel in responsibilities. Row maxima ensure
    # that even very large |y-grid| values cannot underflow every component.
    z=-.5*(y[:,None]-grid[None,:])**2
    rowmax=z.max(axis=1)
    kernel=np.ascontiguousarray(np.exp(z-rowmax[:,None]))
    weights=np.full(grid.size,1.0/grid.size)
    llprev=-np.inf
    const=-.5*np.log(2*np.pi)
    for iteration in range(max_iter):
        den=kernel@weights
        # Since each row has at least one element exactly 1 and all initial
        # weights are positive, den should be positive. Guard extreme cases.
        if not np.all(np.isfinite(den)) or np.any(den<=0):
            raise FloatingPointError('nonfinite or zero mixture density')
        ll=float(np.sum(np.log(den)+rowmax+const))
        # EM update without materializing an n x m responsibility array.
        weights *= (kernel.T@(1.0/den))/y.size
        # Avoid accumulating small floating point drift from repeated sums.
        weights /= weights.sum()
        if abs(ll-llprev)<tol: break
        llprev=ll
    return grid,weights,iteration+1

def posterior_mean(y,grid,weights):
    y=np.asarray(y,dtype=float)
    grid=np.asarray(grid,dtype=float)
    weights=np.asarray(weights,dtype=float)
    if np.any(weights<0) or not np.isfinite(weights).all() or weights.sum()<=0:
        raise ValueError('invalid mixture weights')
    z=-.5*(y[...,None]-grid)**2
    rowmax=np.max(z,axis=-1,keepdims=True)
    k=np.exp(z-rowmax)
    den=k@weights
    if np.any(den<=0):
        # Original logsumexp implementation is robust for a highly degenerate
        # fitted distribution with all mass outside the numerical support.
        logp=norm.logpdf(y[...,None]-grid)+np.log(np.maximum(weights,1e-300))
        post=np.exp(logp-logsumexp(logp,axis=-1,keepdims=True))
        return np.sum(post*grid,axis=-1)
    return (k@(weights*grid))/den
