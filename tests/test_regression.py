import numpy as np
import pytest
from src import model,model_baseline
from src.simulation import one_replicate

@pytest.mark.parametrize('scenario',['two_point','uniform'])
@pytest.mark.parametrize('n',[30,100,400])
def test_em_fitted_results(scenario,n):
    rng=np.random.default_rng(1234)
    y,_=model.generate(n=n,h=.5,scenario=scenario,rng=rng)
    grid=np.linspace(-4,4,121)
    _,w0,_=model_baseline.fit_kw(y,grid,max_iter=350,tol=1e-7)
    _,w1,_=model.fit_kw(y,grid,max_iter=350,tol=1e-7)
    assert np.allclose(w0,w1,atol=2e-5,rtol=2e-5)
    assert np.isclose(w1.sum(),1,atol=1e-12)
    assert np.all(w1>=0)
    assert np.allclose(model_baseline.posterior_mean(y,grid,w0),model.posterior_mean(y,grid,w1),atol=2e-5,rtol=2e-5)

@pytest.mark.parametrize('scenario',['two_point','uniform'])
def test_estimation_and_metrics(scenario):
    y,a=model.generate(n=100,scenario=scenario,rng=np.random.default_rng(77))
    grid=np.linspace(-4,4,121)
    w=np.ones(121)/121
    assert np.allclose(model.posterior_mean(y,grid,w),model_baseline.posterior_mean(y,grid,w),atol=1e-12)
    assert np.allclose(model.oracle(y,scenario),model_baseline.oracle(y,scenario),atol=1e-12)
    out=one_replicate((scenario,0,100,np.random.SeedSequence(77)))
    assert len(out[0])==3
    assert all(np.isfinite(row['mse']) for row in out[0])

def test_extreme_values():
    y=np.array([-1000.,-40.,0.,40.,1000.]);grid=np.linspace(-2,2,61)
    _,w,_=model.fit_kw(y,grid,max_iter=20)
    assert np.isfinite(w).all() and np.isclose(w.sum(),1)
    assert np.isfinite(model.posterior_mean(y,grid,w)).all()

def test_invalid_input():
    with pytest.raises(ValueError):model.fit_kw(np.array([np.nan]))
    with pytest.raises(ValueError):model.fit_kw(np.array([]))
