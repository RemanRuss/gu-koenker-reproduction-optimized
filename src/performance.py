"""Reproducible profiling, complexity, benchmark and stability diagnostics."""
import argparse
import cProfile
import io
import json
import pstats
import time
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .config import GRID_POINTS,MAX_ITER,SEED,SCENARIOS,H,REPS,N
from . import model, model_baseline

OUT=Path('results/performance')

def inputs(n,scenario='two_point',seed=SEED):
    rng=np.random.default_rng(seed)
    y,_=model_baseline.generate(n=n,h=H,scenario=scenario,rng=rng)
    grid=np.linspace(min(-2.,float(y.min())-.2),max(2.,float(y.max())+.2),GRID_POINTS)
    return y,grid

def timed(fn,y,grid,repeats=3):
    vals=[]
    for _ in range(repeats):
        t=time.perf_counter()
        fn(y,grid,max_iter=MAX_ITER,tol=1e-7)
        vals.append(time.perf_counter()-t)
    return float(np.median(vals))

def profile():
    OUT.mkdir(parents=True,exist_ok=True)
    y,grid=inputs(N)
    for label,fn in [('baseline',model_baseline.fit_kw),('optimized',model.fit_kw)]:
        p=cProfile.Profile();p.enable()
        fn(y,grid,max_iter=MAX_ITER,tol=1e-7)
        p.disable();p.dump_stats(str(OUT/f'{label}.prof'))
        stream=io.StringIO()
        pstats.Stats(p,stream=stream).sort_stats('cumulative').print_stats(18)
        (OUT/f'profile_{label}.txt').write_text(stream.getvalue(),encoding='utf-8')
    print('Saved profiles under results/performance/')

def complexity():
    OUT.mkdir(parents=True,exist_ok=True)
    records=[]
    for n in (100,200,400,800):
        for scenario in SCENARIOS:
            y,grid=inputs(n,scenario)
            for label,fn in [('baseline',model_baseline.fit_kw),('optimized',model.fit_kw)]:
                sec=timed(fn,y,grid)
                records.append({'n':n,'scenario':scenario,'version':label,'seconds':sec})
                print(n,scenario,label,f'{sec:.5f}s')
    df=pd.DataFrame(records);df.to_csv(OUT/'complexity.csv',index=False)
    fig,ax=plt.subplots(figsize=(8,5))
    lines=[]
    for scenario in SCENARIOS:
        for label in ('baseline','optimized'):
            d=df[(df.scenario==scenario)&(df.version==label)].sort_values('n')
            slope=float(np.polyfit(np.log(d.n),np.log(d.seconds),1)[0]);lines.append({'scenario':scenario,'version':label,'loglog_slope':slope})
            ax.loglog(d.n,d.seconds,'o-',label=f'{scenario} {label} (slope={slope:.2f})')
    ax.set(xlabel='Sample size n (log scale)',ylabel='EM fit runtime (seconds, log scale)',title='Empirical complexity: baseline vs optimized')
    ax.legend(fontsize=8);fig.tight_layout();fig.savefig(OUT/'complexity.png',dpi=180);plt.close(fig)
    pd.DataFrame(lines).to_csv(OUT/'complexity_slopes.csv',index=False)

def benchmark():
    OUT.mkdir(parents=True,exist_ok=True)
    records=[]
    for n in (100,400,800):
        for scenario in SCENARIOS:
            y,grid=inputs(n,scenario)
            base=timed(model_baseline.fit_kw,y,grid)
            opt=timed(model.fit_kw,y,grid)
            records.append({'n':n,'scenario':scenario,'baseline_s':base,'optimized_s':opt,'speedup':base/opt})
    df=pd.DataFrame(records);df.to_csv(OUT/'benchmark.csv',index=False)
    fig,ax=plt.subplots(figsize=(8,4.5))
    x=np.arange(len(df));w=.38
    ax.bar(x-w/2,df.baseline_s,w,label='Baseline')
    ax.bar(x+w/2,df.optimized_s,w,label='Optimized')
    ax.set_xticks(x,[f'{r.scenario}\nn={r.n}' for r in df.itertuples()])
    ax.set(ylabel='Median EM fit time (seconds)',title='Matched-input runtime comparison')
    ax.legend();fig.tight_layout();fig.savefig(OUT/'benchmark.png',dpi=180);plt.close(fig)
    print(df.to_string(index=False))

def stability():
    OUT.mkdir(parents=True,exist_ok=True)
    rows=[]
    for n in (30,100,400):
        for scenario in SCENARIOS:
            for seed in (SEED,SEED+1,SEED+2):
                y,grid=inputs(n,scenario,seed)
                for label,fn in [('baseline',model_baseline.fit_kw),('optimized',model.fit_kw)]:
                    with warnings.catch_warnings(record=True) as w:
                        warnings.simplefilter('always')
                        err='';converged=False;iters=0;valid=False
                        try:
                            _,weights,iters=fn(y,grid,max_iter=MAX_ITER,tol=1e-7)
                            valid=bool(np.all(np.isfinite(weights)) and np.all(weights>=0) and np.isclose(weights.sum(),1))
                            converged=iters<MAX_ITER
                        except Exception as e:
                            err=f'{type(e).__name__}: {e}'
                        rows.append({'n':n,'scenario':scenario,'seed':seed,'version':label,'warnings':len(w),'warning_messages':' | '.join(str(z.message) for z in w),
                                     'exception':err,'iterations':iters,'converged_before_cap':converged,'valid_weights':valid})
    df=pd.DataFrame(rows);df.to_csv(OUT/'stability.csv',index=False)
    print(df.groupby(['version','scenario'])[['warnings','converged_before_cap','valid_weights']].agg(['sum','count']).to_string())

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('task',choices=['profile','complexity','benchmark','stability-check'])
    args=p.parse_args()
    {'profile':profile,'complexity':complexity,'benchmark':benchmark,'stability-check':stability}[args.task]()
