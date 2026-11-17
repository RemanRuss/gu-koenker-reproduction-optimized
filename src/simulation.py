"""Reproducible simulation with optional parallel independent replicates."""
import argparse
import logging
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
import pandas as pd
from .config import N,H,SEED,REPS,SCENARIOS,GRID_POINTS,MAX_ITER
from .model import generate,fit_kw,posterior_mean,oracle

LOG=logging.getLogger(__name__)

def one_replicate(task):
    scenario,r,n,seed=task
    rng=np.random.default_rng(seed)
    y,alpha=generate(n=n,h=H,scenario=scenario,rng=rng)
    grid=np.linspace(min(-2.,float(y.min())-.2),max(2.,float(y.max())+.2),GRID_POINTS)
    grid,w,iters=fit_kw(y,grid,max_iter=MAX_ITER,tol=1e-7)
    eb=posterior_mean(y,grid,w)
    truth=oracle(y,scenario,H)
    rows=[]
    for method,pred in [('naive',y),('empirical_bayes',eb),('oracle_bayes',truth)]:
        rows.append({'scenario':scenario,'replicate':r,'n':n,'method':method,
                     'mse':float(np.mean((pred-alpha)**2)),
                     'bias':float(np.mean(pred-alpha)),
                     'mae':float(np.mean(np.abs(pred-alpha)))})
    diag=pd.DataFrame({'scenario':scenario,'y':y,'alpha':alpha,'eb':eb,'oracle':truth}) if r==0 else None
    return rows,diag,grid,w,iters,scenario,r

def run(reps=REPS,seed=SEED,n=N,workers=1):
    if reps<1 or n<1 or workers<1: raise ValueError('reps, n and workers must be positive')
    seedseq=np.random.SeedSequence(seed)
    streams=seedseq.spawn(len(SCENARIOS)*reps)
    tasks=[(scen,r,n,streams[i*reps+r]) for i,scen in enumerate(SCENARIOS) for r in range(reps)]
    if workers==1:
        outputs=map(one_replicate,tasks)
        outputs=list(outputs)
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            outputs=list(pool.map(one_replicate,tasks))
    results=[]; diagnostics=[]
    out=Path('results/raw');out.mkdir(parents=True,exist_ok=True)
    for rows,diag,grid,w,iters,scenario,r in outputs:
        results.extend(rows)
        if r==0:
            diagnostics.append(diag)
            pd.DataFrame({'grid':grid,'weight':w,'scenario':scenario}).to_csv(out/f'weights_{scenario}.csv',index=False)
            LOG.info('%s first run finished after %d EM iterations',scenario,iters)
    pd.DataFrame(results).to_csv(out/'simulation.csv',index=False)
    pd.concat(diagnostics,ignore_index=True).to_csv(out/'diagnostic.csv',index=False)
    LOG.info('Completed %d scenarios x %d replicates with %d workers',len(SCENARIOS),reps,workers)

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--reps',type=int,default=REPS)
    parser.add_argument('--seed',type=int,default=SEED)
    parser.add_argument('--n',type=int,default=N)
    parser.add_argument('--workers',type=int,default=1)
    args=parser.parse_args()
    logging.basicConfig(level=logging.INFO,format='%(levelname)s %(message)s')
    run(args.reps,args.seed,args.n,args.workers)
