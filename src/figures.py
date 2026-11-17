from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .model import density,posterior_mean,oracle
from .config import H

def run():
    out=Path('results/figures');out.mkdir(parents=True,exist_ok=True)
    diag=pd.read_csv('results/raw/diagnostic.csv')
    for scen in ('two_point','uniform'):
        d=diag[diag.scenario==scen]
        w=pd.read_csv(f'results/raw/weights_{scen}.csv');grid=w.grid.to_numpy();weights=w.weight.to_numpy()
        xx=np.linspace(-3.5,4.0,500)
        fig,axes=plt.subplots(1,3,figsize=(13.5,4.0))
        axes[0].plot(grid,weights,'o',ms=2.5,label='Estimated mixing masses')
        if scen=='two_point':
            axes[0].vlines([-.5,1],[0,0],[2/3,1/3],color='tab:orange',lw=3,label='True mass probabilities')
        else:
            axes[0].axvspan(-.5,1,alpha=.15,color='tab:orange',label='True uniform support')
        axes[0].set_title('Mixing distribution');axes[0].set_ylabel('Grid mass (not density)');axes[0].legend(fontsize=7)
        axes[1].hist(d.y,bins=28,density=True,alpha=.3,label='Simulated observations')
        axes[1].plot(xx,density(xx,grid,weights),label='Estimated marginal density',lw=2)
        if scen=='two_point':true_density=(2/3)*np.exp(-.5*(xx+.5)**2)/np.sqrt(2*np.pi)+(1/3)*np.exp(-.5*(xx-1)**2)/np.sqrt(2*np.pi)
        else:
            from scipy.stats import norm
            true_density=(norm.cdf(xx+.5)-norm.cdf(xx-1))/1.5
        axes[1].plot(xx,true_density,'--',label='True marginal density')
        axes[1].set_title('Observed Gaussian mixture');axes[1].legend(fontsize=7)
        axes[2].plot(xx,xx,color='gray',ls=':',label='Naive')
        axes[2].plot(xx,posterior_mean(xx,grid,weights),lw=2,label='Empirical Bayes')
        axes[2].plot(xx,oracle(xx,scen,H),'--',label='Oracle Bayes')
        axes[2].set_title('Shrinkage / Bayes rules');axes[2].set_xlabel('Observed y');axes[2].set_ylabel('Estimated alpha');axes[2].legend(fontsize=7)
        fig.suptitle(f'Gu & Koenker Section 3.4.1 — {scen.replace("_"," ").title()} (illustrative replication)')
        fig.tight_layout();fig.savefig(out/f'figure_{scen}.png',dpi=190);plt.close(fig)
    res=pd.read_csv('results/raw/summary.csv')
    fig,ax=plt.subplots(figsize=(8,4.5))
    names=['naive','empirical_bayes','oracle_bayes'];x=np.arange(2);width=.25
    for i,method in enumerate(names):
        a=res[res.method==method].set_index('scenario').loc[['two_point','uniform']]
        ax.bar(x+(i-1)*width,a.mean_mse,width,yerr=1.96*a.mcse_mse,capsize=3,label=method.replace('_',' ').title())
    ax.set_xticks(x,['Two-point mixture','Uniform mixture']);ax.set_ylabel('Mean squared error (lower is better)');ax.set_title('Repeated simulation: individual-effect estimation error');ax.legend();fig.tight_layout();fig.savefig(out/'mse_comparison.png',dpi=200);plt.close(fig)
    print('Saved three figures to results/figures')
if __name__=='__main__':run()
