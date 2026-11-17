from pathlib import Path
import pandas as pd
import numpy as np

def run():
    df=pd.read_csv('results/raw/simulation.csv')
    groups=df.groupby(['scenario','method'],sort=True)
    summary=groups.agg(reps=('mse','size'),mean_mse=('mse','mean'),sd_mse=('mse','std'),mean_mae=('mae','mean'),mean_bias=('bias','mean')).reset_index()
    summary['mcse_mse']=summary['sd_mse']/np.sqrt(summary['reps'])
    summary['mcse_mse']=summary['mcse_mse'].fillna(0.)
    summary.to_csv('results/raw/summary.csv',index=False)
    print(summary.to_string(index=False,float_format=lambda x:f'{x:.4f}'))
if __name__=='__main__':run()
