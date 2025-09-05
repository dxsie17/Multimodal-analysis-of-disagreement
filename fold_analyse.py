import pandas as pd, numpy as np
df = pd.read_csv('disagreement-dataset/metadata.csv')
fold = [c for c in df.columns if c.lower() in ('fold','cv','kfold')][0]
def show(col):
    if col in df.columns:
        print(f'\n[{col}] by fold:')

        if df[col].dtype=='O':
            print(df.groupby(fold)[col].value_counts(normalize=True).unstack(fill_value=0.0).round(3))
        else:
            print(df.groupby(fold)[col].agg(['count','mean','std','min','median','max']).round(3))
for c in ['Item','Gender','Duration','R Choice','C Choice','Consensus']:
    show(c)