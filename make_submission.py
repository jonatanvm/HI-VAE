import sys

import pandas as pd

name = sys.argv[1]
res = pd.read_csv('kaggle_empty.csv')
df = pd.read_csv(f'Results/{name}/{name}_data_reconstruction.csv', header=None)
res['hospital_death'] = df[0]
res = res.astype(int, errors='ignore')

res.to_csv(f'kaggle/{name}.csv', index=None)