import json

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from SALib.analyze import fast, sobol
from SALib.sample import latin, saltelli

from utils import gen_problem

# https://waterprogramming.wordpress.com/2014/02/11/extensions-of-salib-for-more-complex-sensitivity-analyses/


params = dict(
    n1_filters={"bounds": [2, 64], "type": "int"},
    n2_filters={"bounds": [2, 64], "type": "int"},
    n3_filters={"bounds": [16, 64], "type": "int"},
    act_layers={"bounds": [0, 3], "type": "int"},
    latent_layer_div={"bounds": [0.5, 0.7], "type": "float"},
    latent_layer_size={"bounds": [120, 200], "type": "int"},
    epochs={"bounds": [15, 80], "type": "int"},
    batch={"bounds": [0, 32], "type": "int"},
)

# The Saltelli sampler generates N∗(2*D+2)
n_smp = 112  # multiple of 28
# n_smp = 28  # multiple of 28

# Generate the SALib dictionary
problem = gen_problem(params)

# Get the number of samples  to input at Saltelli function
n_stl = int(n_smp / (2 * problem["num_vars"] + 2))
# Generate the samples
smp_sbl = saltelli.sample(problem, n_stl, True)

# Analysis first run
df = pd.read_hdf('ae_add_salib_hist_2_old.h5')
df = df[~df['mse'].isin([np.nan, np.inf, -np.inf])]
# df['mse'][~np.isinf(df['mse'])]
df['1/mse'] = 1/df['mse']

# Si = sobol.analyze(problem, 1/df['loss'].values, print_to_console=True)
# Si = fast.analyze(problem, 1/df['mse'].values[:-8], print_to_console=True)

corr = df.corr(method='spearman')
plt.pcolormesh(df.corr(method='spearman'), vmax=1, vmin=-1)
plt.xticks(np.arange(0, df.shape[1] -1) + 0.5, df.columns[:-1], rotation=90)
plt.yticks(np.arange(0, df.shape[1] -1) + 0.5, df.columns[:-1])
plt.colorbar()

df.plot(x='mse', y='val_mse', kind='scatter')
plt.grid(True)

# act_fn = {0: "relu", 1: "tanh", 2: "sigmoid", 3: "elu", 4: "linear"}
# optm = {0: "adam", 1: "nadam", 2: "adamax"}

# Analysis
# Increase dropout = instabilities but in general it did not make differences
# Sometimes the dropout and l2_reg affect negatively, opting to remove it
df.boxplot(column='val_mse',by=['dropout', 'l2_reg'])
plt.xticks(rotation=90)

# Latent size - Layers [120 200] Div [0.5 0.7]
df.boxplot(column='1/mse',by=['latent_layer_size', 'latent_layer_div'])
plt.xticks(rotation=90)

# Small number of epochs for adamax -> epochs did not affected
df.boxplot(column='1/mse',by=['epochs', 'batch'])
plt.xticks(rotation=90)

df.boxplot(column='val_mse',by=['epochs', 'batch'])
plt.xticks(rotation=90)

# Expansion architecture 
df.boxplot(column='1/mse',by=['n1_filters', 'n3_filters'])
plt.xticks(rotation=90)

# Smaller batch < 32
df.boxplot(column='1/mse',by=['batch'])
plt.xticks(rotation=90)

# Crap
df[['1/mse']].plot.kde()
