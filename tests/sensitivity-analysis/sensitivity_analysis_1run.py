import json

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from SALib.sample import latin, saltelli
from SALib.analyze import fast
from SALib.analyze import sobol

from utils import gen_problem, proper_type

# https://waterprogramming.wordpress.com/2014/02/11/extensions-of-salib-for-more-complex-sensitivity-analyses/


params = dict(
    n1_filters={"bounds": [2, 128], "type": "int"},
    n2_filters={"bounds": [2, 128], "type": "int"},
    n3_filters={"bounds": [0, 128], "type": "int"},
    act_layers={"bounds": [0, 3], "type": "int"},
    act_latent_layers={"bounds": [0, 4], "type": "int"},
    latent_layer_div={"bounds": [0, 1], "type": "float"},
    latent_layer_size={"bounds": [10, 300], "type": "int"},
    dropout={"bounds": [0, 3], "type": "float"},
    l2_reg={"bounds": [-8, -3], "type": "float"},
    learning_rate={"bounds": [-3, 0], "type": "float"},
    optm={"bounds": [0, 3], "type": "int"},
    epochs={"bounds": [15, 300], "type": "int"},
    batch={"bounds": [0, 64], "type": "int"},
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

smp_st = proper_type(smp_sbl, params)


# # Analysis first run
# df = pd.read_hdf('ae_add_salib_hist.h5')
# df.head()
# # df['mse'][~np.isinf(df['mse'])]
# df['mse'][np.isinf(df['mse'])] = 3
# df['1/mse'] = 1/df['mse']

# # Si = sobol.analyze(problem, 1/df['loss'].values, print_to_console=True)
# # Si = fast.analyze(problem, 1/df['mse'].values[:-8], print_to_console=True)

# corr = df.corr(method='spearman')
# plt.pcolormesh(df.corr(method='spearman'), vmax=1, vmin=-1)
# plt.xticks(np.arange(0, df.shape[1] -1) + 0.5, df.columns[:-1], rotation=90)
# plt.yticks(np.arange(0, df.shape[1] -1) + 0.5, df.columns[:-1])
# plt.colorbar()

# # act_fn = {0: "relu", 1: "tanh", 2: "sigmoid", 3: "elu", 4: "linear"}
# # optm = {0: "adam", 1: "nadam", 2: "adamax"}

# # Analysis
# # Increase dropout = instabilities but in general it did not make differences
# # Sometimes the dropout and l2_reg affect negatively, opting to remove it
# df.boxplot(column='val_mse',by=['dropout', 'l2_reg'])
# plt.xticks(rotation=90)

# # Latent size - Layers [120 200] Div [0.5 0.7]
# df.boxplot(column='1/mse',by=['latent_layer_size', 'latent_layer_div'])
# plt.xticks(rotation=90)

# # Small number of epochs for adamax -> epochs did not affected
# df.boxplot(column='1/mse',by=['epochs', 'optm'])
# plt.xticks(rotation=90)

# # Small epochs and learning rate = bad
# df.boxplot(column='1/mse',by=['epochs', 'learning_rate',])
# plt.xticks(rotation=90)

# # Learning rate did not affected the optimization algorithm
# df.boxplot(column='1/mse',by=['optm', 'learning_rate',])
# plt.xticks(rotation=90)

# # Expansion architecture 
# df.boxplot(column='1/mse',by=['n3_filters', 'n1_filters'])
# plt.xticks(rotation=90)

# # Crap
# df[['1/mse']].plot.kde()
