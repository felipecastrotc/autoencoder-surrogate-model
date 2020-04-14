from SALib.sample import latin
from SALib.sample import saltelli
import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns

sns.set()

problem = {
    'num_vars': 4,
    'names': ['x1', 'x2', 'x3', 'x4'],
    'bounds': [[0, 1],
               [0, 1],
               [0, 1],
               [0, 1], ]
}

seed = 2
n_smp = 6

smp_sbl = saltelli.sample(problem, n_smp, False, seed=seed)
smp_ltn = latin.sample(problem, smp_sbl.shape[0], seed=seed)

x_sbl = smp_sbl[:, 0]
x_ltn = smp_ltn[:, 0]

kw = {'marker': 'o', 'alpha': 0.7, 'ls': ''}
plt.plot(x_ltn, smp_ltn[:, 1], **kw)
plt.plot(x_ltn, smp_ltn[:, 2], **kw)
plt.plot(x_ltn, smp_ltn[:, 3], **kw)

plt.plot(x_sbl, smp_sbl[:, 1], **kw)
plt.plot(x_sbl, smp_sbl[:, 2], **kw)
plt.plot(x_sbl, smp_sbl[:, 3], **kw)

x = np.random.rand(smp_sbl.shape[0], 1)
y1 = np.random.rand(smp_sbl.shape[0], 1)
y2 = np.random.rand(smp_sbl.shape[0], 1)
y3 = np.random.rand(smp_sbl.shape[0], 1)
plt.plot(x, y1, **kw)
plt.plot(x, y2, **kw)
plt.plot(x, y3, **kw)
