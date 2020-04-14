from SALib.sample import latin
from SALib.sample import saltelli
import numpy as np
import json
import matplotlib.pyplot as plt
# import seaborn as sns

# sns.set()

# $10 ^ 3 < Ra < 10 ^ 7$ and $10 ^ {-1} < Pr < 70$
# $0 °C < T_{hot} < 90 °C$ and the $1°C <$ $\Delta T$ $< 10°C$

problem = {
    'num_vars': 4,
    'names': ['Ra', 'Pr', 'T_cold', 'dT'],
    'bounds': [[3, 6],
               [np.log10(0.1), np.log10(70)],
               [10, 100],
               [1, 10], ]
}

seed = 1
# The Saltelli sampler generates N∗(2*D+2)
n_smp = 30               # multiple of 10
# Initial was 30

# Get the number of samples  to input at Saltelli function
n_stl = int(n_smp/(2*problem['num_vars'] + 2))
# Generate the samples
smp_sbl = saltelli.sample(problem, n_stl, True, seed=seed)

# Graph X
x_sbl = smp_sbl[:, 0]
# Plot graphs
kw = {'marker': 'o', 'alpha': 0.7, 'ls': ''}
plt.plot(x_sbl, smp_sbl[:, 1], **kw)
plt.plot(x_sbl, smp_sbl[:, 2], **kw)
plt.plot(x_sbl, smp_sbl[:, 3], **kw)
plt.xlabel(problem['names'][0])
plt.legend(problem['names'][1::])
plt.title('Samples generated using Sobol QMC')

# Generate JSON
smp_sbl[:, 0:2] = 10**(smp_sbl[:, 0:2])
smp_sbl[:, 2] += 273.15
smp_sbl[:, 3] = smp_sbl[:, 2] - smp_sbl[:, 3]

# Save JSON
f = open('args.json', 'w')
tst = smp_sbl.tolist()
json.dump(smp_sbl.tolist(), f)
f.close()