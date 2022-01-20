import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import shelve
from os import listdir
from os.path import isfile, join

########################## Plot parameters #####################################
plt.rc('text', usetex=True)
plt.rc('text.latex', preamble=r'\usepackage{amsmath,amssymb}')
#font = {'size': 26}
font = {'size': 12}
plt.rc('font', **font)
################################################################################

def MakeBoxplot(ax, file, yvariable, boxplot = True, jitter = True, lines = False):
    database = shelve.open(file)
    data = database['data']
    d = 6 if '6' in file else 100 # 8 if '8' in file else len(database['generator_objects'][0].g.nodes()) - 2
    
    N = data['N'].unique().tolist()
    positions_seq = [i + 1 for i in range(len(N))]
    if boxplot:
        ax.boxplot(
            [data[data['N'] == n][yvariable].tolist() for n in N],
            positions = positions_seq,
            showfliers = (not jitter and not lines),
            medianprops = dict(linewidth=2.5)
        )
    if jitter and not lines:
        xdata = ((data['N'] == 100) * 1 + (data['N'] == 1000) * 2 + (data['N'] == 10000) * 3 + (data['N'] == 100000) * 4) + np.random.normal(0, 0.05, data.shape[0])
        ax.scatter(xdata, data[yvariable], color = 'black', alpha = 0.1)
    
    if lines:
        data['i'] = np.repeat(range(int(data.shape[0] / len(N))), len(data['N'].unique()))
        for i in range(int(data.shape[0] / len(N))):
            jit = np.array(positions_seq) + np.random.normal(0, 0.05, len(N)) if jitter and lines else positions_seq
            ax.plot(
                jit,
                data[data['i'] == i][yvariable],
                alpha = 0.05,
                color = 'black'
            )
            ax.scatter(
                jit,
                data[data['i'] == i][yvariable],
                alpha = 0.05,
                color = 'black',
                s = 10
            )
    if 'alpha0' in file:
        if len(N) == 4:
            ax.set_xticks(positions_seq, [r'$10^2$', r'$10^3$', r'$10^4$', r'$10^5$'])
        else:
            ax.set_xticks(positions_seq, [r'$10^2$', r'$10^3$', r'$10^4$'])
    else:
        ax.set_xticks(positions_seq, [r'$10^3$', r'$10^4$', r'$10^4$', r'$10^5$'])
    database.close()

    return

############### Boxplots ###############
#### P(level) and P(empty) ####
## alpha0 = 10e-6
fig, (ax1, ax2) = plt.subplots(2, 2, sharex = 'all', sharey = 'all')
relevant_files = ['output/finite_experiment/d_equal_6/alpha0_equal_10Eminus6', 'output/finite_experiment/d_equal_100/alpha0_equal_10Eminus6']
relevant_d = [6, 100]
for i, f in enumerate(relevant_files):
    MakeBoxplot(ax1[i], f, 'Plevel', lines = True, jitter = True)
    MakeBoxplot(ax2[i], f, 'Pempty', lines = True, jitter = True)
    ax1[i].set_title(f'$d = {relevant_d[i]}$')
    ax2[i].set_xlabel(r'Number of observations')
ax1[0].set_ylabel(r'$\mathbb{P}_n(\hat{S}_{\operatorname{IAS}}   \subseteq \operatorname{AN}_Y)$')
ax1[1].set_ylabel(r'$\mathbb{P}_n(\hat{S}_{\operatorname{IAS}}^1 \subseteq \operatorname{AN}_Y)$')
ax2[0].set_ylabel(r'$\mathbb{P}_n(\hat{S}_{\operatorname{IAS}}   = \emptyset)$')
ax2[1].set_ylabel(r'$\mathbb{P}_n(\hat{S}_{\operatorname{IAS}}^1 = \emptyset)$')
fig.tight_layout()
fig.savefig('output/finite_experiment/figures/boxplot_appendix_minus6.svg')


## alpha0 = 10e-12
fig, (ax1, ax2) = plt.subplots(2, 2, sharex = 'all', sharey = 'all')
relevant_files = ['output/finite_experiment/d_equal_6/alpha0_equal_10Eminus12', 'output/finite_experiment/d_equal_100/alpha0_equal_10Eminus12']
relevant_d = [6, 100]
for i, f in enumerate(relevant_files):
    MakeBoxplot(ax1[i], f, 'Plevel', lines = True, jitter = True)
    MakeBoxplot(ax2[i], f, 'Pempty', lines = True, jitter = True)
    ax1[i].set_title(f'$d = {relevant_d[i]}$')
    ax2[i].set_xlabel(r'Number of observations')
ax1[0].set_ylabel(r'$\mathbb{P}_n(\hat{S}_{\operatorname{IAS}}   \subseteq \operatorname{AN}_Y)$')
ax1[1].set_ylabel(r'$\mathbb{P}_n(\hat{S}_{\operatorname{IAS}}^1 \subseteq \operatorname{AN}_Y)$')
ax2[0].set_ylabel(r'$\mathbb{P}_n(\hat{S}_{\operatorname{IAS}}   = \emptyset)$')
ax2[1].set_ylabel(r'$\mathbb{P}_n(\hat{S}_{\operatorname{IAS}}^1 = \emptyset)$')
fig.tight_layout()
fig.savefig('output/finite_experiment/figures/boxplot_appendix_minus12.svg')

## alpha0 = alpha
fig, (ax1, ax2) = plt.subplots(2, 2, sharex = 'all', sharey = 'all')
relevant_files = ['output/finite_experiment/d_equal_6/alpha0_equal_None', 'output/finite_experiment/d_equal_100/alpha0_equal_None']
relevant_d = [6, 100]
for i, f in enumerate(relevant_files):
    MakeBoxplot(ax1[i], f, 'Plevel', lines = True, jitter = True)
    MakeBoxplot(ax2[i], f, 'Pempty', lines = True, jitter = True)
    ax1[i].set_title(f'$d = {relevant_d[i]}$')
    ax2[i].set_xlabel(r'Number of observations')
ax1[0].set_ylabel(r'$\mathbb{P}_n(\hat{S}_{\operatorname{IAS}}   \subseteq \operatorname{AN}_Y)$')
ax1[1].set_ylabel(r'$\mathbb{P}_n(\hat{S}_{\operatorname{IAS}}^1 \subseteq \operatorname{AN}_Y)$')
ax2[0].set_ylabel(r'$\mathbb{P}_n(\hat{S}_{\operatorname{IAS}}   = \emptyset)$')
ax2[1].set_ylabel(r'$\mathbb{P}_n(\hat{S}_{\operatorname{IAS}}^1 = \emptyset)$')
fig.tight_layout()
fig.savefig('output/finite_experiment/figures/boxplot_appendix_None.svg')
