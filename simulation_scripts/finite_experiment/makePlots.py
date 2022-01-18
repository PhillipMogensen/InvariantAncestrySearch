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

bins = 10
################################################################################

filenames = [
    join('output/finite_experiment/', f) for f in listdir('output/finite_experiment/') if isfile(join('output/finite_experiment/', f)) and '.svg' not in f
]

def MakeHistogram(ax, file, yvariable):
    database = shelve.open(file)
    data = database['data']
    d = 8 if '8' in file else len(database['generator_objects'][0].g.nodes()) - 2
    
    N = [1000, 10000, 100000]
    for i in range(3):
        ax[i].hist(
            data[data['N'] == N[i]][yvariable],
            bins = bins
        )
        titlestr = '$d = ' + str(d) + '$, ' '$N = ' + '{:,}'.format(N[i]) + '$'
        ax[i].set_title(titlestr)
    database.close()

    return

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
            showfliers = (not jitter and not lines)
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
                color = 'black'
            )
            # jit = [(np.array([1, 2, 3]) + np.random.normal(0, 0.05, 3)).tolist() for i in range(int(data.shape[0] / 3))]
            # yvar = [data[data['i'] == i][yvariable].tolist() for i in range(int(data.shape[0] / 3))]
            # ax.plot (
            #     jit,
            #     yvar,
            #     alpha = 0.05,
            #     color = 'black'
            # )
            # ax.plot(jit, yvar, alpha = 0.05, color = 'black')
    if 'alpha0' in file:
        if len(N) == 4:
            ax.set_xticks(positions_seq, [r'$10^2$', r'$10^3$', r'$10^4$', r'$10^5$'])
        else:
            ax.set_xticks(positions_seq, [r'$10^2$', r'$10^3$', r'$10^4$'])
    else:
        ax.set_xticks(positions_seq, [r'$10^3$', r'$10^4$', r'$10^4$', r'$10^5$'])
    database.close()

    return

############### Boxplot ###############
######## Main text ########
#### P(level) and P(empty) ####
fig, (ax1, ax2) = plt.subplots(2, 2, sharex = 'all', sharey = 'all')
#relevant_files = ['output/finite_experiment/d_eq_6_alpha0eq10eminus12', 'output/finite_experiment/d_eq_100_alpha0eq10eminus12']
relevant_files = ['output/finite_experiment/d_eq_6_alpha0eq10eminus6', 'output/finite_experiment/d_eq_100_alpha0eq10eminus6']
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
fig.savefig('output/finite_experiment/boxplot_main_minus6.svg')

######## Appendix text ########
#### P(level) and P(empty) ####
fig, (ax1, ax2) = plt.subplots(2, 2, sharex = 'all', sharey = 'all')
relevant_files = ['output/finite_experiment/data_d_eq_6_noStandardization', 'output/finite_experiment/data_d_eq_100_noStandardization']
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
fig.savefig('output/finite_experiment/boxplot_appendix.svg')

#### Tables w percentages for appendix ####
db1    = shelve.open('output/finite_experiment/data_d_eq_6')
db1_ns = shelve.open('output/finite_experiment/data_d_eq_6_noStandardization')
db2    = shelve.open('output/finite_experiment/data_d_eq_100')
db2_ns = shelve.open('output/finite_experiment/data_d_eq_100_noStandardization')
d1    = db1['data']
d1_ns = db1_ns['data']
d2    = db2['data']
d2_ns = db2_ns['data']
db1.close() ; db1_ns.close() ; db2.close() ; db2_ns.close()
for dt in [d1, d2, d1_ns, d2_ns]:
    dt['Plevel'].transform(lambda x: x >= 0.95).groupby(dt['N']).agg('mean')

############### Histograms (not used) ###############
######### Main text plots ###########
#### P(level) plot for main text ####
# fig = plt.figure(constrained_layout=True, figsize = (15, 15))
# subfigs = fig.subfigures(nrows=3, ncols=1)
# subtitles = [f'$d = {d}$' for d in [6, 8, 100]]
# fnames = [f'output/finite_experiment/data_d_eq_{d}' for d in [6, 8, 100]]
# for row, subfig in enumerate(subfigs):
#     subtitle = subtitles[row]
#     # subfig.suptitle(subtitle)


#     # create 1x3 subplots per subfig
#     axs = subfig.subplots(nrows=1, ncols=3, sharex = 'all', sharey = 'all')
#     f = fnames[row]
#     MakeHistogram(axs, f, 'Plevel')
#     xlabel = '$\mathbb{P}_n(\hat{S}_{\operatorname{IAS}} \subseteq \operatorname{AN}_Y)$' if row <= 1 else '$\mathbb{P}_n(\hat{S}_{\operatorname{IAS}}^1 \subseteq \operatorname{AN}_Y)$'
#     subfig.supxlabel(xlabel)

# fig.supylabel('Frequency')
# fig.savefig('output/finite_experiment/fig.svg')

# #### P(empty) plot for main text ####
# fig = plt.figure(constrained_layout=True, figsize = (15, 15))
# subfigs = fig.subfigures(nrows=3, ncols=1)
# subtitles = [f'$d = {d}$' for d in [6, 8, 100]]
# fnames = [f'output/finite_experiment/data_d_eq_{d}' for d in [6, 8, 100]]
# for row, subfig in enumerate(subfigs):
#     subtitle = subtitles[row]
#     # subfig.suptitle(subtitle)


#     # create 1x3 subplots per subfig
#     axs = subfig.subplots(nrows=1, ncols=3, sharex = 'all', sharey = 'all')
#     f = fnames[row]
#     MakeHistogram(axs, f, 'Pempty')
#     xlabel = '$\mathbb{P}_n(\hat{S}_{\operatorname{IAS}} = \emptyset)$' if row <= 1 else '$\mathbb{P}_n(\hat{S}_{\operatorname{IAS}}^1 = \emptyset)$'
#     subfig.supxlabel(xlabel)

# fig.supylabel('Frequency')
# fig.savefig('output/finite_experiment/fig_Pempty.svg')
# ############################################

# ######### Appendix plots ###########
# #### P(level) plot for appendix ####
# fig = plt.figure(constrained_layout=True, figsize = (15, 15))
# subfigs = fig.subfigures(nrows=3, ncols=1)
# subtitles = [f'$d = {d}$' for d in [6, 8, 100]]
# fnames = [f'output/finite_experiment/data_d_eq_{d}_noStandardization' for d in [6, 8, 100]]
# for row, subfig in enumerate(subfigs):
#     subtitle = subtitles[row]
#     # subfig.suptitle(subtitle)


#     # create 1x3 subplots per subfig
#     axs = subfig.subplots(nrows=1, ncols=3, sharex = 'all', sharey = 'all')
#     f = fnames[row]
#     MakeHistogram(axs, f, 'Plevel')
#     xlabel = '$\mathbb{P}_n(\hat{S}_{\operatorname{IAS}} \subseteq \operatorname{AN}_Y)$' if row <= 1 else '$\mathbb{P}_n(\hat{S}_{\operatorname{IAS}}^1 \subseteq \operatorname{AN}_Y)$'
#     subfig.supxlabel(xlabel)

# fig.supylabel('Frequency')
# fig.savefig('output/finite_experiment/fig_noStandardization.svg')

# #### P(empty) plot for appendix ####
# fig = plt.figure(constrained_layout=True, figsize = (15, 15))
# subfigs = fig.subfigures(nrows=3, ncols=1)
# subtitles = [f'$d = {d}$' for d in [6, 8, 100]]
# fnames = [f'output/finite_experiment/data_d_eq_{d}' for d in [6, 8, 100]]
# for row, subfig in enumerate(subfigs):
#     subtitle = subtitles[row]
#     # subfig.suptitle(subtitle)


#     # create 1x3 subplots per subfig
#     axs = subfig.subplots(nrows=1, ncols=3, sharex = 'all', sharey = 'all')
#     f = fnames[row]
#     MakeHistogram(axs, f, 'Pempty')
#     xlabel = '$\mathbb{P}_n(\hat{S}_{\operatorname{IAS}} = \emptyset)$' if row <= 1 else '$\mathbb{P}_n(\hat{S}_{\operatorname{IAS}}^1 = \emptyset)$'
#     subfig.supxlabel(xlabel)

# fig.supylabel('Frequency')
# fig.savefig('output/finite_experiment/fig_Pempty_noStandardization.svg')
# ###########################################