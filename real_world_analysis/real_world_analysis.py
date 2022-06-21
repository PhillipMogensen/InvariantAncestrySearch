import pandas as pd
from real_world_analysis.real_world_search import is_true_positive, test_gene
import numpy as np
from scipy.stats.stats import pearsonr  
import matplotlib.pyplot as plt

## Plot-stuff
plt.rc('text', usetex=True)
plt.rc('text.latex', preamble=r'\usepackage{amsmath,amssymb}')
font = {'size': 14}
plt.rc('font', **font)


#### Loading of data ####
observation, intervention, intervention_pos = (np.genfromtxt(
    # f'data/study4_gene_expression/Kemmeren{spec}.csv', delimiter='\t', skip_header=1
    f'real_world_analysis/data/Kemmeren{spec}.csv', delimiter='\t', skip_header=1
  ) for spec in ['Obs', 'Int', 'IntPos']
)
# Fix indentation starting at 1
intervention_pos = (intervention_pos - 1).astype(int)

def random_guess_rate(true_pos_cut):
  return np.mean(intervention < np.quantile(observation, true_pos_cut, axis=0)) + \
         np.mean(intervention > np.quantile(observation, 1-true_pos_cut, axis=0))

# Load data
# results = pd.read_pickle(f"output/study4_gene_expression/saved_results.pkl")
results = pd.read_pickle(f"real_world_analysis/saved_results.pkl")

# Hyperparameters
true_pos_cut = 0.025
alpha_gene = .25
alpha_empty = 1e-12

# Transformations
df = results.copy()
df = df[df['gene_x'] >= 0]
df['true_positive'] = [is_true_positive(v.gene_x, v.gene_y, true_pos_cut) for v in df.itertuples()]
df['empty_p_val'] = [test_gene(None, v.gene_y) for v in df.itertuples()]
df['empty_p_modified'] = [test_gene(v.gene_x, v.gene_y, test_empty=True) for v in df.itertuples()]
df['predicted_positive'] = (df['p_val'] >= alpha_gene)
df['gene_cor'] = [pearsonr(observation[:,v.gene_x], observation[:,v.gene_y])[0] for v in df.itertuples()]

# Take subsets
subframe = df[df['empty_p_val'] < alpha_empty]
# subframe = subframe[subframe['empty_p_modified'] < alpha_empty]
subframe = subframe[subframe['p_val'] > alpha_gene]

# Print count of invariant empty sets
print((results['gene_x'] == -1).agg(["sum", "mean"]))
print(f"Total tests: {int(intervention_pos.size*(intervention.shape[1] - (results['gene_x'] == -1).sum()))}")

# Print true positive rates
cross = pd.crosstab(subframe['true_positive'], subframe['predicted_positive'])
tpr = (cross.loc[True]/cross.sum()).to_numpy()[0]
print(cross)
print(f"Random guess rate: {np.round(random_guess_rate(true_pos_cut), 3)}")
print(f"True positive rate: {np.round(tpr, 3)}")

# Plot different p-val cutoffs
tprs, found = [], []
p_vals = list(reversed([1e-12, 1e-13, 1e-14, 1e-15, 1e-16, 1e-17]))
for p_cutoff in p_vals:
  subframe = df[df['empty_p_modified'] < p_cutoff]
  subframe = subframe[subframe['p_val'] > alpha_gene]
  cross = pd.crosstab(subframe['true_positive'], subframe['predicted_positive'])
  tprs.append((cross.loc[True]/cross.sum()).to_numpy()[0])
  found.append(cross.sum())

# Plot changing p-cutoff
colors = ["#F8766D", "#A3A500", "#00BFC4"]
ticks = list(range(len(p_vals)))
fig, ax = plt.subplots(1, 1, dpi = 200, figsize=(6, 3))
ax = [ax]
line1 = ax[0].plot(ticks, tprs, label = r'TPR', c = colors[0])
ax[0].set_xticks(ticks)
ax[0].set_xticklabels(p_vals)
ax[0].set_ylim(bottom=0)
# ax[0].set_ylabel("True positive rate")

# On same x-axis, plot number of found 
ax_found = ax[0].twinx()
line2 = ax_found.plot(ticks, found, label=r'Number of pairs found', c=colors[2],)
ax_found.set_ylim(bottom=0)

ax_found.set_ylabel('Number of pairs found', color=colors[2])  # we already handled the x-label with ax1
ax_found.tick_params(axis='y', labelcolor=colors[2])

ax[0].set_xlabel(r'$\alpha_0$')
ax[0].set_ylabel('True positive rate')

lines = line1 + line2
ax[0].legend(lines, [l.get_label() for l in lines], loc = 4, prop={'size': 11})


# fig.supylabel(r'True positive rate')
# fig.savefig('output/study4_gene_expression/fig_genes.svg')
fig.savefig('output/real_world_analysis/fig_genes.pdf')
# plt.show()


fig, ax = plt.subplots()

# Plot different true_positive_cutoffs
tprs, random_guess = [], []
tp_cuts = [0.01, 0.025, 0.05, 0.1]

ticks = list(range(len(tp_cuts)))
for tp_cut in tp_cuts:
  subframe = df[df['empty_p_modified'] < alpha_empty]
  subframe = subframe[subframe['p_val'] > alpha_gene]
  subframe['true_positive'] = [is_true_positive(v.gene_x, v.gene_y, tp_cut) for v in subframe.itertuples()]
  cross = pd.crosstab(subframe['true_positive'], subframe['predicted_positive'])
  tprs.append((cross.loc[True]/cross.sum()).to_numpy()[0])
  random_guess.append(random_guess_rate(tp_cut))

ax.plot(ticks, tprs, c = colors[0], label=r'TPR (IAS)')
ax.plot(ticks, random_guess, c = colors[1], label=r'TPR (Random guess)')
ax.set_xticks(ticks)
ax.set_xticklabels(tp_cuts)
ax.set_ylim(bottom=0)

ax.legend(loc = 'lower right', prop={'size': 14})
ax.set_title(r'Varying $q_{TP}$')
ax.set_xlabel(r'$q_{TP}$')
ax.set_ylabel(r'True positive rate')
# fig.savefig('output/study4_gene_expression/fig_TPRs.svg')
fig.savefig('output/real_world_analysis/fig_TPRs.pdf')
# plt.show()