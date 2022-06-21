import pandas as pd
import shelve
import numpy as np
import matplotlib.pyplot as plt

db1 = shelve.open('output/population_experiment/data_d_in_100_1000_m_1_2')
d_large = db1['data']
db1.close()
# db2 = shelve.open('output/population_experiment/data_d_in_6_8_10_12')
db2 = shelve.open('output/population_experiment/data_d_in_4to20')
d_small = db2['data']
db2.close()

########################## Plot parameters #####################################
plt.rc('text', usetex=True)
plt.rc('text.latex', preamble=r'\usepackage{amsmath,amssymb}')
font = {'size': 14}
plt.rc('font', **font)

B_G_large = 50000  # Samples per. combination in the large-dimension study
B_G_small = 50000  # Samples per. combination in the small-dimension study
################################################################################

######## Plots for large graphs ########
colors = ["#F8766D", "#00BFC4"]
fig, ax = plt.subplots(1, 2, dpi = 300, sharex = 'all', sharey = 'all', figsize = (8.5, 4))
xval01 = d_large[(d_large['d'] == 100) & (d_large['MaxSize'] == 2)]['Nint'] / 100 * 100
xval02 = d_large[(d_large['d'] == 100) & (d_large['MaxSize'] == 1)]['Nint'] / 100 * 100
xval03 = d_large[(d_large['d'] == 100) & (d_large['MaxSize'] == 2)]['Nint'] / 100 * 100
yval01 = d_large[(d_large['d'] == 100) & (d_large['MaxSize'] == 2)]['lenICP']['mean']
yval02 = d_large[(d_large['d'] == 100) & (d_large['MaxSize'] == 1)]['lenAS']['mean']
yval03 = d_large[(d_large['d'] == 100) & (d_large['MaxSize'] == 2)]['lenAS']['mean']
yval01sd = d_large[(d_large['d'] == 100) & (d_large['MaxSize'] == 2)]['lenICP']['std'] / np.sqrt(B_G_large)
yval02sd = d_large[(d_large['d'] == 100) & (d_large['MaxSize'] == 1)]['lenAS']['std'] / np.sqrt(B_G_large)
yval03sd = d_large[(d_large['d'] == 100) & (d_large['MaxSize'] == 2)]['lenAS']['std'] / np.sqrt(B_G_large)
ax[0].plot(
    xval01,
    yval01,
    label = r'ICP',
    c = colors[0]
)
ax[0].fill_between(xval01, yval01 - 1.96 * yval01sd, yval01 + 1.96 * yval01sd, color = colors[0], alpha = 0.25)
ax[0].plot(
    xval02,
    yval02,
    label = r'IAS ($m = 1$)',
    c = colors[1]
)
ax[0].fill_between(xval02, yval02 - 1.96 * yval02sd, yval02 + 1.96 * yval02sd, color = colors[1], alpha = 0.25)
ax[0].plot(
    xval03,
    yval03,
    '--',
    label = r'IAS ($m = 2$)',
    c = colors[1]
)
ax[0].fill_between(xval03, yval03 - 1.96 * yval03sd, yval03 + 1.96 * yval03sd, color = colors[1], alpha = 0.25)

xval11 = d_large[(d_large['d'] == 1000) & (d_large['MaxSize'] == 1)]['Nint'] / 1000 * 100
xval12 = d_large[(d_large['d'] == 1000) & (d_large['MaxSize'] == 1)]['Nint'] / 1000 * 100
yval11 = d_large[(d_large['d'] == 1000) & (d_large['MaxSize'] == 1)]['lenICP']['mean']
yval12 = d_large[(d_large['d'] == 1000) & (d_large['MaxSize'] == 1)]['lenAS']['mean']
yval11sd = d_large[(d_large['d'] == 1000) & (d_large['MaxSize'] == 1)]['lenICP']['std'] / np.sqrt(B_G_large)
yval12sd = d_large[(d_large['d'] == 1000) & (d_large['MaxSize'] == 1)]['lenAS']['std'] / np.sqrt(B_G_large)
ax[1].plot(
    xval11,
    yval11,
    label = r'ICP',
    c = colors[0]
)
ax[1].fill_between(xval11, yval11 - 1.96 * yval11sd, yval11 + 1.96 * yval11sd, color = colors[0], alpha = 0.25)
ax[1].plot(
    xval12,
    yval12,
    label = r'IAS ($m = 1$)',
    c = colors[1]
)
ax[1].fill_between(xval12, yval12 - 1.96 * yval12sd, yval12 + 1.96 * yval12sd, color = colors[1], alpha = 0.25)
ax[1].plot(
    [],
    [],
    '--',
    label = r'IAS ($m = 2$)',
    c = colors[1]
)
#ax[0].legend()
#ax[1].legend()
#fig.legend(loc = 'lower center', ncol = 3, bbox_to_anchor = (0.5, 0.025))
ax[1].legend(loc = 'upper right', prop={'size': 11})
ax[0].set_title(r'$d = 100$')
ax[1].set_title(r'$d = 1{,}000$')
fig.supxlabel(r'Proportion of predictors intervened on (\%)')
fig.supylabel(r'Average size of oracle set')
fig.tight_layout()
#fig.subplots_adjust(top = 1.1, bottom = 0.1)
# fig.savefig('output/population_experiment/fig_large.svg')
fig.savefig('output/population_experiment/fig_large.pdf')


######## Plots for small graphs ########
# colors = ["#F8766D", "#A3A500", "#00BF7D", "#00B0F6", "#E76BF3"]
colors = ["#F8766D", "#D39200", "#93AA00", "#00BA38", "#00C19F", "#00B9E3", "#619CFF", "#DB72FB","#FF61C3"]
ds = [4, 6, 8, 10, 12, 14, 16, 18, 20]
fig, ax = plt.subplots(1, 2, dpi = 300, sharex = 'all', sharey = 'all', figsize = (8.5, 4.5))
for j, ax_ in enumerate(ax):
    p_ = 'sparse' if j == 0 else 'dense'
    for i, d in enumerate(ds):
        # N_g = 1000
        xval = d_small[(d_small['d'] == d) & (d_small['p'] == p_)]['Nint'] / d * 100
        yval = d_small[(d_small['d'] == d) & (d_small['p'] == p_)]['StrictlyLarger']['mean']
        y_sd = d_small[(d_small['d'] == d) & (d_small['p'] == p_)]['StrictlyLarger']['std'] / np.sqrt(B_G_small)
        ax_.plot(
            xval,
            yval,
            label = f'$d = {d}$',
            c = colors[i]
        )
        ax_.fill_between(
            xval,
            yval - 1.96 * y_sd,
            yval + 1.96 * y_sd,
            color = colors[i],
            alpha = 0.25
        )
ax[0].set_xticks([(i + 1) / 10 * 100 for i in range(0, 10, 2)])
ax[1].set_xticks([(i + 1) / 10 * 100 for i in range(0, 10, 2)])
ax[0].legend(loc = 'upper right', ncol = 2)
#ax[1].legend()
ax[0].set_title(r'Sparse graphs')
ax[1].set_title(r'Dense graphs')
fig.supxlabel(r'Proportion of predictors intervened on (\%)') #, labelpad = 10)
fig.supylabel(r'$\mathbb{P}_n(S_{\operatorname{ICP}} \subsetneq S_{\operatorname{IAS}}) $')
fig.tight_layout()
# fig.savefig('output/population_experiment/fig_small.svg')
fig.savefig('output/population_experiment/fig_small.pdf')