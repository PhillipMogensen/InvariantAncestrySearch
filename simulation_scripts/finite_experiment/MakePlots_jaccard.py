from types import new_class
import pandas as pd
import shelve
from InvariantAncestrySearch.IASfunctions import *
import matplotlib.pyplot as plt
import seaborn as sns


########################## Plot parameters #####################################
plt.rc('text', usetex=True)
plt.rc('text.latex', preamble=r'\usepackage{amsmath,amssymb}')
#font = {'size': 26}
font = {'size': 12}
plt.rc('font', **font)
################################################################################

def CompareOutput(i, objs):
    MIP = get_MIP_dagitty(objs[i].g)
    rMIP = [a for a in MIP if len(a) <= 1]
    S_AS = set.union(*rMIP) if len(rMIP) > 0 else set()
    S_ICP = getICP(objs[i].g)
    
    return len(S_AS) - len(S_ICP)

############################################################ Main text ############################################################
db1 = shelve.open('output/finite_experiment/d_equal_6/alpha0_equal_10Eminus6')
#db1 = shelve.open('output/finite_experiment/d_equal_6/alpha0_equal_None')
#db1 = shelve.open('output/finite_experiment/d_equal_6/alpha0_equal_10Eminus12')
data = db1['data']
objs = db1['objlist']
db1.close()
db2 = shelve.open('output/finite_experiment/d_equal_100/alpha0_equal_10Eminus6')
#db2 = shelve.open('output/finite_experiment/d_equal_100/alpha0_equal_None')
#db2 = shelve.open('output/finite_experiment/d_equal_100/alpha0_equal_10Eminus12')
data2 = db2['data']
objs2 = db2['objlist']
db2.close()

data['d'] = 6
data2['d'] = 100
indices_d6   = np.where([CompareOutput(i, objs) != 0 for i in range(len(objs))])[0]
indices_d100 = np.where([CompareOutput(i, objs2) != 0 for i in range(len(objs2))])[0]
data['group']  = data['i'].transform(lambda x: r'different' if x in indices_d6 else r'equal')
data2['group'] = data2['i'].transform(lambda x: r'different' if x in indices_d100 else r'equal')

plotdata = pd.concat(
    [
        data[['N', 'J_AS', 'J_ICP', 'd', 'group']],
        data2[['N', 'J_AS', 'J_ICP', 'd', 'group']]
    ]
).melt(id_vars = ['N', 'd', 'group']).sort_values(by = 'group')


palette = {"J_AS": "#F8766D",
           "J_ICP": "#00BFC4"}
g = sns.FacetGrid(plotdata, col = 'group', row = 'd', margin_titles=True, legend_out=False)
g.map_dataframe(sns.boxplot, x = 'N', y = 'value', hue = 'variable', hue_order = ['J_AS', 'J_ICP'], palette = palette, fliersize=1.5)
for ax in g.axes[0]:
    ax.set_xticklabels([r'$10^2$', r'$10^3$', r'$10^4$', r'$10^5$'])
g.set_titles(col_template = r"Oracle IAS and ICP {col_name}", row_template = r"$d = {row_name}$")
g.set_axis_labels(x_var = r'Number of observations', y_var = r'Jaccard similarity to $\operatorname{AN}_Y$')
g.add_legend(prop={'size': 10}, loc = 'upper left')
new_labels = [r'$\hat{S}_{\operatorname{IAS}}$', r'$\hat{S}_{\operatorname{ICP}}^{\operatorname{MB}}$']
for t, l in zip(g._legend.texts, new_labels):
   t.set_text(l)
for ax in g.axes.flat:
    # Only the 2nd and 4th axes have something in ax.texts
    if ax.texts:
        # This contains the right ylabel text
        txt = ax.texts[0]
        ax.text(txt.get_unitless_position()[0], txt.get_unitless_position()[1],
                txt.get_text(),
                transform=ax.transAxes,
                va='center',
                rotation=0)
        # Remove the original text
        ax.texts[0].remove()
g.savefig('output/finite_experiment/figures/jaccard_fig_main.svg', dpi = 600)


#################### Graphs per window ####################
data.groupby(['N', 'group'])['group'].agg('count')
data2.groupby(['N', 'group'])['group'].agg('count')
###########################################################
##########################################################################################################################################


############################################################ Standard correction ############################################################
db1 = shelve.open('output/finite_experiment/d_equal_6/alpha0_equal_10Eminus6_StandardCorrection')
data = db1['data']
objs = db1['objlist']
db1.close()

data['d'] = 6
indices_d6   = np.where([CompareOutput(i, objs) != 0 for i in range(len(objs))])[0]
data['group']  = data['i'].transform(lambda x: r'different' if x in indices_d6 else r'equal')

plotdata = data[['N', 'J_AS', 'J_ICP', 'd', 'group']].melt(id_vars = ['N', 'd', 'group']).sort_values(by = 'group')


palette = {"J_AS": "#F8766D",
           "J_ICP": "#00BFC4"}
g = sns.FacetGrid(plotdata, col = 'group', legend_out=False)
g.map_dataframe(sns.boxplot, x = 'N', y = 'value', hue = 'variable', hue_order = ['J_AS', 'J_ICP'], palette = palette, fliersize=1.5)
for ax in g.axes[0]:
    ax.set_xticklabels([r'$10^2$', r'$10^3$', r'$10^4$', r'$10^5$'])
g.set_titles(col_template = r"Oracle IAS and ICP {col_name}", row_template = r"$d = {row_name}$")
g.set_axis_labels(x_var = r'Number of observations', y_var = r'Jaccard similarity to $\operatorname{AN}_Y$')
g.add_legend(prop={'size': 10}, loc = 'upper left')
new_labels = [r'$\hat{S}_{\operatorname{IAS}}$', r'$\hat{S}_{\operatorname{ICP}}^{\operatorname{MB}}$']
for t, l in zip(g._legend.texts, new_labels):
   t.set_text(l)
for ax in g.axes.flat:
    # Only the 2nd and 4th axes have something in ax.texts
    if ax.texts:
        # This contains the right ylabel text
        txt = ax.texts[0]
        ax.text(txt.get_unitless_position()[0], txt.get_unitless_position()[1],
                txt.get_text(),
                transform=ax.transAxes,
                va='center',
                rotation=0)
        # Remove the original text
        ax.texts[0].remove()
g.savefig('output/finite_experiment/figures/jaccard_fig_main_Repeat_StandardCorrection.svg', dpi = 600)


#################### Graphs per window ####################
data.groupby(['N', 'group'])['group'].agg('count')
data2.groupby(['N', 'group'])['group'].agg('count')
###########################################################
##########################################################################################################################################



############################################################ Appendix figures ############################################################
############################ alpha0 = alpha ############################
db1 = shelve.open('output/finite_experiment/d_equal_6/alpha0_equal_None')
data = db1['data']
objs = db1['objlist']
db1.close()
db2 = shelve.open('output/finite_experiment/d_equal_100/alpha0_equal_None')
data2 = db2['data']
objs2 = db2['objlist']
db2.close()

data['d'] = 6
data2['d'] = 100
indices_d6   = np.where([CompareOutput(i, objs) != 0 for i in range(len(objs))])[0]
indices_d100 = np.where([CompareOutput(i, objs2) != 0 for i in range(len(objs2))])[0]
data['group']  = data['i'].transform(lambda x: r'different' if x in indices_d6 else r'equal')
data2['group'] = data2['i'].transform(lambda x: r'different' if x in indices_d100 else r'equal')

plotdata = pd.concat(
    [
        data[['N', 'J_AS', 'J_ICP', 'd', 'group']],
        data2[['N', 'J_AS', 'J_ICP', 'd', 'group']]
    ]
).melt(id_vars = ['N', 'd', 'group']).sort_values(by = 'group')


palette = {"J_AS": "#F8766D",
           "J_ICP": "#00BFC4"}
g = sns.FacetGrid(plotdata, col = 'group', row = 'd', margin_titles=True, legend_out=False)
g.map_dataframe(sns.boxplot, x = 'N', y = 'value', hue = 'variable', hue_order = ['J_AS', 'J_ICP'], palette = palette, fliersize=1.5)
for ax in g.axes[0]:
    ax.set_xticklabels([r'$10^2$', r'$10^3$', r'$10^4$', r'$10^5$'])
g.set_titles(col_template = r"Oracle IAS and ICP {col_name}", row_template = r"$d = {row_name}$")
g.set_axis_labels(x_var = r'Number of observations', y_var = r'Jaccard similarity to $\operatorname{AN}_Y$')
g.add_legend(prop={'size': 10})
new_labels = [r'$\hat{S}_{\operatorname{IAS}}$', r'$\hat{S}_{\operatorname{ICP}}^{\operatorname{MB}}$']
for t, l in zip(g._legend.texts, new_labels):
   t.set_text(l)
for ax in g.axes.flat:
    # Only the 2nd and 4th axes have something in ax.texts
    if ax.texts:
        # This contains the right ylabel text
        txt = ax.texts[0]
        ax.text(txt.get_unitless_position()[0], txt.get_unitless_position()[1],
                txt.get_text(),
                transform=ax.transAxes,
                va='center',
                rotation=0)
        # Remove the original text
        ax.texts[0].remove()
g.savefig('output/finite_experiment/figures/jaccard_fig_appendix_alpha0None.svg', dpi = 600)

############################ alpha0 = 10e-12 ############################
db1 = shelve.open('output/finite_experiment/d_equal_6/alpha0_equal_10Eminus12')
data = db1['data']
objs = db1['objlist']
db1.close()
db2 = shelve.open('output/finite_experiment/d_equal_100/alpha0_equal_10Eminus12')
data2 = db2['data']
objs2 = db2['objlist']
db2.close()

data['d'] = 6
data2['d'] = 100
indices_d6   = np.where([CompareOutput(i, objs) != 0 for i in range(len(objs))])[0]
indices_d100 = np.where([CompareOutput(i, objs2) != 0 for i in range(len(objs2))])[0]
data['group']  = data['i'].transform(lambda x: r'different' if x in indices_d6 else r'equal')
data2['group'] = data2['i'].transform(lambda x: r'different' if x in indices_d100 else r'equal')

plotdata = pd.concat(
    [
        data[['N', 'J_AS', 'J_ICP', 'd', 'group']],
        data2[['N', 'J_AS', 'J_ICP', 'd', 'group']]
    ]
).melt(id_vars = ['N', 'd', 'group']).sort_values(by = 'group')


palette = {"J_AS": "#F8766D",
           "J_ICP": "#00BFC4"}
g = sns.FacetGrid(plotdata, col = 'group', row = 'd', margin_titles=True, legend_out=False)
g.map_dataframe(sns.boxplot, x = 'N', y = 'value', hue = 'variable', hue_order = ['J_AS', 'J_ICP'], palette = palette, fliersize=1.5)
for ax in g.axes[0]:
    ax.set_xticklabels([r'$10^2$', r'$10^3$', r'$10^4$', r'$10^5$'])
g.set_titles(col_template = r"Oracle IAS and ICP {col_name}", row_template = r"$d = {row_name}$")
g.set_axis_labels(x_var = r'Number of observations', y_var = r'Jaccard similarity to $\operatorname{AN}_Y$')
g.add_legend(prop={'size': 10})
new_labels = [r'$\hat{S}_{\operatorname{IAS}}$', r'$\hat{S}_{\operatorname{ICP}}^{\operatorname{MB}}$']
for t, l in zip(g._legend.texts, new_labels):
   t.set_text(l)
for ax in g.axes.flat:
    # Only the 2nd and 4th axes have something in ax.texts
    if ax.texts:
        # This contains the right ylabel text
        txt = ax.texts[0]
        ax.text(txt.get_unitless_position()[0], txt.get_unitless_position()[1],
                txt.get_text(),
                transform=ax.transAxes,
                va='center',
                rotation=0)
        # Remove the original text
        ax.texts[0].remove()
g.savefig('output/finite_experiment/figures/jaccard_fig_appendix_alpha010Eminus12.svg', dpi = 600)
##########################################################################################################################################










