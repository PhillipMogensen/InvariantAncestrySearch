import imp
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import shelve
from InvariantAncestrySearch.utils import DataGenerator
from InvariantAncestrySearch.IASfunctions import *

########################################## Global parameters ##########################################
num_predictors = 6  # Number predictors (i.e., variables different from E and Y)
num_interventions = lambda: 1  # samples the number of interventions in each graph
p_connection = 2 / num_predictors  # Probability of any single edge being present in the graph
B_G = 100  # Number of graphs to sample
B_d = 50   # Number of datasets to sample per graph
Nseq = [100, 1000, 10000, 100000]  # Sequence of sample sizes to try

alpha_0  = 10**(-6)        # Level to test the empty set at
alpha_   = 0.05 * 2**6 / 9   # Level to test remaining sets at. (we test at 0.05 / 9, but multiply by 2**6, because the test ExhaustiveSearch tests at level alpha / 2**d)
#########################################################################################################

#### Load objects from previous simulation ####
# These are used to extract the graphs and coefficients, such that we run on the exact same SCMs
# only with a weaker instrument strength (here 0.5 instead of 1.0)
database = shelve.open('output/finite_experiment/d_equal_6/alpha0_equal_10Eminus6')
objlist_old = database['objlist']
database.close()
# Make new datagenerators with same graphs/coefficients as main experiment but different intervention strength
objlist = []
for i in range(B_G):
    objlist.append(DataGenerator(
        d = num_predictors, N_interventions = num_interventions(), p_conn = p_connection, InterventionStrength = 0.5
    ))
    objlist[i].g = objlist_old[i].g
    objlist[i].CM = objlist_old[i].CM


def run(i, alpha = alpha_):  # Wrapper function that; 1) simulates data from objlist[i] for samples sizes in Nseq, 2) applies IAS for each dataset
    """
    This function runs the ancestral search on the i'th object in objlist B_d times for each n in Nseq
    """
    IAS_out  = []
    ICP_out  = []  

    for n in np.repeat(Nseq, B_d):
        E, X, Y = objlist[i].MakeData(n)
        IP = ExhaustiveSearch(E, X, Y, alpha, alpha0 = alpha_0)
        MIP = list(getMIP(IP))
        IAS_out.append(set.union(*MIP) if len(MIP) > 0 else set())
        ICP_ = ICP(E, X, Y, alpha)
        ICP_out.append(ICP_)

    return IAS_out, ICP_out

def CheckRes(x, i):
    """
    Checks whether the output of run(i) is ancestral/in S_AS/empty and so forth
    """
    d = len(objlist[i].g.nodes()) - 2
    AN_Y = nx.ancestors(objlist[i].g, 'Y') - {'E'}
    PA_Y = set(objlist[i].g.predecessors('Y'))
    MIP = get_MIP_dagitty(objlist[i].g)
    S_AS = set.union(*MIP) if len(MIP) > 0 else set()

    x2 = np.split(np.array(x[0]), len(Nseq))
    x3 = np.split(np.array(x[1]), len(Nseq))  # ICP not recorded here

    Plevel = [
        np.mean([a <= AN_Y for a in x2[j]]) for j in range(len(Nseq))
    ]
    Pempty = [
        np.mean([a <= set() for a in x2[j]])  for j in range(len(Nseq))
    ]
    J_AS = [
        np.mean([jaccard(a, AN_Y) for a in x2[j]]) for j in range(len(Nseq))
    ]
    J_ICP = [
        np.mean([jaccard(a, AN_Y) for a in x3[j]]) for j in range(len(Nseq))
    ]
    PemptyICP = [
        np.mean([a <= set() for a in x3[j]])  for j in range(len(Nseq))
    ]
    PlevelICP = [
        np.mean([a <= PA_Y for a in x3[j]]) for j in range(len(Nseq))
    ]

    data_out = pd.DataFrame([Nseq, Plevel, Pempty, J_AS, J_ICP, PemptyICP, PlevelICP, [len(AN_Y)] * len(Nseq), [len(S_AS)] * len(Nseq)]).transpose()
    data_out.set_axis(['N', 'Plevel', 'Pempty', 'J_AS', 'J_ICP', 'PemptyICP', 'PlevelICP', 'lenAN', 'lenTrueS'], axis = 1, inplace = True)

    return data_out

from time import time
res = []
t0 = time()
for i in range(len(objlist)):
    print("\r", i, " of ", len(objlist), ". Time spent: ", round(time() - t0), end = " ", flush = True)
    res.append(run(i))

data = pd.concat([CheckRes(res[i], i) for i in range(len(res))])
data['n_int'] = np.repeat([i.N_interventions for i in objlist], len(Nseq))
data['i'] = np.repeat(range(B_G), len(Nseq))

data.groupby('N')[['Plevel', 'Pempty']].agg(['mean', 'median'])
# data['Plevel'].transform(lambda x: x >= alpha_).groupby(data['N']).agg('mean')

database = shelve.open('output/finite_experiment/d_equal_6/alpha0_equal_10Eminus6_WeakInstrument')  # Change file name according to value of alpha_0
database['data'] = data
database['raw_results'] = res
database['objlist'] = objlist
database.close()






######################## Jaccard plot #########################

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
indices_d6   = np.where([CompareOutput(i, objlist) != 0 for i in range(len(objlist))])[0]
data['group']  = data['i'].transform(lambda x: r'different' if x in indices_d6 else r'equal')
plotdata = data[['N', 'J_AS', 'J_ICP', 'group']].melt(id_vars = ['N', 'group']).sort_values(by = 'group')

palette = {"J_AS": "#F8766D",
           "J_ICP": "#00BFC4"}
g = sns.FacetGrid(plotdata, col = 'group', legend_out=False)
g.map_dataframe(sns.boxplot, x = 'N', y = 'value', hue = 'variable', hue_order = ['J_AS', 'J_ICP'], palette = palette, fliersize=1.5)
for ax in g.axes[0]:
    ax.set_xticklabels([r'$10^2$', r'$10^3$', r'$10^4$', r'$10^5$'])
g.set_titles(col_template = r"Oracle IAS and ICP {col_name}", row_template = r"$d = {row_name}$")
g.set_axis_labels(x_var = r'Number of observations', y_var = r'Jaccard similarity between $\hat{S}_{\operatorname{IAS}}$ and $\operatorname{AN}_Y$')
g.add_legend(prop={'size': 10}, loc = 'upper left')
new_labels = [r'$\hat{S}_{\operatorname{IAS}}$', r'$\hat{S}_{\operatorname{ICP}}^{\operatorname{MB}}$']
for t, l in zip(g._legend.texts, new_labels):
   t.set_text(l)
g.savefig('output/finite_experiment/figures/jaccard_fig_WeakInstruments.svg', dpi = 600)
