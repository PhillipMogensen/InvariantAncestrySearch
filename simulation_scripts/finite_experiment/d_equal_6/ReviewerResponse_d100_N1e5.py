import pandas as pd
import shelve
from InvariantAncestrySearch.utils import DataGenerator
from InvariantAncestrySearch.IASfunctions import *
import numpy as np
from causaldag import unknown_target_igsp, partial_correlation_suffstat, partial_correlation_test, MemoizedCI_Tester, gsp, gauss_invariance_suffstat, MemoizedInvarianceTester, gauss_invariance_test
import random
from time import time

########################################## Global parameters ##########################################
num_predictors = 100  # Number predictors (i.e., variables different from E and Y)
num_interventions = lambda: random.choice([i + 1 for i in range(5)])  # samples the number of interventions in each graph
p_connection = 2 / num_predictors  # Probability of any single edge being present in the graph
B_G = 100  # Number of graphs to sample
B_d = 50   # Number of datasets to sample per graph
Nseq = [100000] # Sequence of sample sizes to try

alpha_0  = 10**(-6)        # Level to test the empty set at
alpha_   = 0.05 # * 2**6 / 9   # Level to test remaining sets at. (we test at 0.05 / 9, but multiply by 2**6, because the test ExhaustiveSearch tests at level alpha / 2**d)
#########################################################################################################

db = shelve.open('output/finite_experiment/d_equal_100/ReviewerResponse')  # Change file name according to value of alpha_0
objlist = db['objlist']
db.close()
# objlist = []
# for i in range(B_G):  # Generates B_G objects to sample data from
#     objlist.append(DataGenerator(
#         d = num_predictors, N_interventions = num_interventions(), p_conn = p_connection, InterventionStrength = .5
#     ))
#     objlist[i].SampleDAG()
#     objlist[i].BuildCoefMatrix()

# pd.DataFrame({'value': [objlist[i].N_interventions for i in range(len(objlist))]}).agg(['mean', 'median'])

def UT_IGSP(E, X, Y, g, alpha = 0.05):  # Runs the method proposed by Reviewer #5
    obs_samples = np.c_[X[E == 0], Y[E == 0]]
    iv_samples_list = [np.c_[X[E == 1], Y[E == 1]]]
    obs_suffstat = partial_correlation_suffstat(obs_samples)
    invariance_suffstat = gauss_invariance_suffstat(obs_samples, iv_samples_list)
    ci_tester = MemoizedCI_Tester(partial_correlation_test, obs_suffstat, alpha=alpha)
    invariance_tester = MemoizedInvarianceTester(gauss_invariance_test, invariance_suffstat, alpha=alpha)
    setting_list = [dict(known_interventions=[])]  
    est_dag, est_targets_list = unknown_target_igsp(setting_list, len(g.nodes()) - 1, ci_tester, invariance_tester)
    g_ = nx.relabel_nodes(est_dag.to_nx(), {num_predictors: 'Y'})
    g_.add_nodes_from('E')
    g_.add_edges_from([('E', a) for a in est_targets_list[0]])
    MIP = get_MIP_dagitty(g_)
    return set.union(*MIP) if len(MIP) > 0 else set()

def run(i, alpha = alpha_):  # Wrapper function that; 1) simulates data from objlist[i] for samples sizes in Nseq, 2) applies IAS for each dataset
    """
    This function runs the ancestral search on the i'th object in objlist B_d times for each n in Nseq
    """
    IAS_out     = []
    IAS_runtime = []
    UT_IGSP_out = []
    UT_IGSP_runtime = []

    for n in np.repeat(Nseq, B_d):
        E, X, Y = objlist[i].MakeData(n)
        t_IAS = time()
        IP = RestrictedSearch(E, X, Y, alpha, MaxSize = 1, alpha0 = alpha_0)
        MIP = list(getMIP(IP))
        IAS_runtime.append(time() - t_IAS)
        t_UT_IGSP = time()
        UT_IGSP_ = UT_IGSP(E, X, Y, objlist[i].g, 0.05)
        UT_IGSP_runtime.append(time() - t_UT_IGSP)
        IAS_out.append(set.union(*MIP) if len(MIP) > 0 else set())
        UT_IGSP_out.append(UT_IGSP_)

    return IAS_out, UT_IGSP_out, np.mean(IAS_runtime), np.mean(UT_IGSP_runtime)


def CheckRes(x, i):
    """
    Checks whether the output of run(i) is ancestral/in S_AS/empty and so forth
    """
    d = len(objlist[i].g.nodes()) - 2
    AN_Y = nx.ancestors(objlist[i].g, 'Y') - {'E'}
    PA_Y = set(objlist[i].g.predecessors('Y'))
    MIP = get_MIP_dagitty(objlist[i].g)
    S_AS = set.union(*MIP) if len(MIP) > 0 else set()

    x2 = np.split(np.array(x[0]), len(Nseq))  # IAS
    x4 = np.split(np.array(x[1]), len(Nseq))  # UT-IGSP

    Plevel = [
        np.mean([a <= AN_Y for a in x2[j]]) for j in range(len(Nseq))
    ]
    Plevel_UT_IGSP = [
        np.mean([a <= AN_Y for a in x4[j]]) for j in range(len(Nseq))
    ]
    Pempty = [
        np.mean([a <= set() for a in x2[j]])  for j in range(len(Nseq))
    ]
    Pempty_UT_IGSP = [
        np.mean([a <= set() for a in x4[j]])  for j in range(len(Nseq))
    ]
    J_AS = [
        np.mean([jaccard(a, AN_Y) for a in x2[j]]) for j in range(len(Nseq))
    ]
    J_UT_IGSP = [
        np.mean([jaccard(a, AN_Y) for a in x4[j]]) for j in range(len(Nseq))
    ]
    J_AS_IAS = [
        np.mean([jaccard(a, S_AS) for a in x2[j]]) for j in range(len(Nseq))
    ]
    J_UT_IGSP_IAS = [
        np.mean([jaccard(a, S_AS) for a in x4[j]]) for j in range(len(Nseq))
    ]
    IsInvariant_AS = [
        np.mean([nx.d_separated(objlist[i].g, {"E"}, {'Y'}, a) for a in x2[j]]) for j in range(len(Nseq))
    ]
    IsInvariant_UT_IGSP = [
        np.mean([nx.d_separated(objlist[i].g, {"E"}, {'Y'}, a) for a in x4[j]]) for j in range(len(Nseq))
    ]


    data_out = pd.DataFrame([Nseq, IsInvariant_AS, IsInvariant_UT_IGSP, Plevel, Plevel_UT_IGSP, Pempty, Pempty_UT_IGSP, J_AS, J_UT_IGSP, J_AS_IAS, J_UT_IGSP_IAS, [len(AN_Y)] * len(Nseq), [len(S_AS)] * len(Nseq)]).transpose()
    data_out.set_axis(['N', 'IsInvariant_AS', 'IsInvariant_UT_IGSP', 'Plevel', 'Plevel_UT_IGSP', 'Pempty', 'Pempty_UT_IGSP', 'J_AS', 'J_UT_IGSP', 'J_AS_IAS', 'J_UT_IGSP_IAS', 'lenAN', 'lenTrueS'], axis = 1, inplace = True)

    return data_out


res = []
t0 = time()
for i in range(len(objlist)):
    print("\r", i, " of ", len(objlist), ". Time spent: ", round(time() - t0), end = " ", flush = True)
    res.append(run(i))

data = pd.concat([CheckRes(res[i], i) for i in range(len(res))])
data['i'] = np.repeat(range(B_G), len(Nseq))


# database = shelve.open('output/finite_experiment/d_equal_100/ReviewerResponse_N1e5')  # Change file name according to value of alpha_0
# database['data'] = data
# database['raw_results'] = res
# database['objlist'] = objlist
# database.close()



########################## Plot parameters #####################################
import matplotlib.pyplot as plt
plt.rc('text', usetex=True)
plt.rc('text.latex', preamble=r'\usepackage{amsmath,amssymb}')
#font = {'size': 26}
font = {'size': 16}
plt.rc('font', **font)
################################################################################

d100     = shelve.open('output/finite_experiment/d_equal_100/ReviewerResponse_N1e5')
data = d100['data']
objlist  = d100['objlist']
d100.close()


data.groupby('N')[['Plevel', 'Pempty', 'Plevel_UT_IGSP', 'Pempty_UT_IGSP', 'J_AS', 'J_UT_IGSP']].agg('mean')

def CompareOutput(i, objs):
    MIP = get_MIP_dagitty(objs[i].g)
    rMIP = [a for a in MIP if len(a) <= 1]
    S_AS = set.union(*rMIP) if len(rMIP) > 0 else set()
    S_ICP = getICP(objs[i].g)
    
    return len(S_AS) - len(S_ICP)

indices_d6   = np.where([CompareOutput(i, objlist) != 0 for i in range(len(objlist))])[0]
data['group']  = data['i'].transform(lambda x: r'different' if x in indices_d6 else r'equal')

# plotdata = pd.DataFrame(
#     {
#         "Method": np.repeat(['IAS', 'UT-IGSP'], [data.shape[0]] * 2),
#         "N": data['N'].tolist() * 2,
#         "Jaccard similarity to AN_Y": data['J_AS'].tolist() + data['J_UT_IGSP'].tolist(),
#         "Group": data['group'].tolist() * 2
#     }
# )
plotdata = pd.DataFrame(
    {
        "Method": np.repeat(['IAS', 'UT-IGSP'], [data.shape[0]] * 2),
        "N": data['N'].tolist() * 2,
        "Jaccard similarity to AN_Y": data['J_AS'].tolist() + data['J_UT_IGSP'].tolist(),
        "Group": data['group'].tolist() * 2
    }
)
import seaborn as sns
import matplotlib.pyplot as plt
plt.clf() ; plt.cla()
sns.boxplot(x = 'N', y = 'Jaccard similarity to AN_Y', hue = 'Method', data = plotdata, showmeans = True)
plt.ylabel(r'Jaccard similarity to $AN_Y$')
plt.show()
# plt.savefig('output/finite_experiment/figures/ReviewerResponse.svg')
