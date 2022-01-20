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
Nseq = [100, 1000, 10000, 100000] #, 100000]  # Sequence of sample sizes to try

alpha_0  = 10**(-6)        # Level to test the empty set at
alpha_   = 0.05 * 2**6 / 9   # Level to test remaining sets at. (we test at 0.05 / 9, but multiply by 2**6, because the test ExhaustiveSearch tests at level alpha / 2**d)
#########################################################################################################

objlist = []
for i in range(B_G):  # Generates B_G objects to sample data from
    objlist.append(DataGenerator(
        d = num_predictors, N_interventions = num_interventions(), p_conn = p_connection, InterventionStrength = 1
    ))
    objlist[i].SampleDAG()
    objlist[i].BuildCoefMatrix()


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
        ICP_ = ICP(E, X, Y, alpha)
        IAS_out.append(set.union(*MIP) if len(MIP) > 0 else set())
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
    x3 = np.split(np.array(x[1]), len(Nseq))

    Plevel = [
        np.mean([a <= AN_Y for a in x2[j]]) for j in range(len(Nseq))
    ]
    PinS = [
        np.mean([a <= S_AS for a in x2[j]]) for j in range(len(Nseq))
    ]
    Pempty = [
        np.mean([a <= set() for a in x2[j]])  for j in range(len(Nseq))
    ]
    lenS = [
        np.mean([len(a) for a in x2[j]])  for j in range(len(Nseq))
    ]
    J_AS = [
        np.mean([jaccard(a, AN_Y) for a in x2[j]]) for j in range(len(Nseq))
    ]
    J_ICP = [
        np.mean([jaccard(a, AN_Y) for a in x3[j]]) for j in range(len(Nseq))
    ]
    N_mistakes = [
        np.mean([len(a - AN_Y) for a in x2[j]]) for j in range(len(Nseq))
    ]
    lenICP = [
        np.mean([len(a) for a in x3[j]])  for j in range(len(Nseq))
    ]
    PemptyICP = [
        np.mean([a <= set() for a in x3[j]])  for j in range(len(Nseq))
    ]
    PlevelICP = [
        np.mean([a <= PA_Y for a in x3[j]]) for j in range(len(Nseq))
    ]
    ICP_in_AN = [
        np.mean([a <= AN_Y for a in x3[j]]) for j in range(len(Nseq))
    ]
    FPR = [  # Number of mistakes over number of possible mistakes. Is set to 0 if no mistakes are possible
        np.mean([len(a - AN_Y) / (d - len(AN_Y)) if len(AN_Y) < d else 0 for a in x2[j]]) for j in range(len(Nseq))
    ]
    ANcestorsFound_IAS = [
        np.mean([len(set.intersection(AN_Y, a)) for a in x2[j]]) for j in range(len(Nseq))
    ]
    ANcestorsFound_ICP = [
        np.mean([len(set.intersection(AN_Y, a)) for a in x3[j]]) for j in range(len(Nseq))
    ]

    data_out = pd.DataFrame([Nseq, Plevel, PinS, Pempty, lenS, J_AS, J_ICP, N_mistakes, lenICP, PemptyICP, PlevelICP, ICP_in_AN, FPR, [len(AN_Y)] * len(Nseq), [len(S_AS)] * len(Nseq)]).transpose()
    data_out.set_axis(['N', 'Plevel', 'PinS', 'Pempty', 'lenS', 'J_AS', 'J_ICP', 'N_mistakes', 'lenICP', 'PemptyICP', 'PlevelICP', 'ICP_in_AN', 'FPR', 'lenAN', 'lenTrueS'], axis = 1, inplace = True)

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

# data.groupby('N')[['Plevel', 'Pempty', 'PlevelICP', 'PemptyICP', 'FPR']].agg('mean')
# data['Plevel'].transform(lambda x: x >= alpha_).groupby(data['N']).agg('mean')

database = shelve.open('output/finite_experiment/d_equal_6/alpha0_equal_10Eminus6_9Correction')  # Change file name according to value of alpha_0
database['data'] = data
database['raw_results'] = res
database['objlist'] = objlist
database.close()
