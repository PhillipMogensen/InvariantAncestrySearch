import pandas as pd
import shelve
from InvariantAncestralPrediction.utils import DataGenerator
from InvariantAncestralPrediction.IAPfunctions import *

num_predictors = 6  # Number predictors (i.e., variables different from E and Y)
num_interventions = lambda: random.choice([1])  # samples the number of interventions in each graph
p_connection = 2 / num_predictors  # Probability of any single edge being present in the graph
B_G = 250  # Number of graphs to sample
B_d = 50  # Number of datasets to sample per graph
Nseq = [100, 1000, 10000, 100000] #, 100000]  # Sequence of sample sizes to try
standardization = 'causalorder'  # The type of standardization scheme to use
IS = 1  # lambda: [random.choice([-3, -2, -1, 1, 2, 3])]
IT  = 'do'
sd = 1  #lambda: np.random.choice([0.5, 1, 2], 10).tolist()
assignments = 'linear'

objlist = []
for i in range(B_G):  # Generates B_G objects to samplte data from
    objlist.append(DataGenerator(
        d = num_predictors, N_interventions = num_interventions(), p_conn = p_connection, standardize = standardization
        # InterventionStrength = IT, InterventionType = IS, sd = sd, assignments = assignments
    ))
    objlist[i].SampleDAG()
    objlist[i].BuildCoefMatrix()


def run(i, type = 'linear', alpha = 0.05 * 2**6 / 8 ): #* 2**6 / 6):
    """
    This function runs the ancestral search on the i'th object in objlist B_d times for each n in Nseq
    """
    out   = []
    out2  = []

    for n in np.repeat(Nseq, B_d):
        tmp = ExhaustiveSearch(*objlist[i].MakeData(n), alpha, type = 'linear', unadjusted = False, alpha0 = 10**(-6))
        tmp2 = list(getMIP(tmp))
        tmp4 = tmp
        out.append(set.union(*tmp2) if len(tmp2) > 0 else set())
        out2.append(set.intersection(*tmp4) if len(tmp4) > 0 else set())

    return out, out2  #, outNA

def CheckRes(x, i):
    """
    Checks whether the output of run(i) is ancestral/in S_AS/empty and so forth
    """
    d = len(objlist[i].g.nodes()) - 2
    AN_Y = nx.ancestors(objlist[i].g, 'Y') - {'E'}
    PA_Y = set(objlist[i].g.predecessors('Y'))
    # # # # # # # IP = getIP_restricted_ANcheat(objlist[i].g, 1)
    # IP = getIP_ANcheat(objlist[i].g)
    # MIP = list(getMIP(IP))
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
    FPR = [  # Number of mistakes over number of possible mistakes. Is set to 0 if no mistakes are possible
        np.mean([len(a - AN_Y) / (d - len(AN_Y)) if len(AN_Y) < d else 0 for a in x2[j]]) for j in range(len(Nseq))
    ]

    data_out = pd.DataFrame([Nseq, Plevel, PinS, Pempty, lenS, J_AS, N_mistakes, lenICP, PemptyICP, PlevelICP, FPR, [len(AN_Y)] * len(Nseq), [len(S_AS)] * len(Nseq)]).transpose()
    data_out.set_axis(['N', 'Plevel', 'PinS', 'Pempty', 'lenS', 'J_AS', 'N_mistakes', 'lenICP', 'PemptyICP', 'PlevelICP', 'FPR', 'lenAN', 'lenTrueS'], axis = 1, inplace = True)

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

data.groupby('N')[['Plevel', 'Pempty', 'PlevelICP', 'PemptyICP', 'FPR']].agg('mean')
data['Plevel'].transform(lambda x: x >= 0.95).groupby(data['N']).agg('mean')

database = shelve.open('output/finite_experiment/d_eq_6_alpha0eq10eminus6')
database['data'] = data
database.close()

############################# d = 100 ####################################
num_predictors = 100  # Number predictors (i.e., variables different from E and Y)
num_interventions = lambda: random.choice([i + 1 for i in range(10)])  # samples the number of interventions in each graph
p_connection = 2 / num_predictors  # Probability of any single edge being present in the graph
B_G = 250  # Number of graphs to sample
B_d = 50  # Number of datasets to sample per graph
Nseq = [100, 1000, 10000, 100000] #, 100000]  # Sequence of sample sizes to try
standardization = 'causalorder'  # The type of standardization scheme to use
IS = 1  # lambda: [random.choice([-3, -2, -1, 1, 2, 3])]
IT  = 'do'
sd = 1  #lambda: np.random.choice([0.5, 1, 2], 10).tolist()
assignments = 'linear'

objlist = []
for i in range(B_G):  # Generates B_G objects to sample data from
    objlist.append(DataGenerator(
        d = num_predictors, N_interventions = num_interventions(), p_conn = p_connection, standardize = standardization
        # InterventionStrength = IT, InterventionType = IS, sd = sd, assignments = assignments
    ))
    objlist[i].SampleDAG()
    objlist[i].BuildCoefMatrix()


def run(i, type = 'linear', alpha = 0.05): #* 2**6 / 6):
    """
    This function runs the ancestral search on the i'th object in objlist B_d times for each n in Nseq
    """
    out   = []

    for n in np.repeat(Nseq, B_d):
        tmp = RestrictedSearch(*objlist[i].MakeData(n), alpha, 1, type = 'linear', alpha0 = 10**(-6))
        tmp2 = list(getMIP(tmp))
        tmp4 = tmp
        out.append(set.union(*tmp2) if len(tmp2) > 0 else set())

    return out

def CheckRes(x, i):
    """
    Checks whether the output of run(i) is ancestral/in S_AS/empty and so forth
    """
    d = len(objlist[i].g.nodes()) - 2
    AN_Y = nx.ancestors(objlist[i].g, 'Y') - {'E'}
    # # # # # # # IP = getIP_ANcheat(objlist[i].g)
    IP = getIP_restricted_ANcheat(objlist[i].g, 1)
    MIP = list(getMIP(IP))
    S_AS = set.union(*MIP) if len(MIP) > 0 else set()

    x2 = np.split(np.array(x), len(Nseq))

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
    N_mistakes = [
        np.mean([len(a - AN_Y) for a in x2[j]]) for j in range(len(Nseq))
    ]
    FPR = [
        np.mean([len(a - AN_Y) / (d - len(AN_Y)) if len(AN_Y) < d else 0 for a in x2[j]]) for j in range(len(Nseq))
    ]

    data_out = pd.DataFrame([Nseq, Plevel, PinS, Pempty, lenS, J_AS, N_mistakes, FPR, [len(AN_Y)] * len(Nseq), [len(S_AS)] * len(Nseq)]).transpose()
    data_out.set_axis(['N', 'Plevel', 'PinS', 'Pempty', 'lenS', 'J_AS', 'N_mistakes', 'FPR', 'lenAN', 'lenTrueS'], axis = 1, inplace = True)

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

data.groupby('N')[['Plevel', 'Pempty', 'FPR']].agg('mean')
data['Plevel'].transform(lambda x: x >= 0.95).groupby(data['N']).agg('mean')

database = shelve.open('output/finite_experiment/d_eq_100_alpha0eq10eminus6')
database['data'] = data
database.close()
