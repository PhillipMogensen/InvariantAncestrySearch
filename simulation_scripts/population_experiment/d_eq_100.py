import pandas as pd
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from InvariantAncestralPrediction.utils import SampleDAG
from InvariantAncestralPrediction.IAPfunctions import *
import shelve

def WrapperFun(args):
    """
    args[0] = d
    args[1] = N_interventions
    args[2] = dense or sparse
    args[3] = max size to search
    args[4] = counter
    """
    if args[2] == 'dense':
        p = 0.05 if args[0] == 100 else 0.005  # Not used
    else:
        p = 2 / args[0]
    g = SampleDAG(args[0], args[1], p)
    AN_Y  = nx.ancestors(g, 'Y') - {'E'}
    g_ = nx.Graph.subgraph(g, list(set.union({'E', 'Y'}, AN_Y)))
    # IP    = list(getIP_restricted_ANcheat(g, args[3]))
    # MIP   = list(getMIP(IP))
    MIP   = [a for a in get_MIP_dagitty(g_) if len(a) <= args[3]]
    MB    = list(MarkovBlanket(g))
    N_MB  = len(MB)
    if N_MB > 10:
        MB = set(np.random.choice(list(MB), 10).tolist())
    
    # S_ICP = getICP(g)
    IPMB  = list(getIP_from_MarkovBlanket(g, MB))
    S_ICP = set.intersection(*IPMB) if len(IPMB) > 0 else set()
    S_AS  = set.union(*MIP) if len(MIP) > 0 else set()
    J_AS  = jaccard(S_AS, AN_Y)
    J_ICP = jaccard(S_ICP, AN_Y)
    N_AS  = len(S_AS)
    N_AN  = len(AN_Y)
    N_ICP = len(S_ICP)

    return (S_ICP < S_AS, S_AS < S_ICP, S_ICP < set(), J_AS, J_ICP, N_AS, N_ICP, N_MB, N_AS, *args)

B = 50000          # Numper of graphs to sample at each combination
ds = [100, 1000]   # Predictor space sizes to try
m1 = [1, 2]        # Max set size to search for when d = 100
m2 = [1]           # Max set size to search for when d = 1000
counter = 0        # Only used to keep track of iteration numbers
Inputs = []

for p in ['sparse']:
    for d in ds:
        Nmax = 10 if d == 100 else 100  # Number of interventions to try up to
        for Nint in range(1, Nmax + 1):
            mlist = m1 if d == 100 else m2
            for m in mlist:
                Inputs.extend([(d, Nint, p, m, counter)] * B)
                counter += 1

results = []
pool = Pool(cpu_count() - 1)
for x in list(tqdm(pool.imap(WrapperFun, Inputs), total = len(Inputs))):
    results.append(x)

data = pd.DataFrame.from_records(
    results,
    columns = [
        'StrictlyLarger', 'StrictlySmaller', 'ICPempty', 
        'JaccardAS', 'jaccardICP', 'lenAS', 'lenICP', 
        'lenMarkovBlanket', 'lenAN',
        'd', 'Nint', 'p', 'MaxSize', 'iteration'
    ]
)
data_aggregated = data.groupby(['iteration', 'd', 'Nint', 'p', 'MaxSize']).agg(['mean', 'std']).reset_index()
# output/population_experiment/
database = shelve.open('data_d_in_100_1000_m_1_2')
database['data'] = data_aggregated
database.close()
