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
    args[3] = counter
    """
    if args[2] == 'dense':
        p = 0.75
    else:
        p = 2 / args[0]
    g = SampleDAG(args[0], args[1], p)
    AN_Y  = nx.ancestors(g, 'Y') - {'E'}
    #IP    = list(getIP_ANcheat(g))
    #MIP   = list(getMIP(IP))
    MIP   = get_MIP_dagitty(g)
    S_ICP = getICP(g)
    S_AS  = set.union(*MIP) if len(MIP) > 0 else set()
    J_AS  = jaccard(S_AS, AN_Y)
    J_ICP = jaccard(S_ICP, AN_Y)
    N_AS  = len(S_AS)
    N_AN  = len(AN_Y)
    N_ICP = len(S_ICP)
    N_MB  = len(MarkovBlanket(g))

    return (S_ICP < S_AS, S_AS < S_ICP, S_ICP < set(), J_AS, J_ICP, N_AS, N_ICP, N_MB, N_AS, *args)

B = 50000           # Numper of graphs to sample at each combination
ds = [4, 6, 8, 10, 12, 14, 16, 18, 20] # Predictor space sizes to try
counter = 0        # Only used to keep track of iteration numbers
Inputs = []

for p in ['dense', 'sparse']:
    for d in ds:
        for Nint in range(1, d):
            Inputs.extend([(d, Nint, p, counter)] * B)
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
        'd', 'Nint', 'p', 'iteration'
    ]
)
data_aggregated = data.groupby(['iteration', 'd', 'Nint', 'p']).agg(['mean', 'std']).reset_index()

database = shelve.open('data_d_in_4to20')
database['data'] = data_aggregated
database.close()