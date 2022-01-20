from InvariantAncestrySearch.IASfunctions import getIP_ANcheat, getMIP, get_MIP_dagitty
from InvariantAncestrySearch.utils import SampleDAG
from time import time
import random
import networkx as nx

def getTime(d):
    g = SampleDAG(d, 5, 0.75)
    g_ = nx.Graph.subgraph(g, set.union({'Y'}, nx.ancestors(g, 'Y')))

    t0 = time() ; list(getMIP(getIP_ANcheat(g_))) ; t0 = time() - t0
    t1 = time() ; get_MIP_dagitty(g_) ; t1 = time() - t1

    return t0, t1, t0 - t1, t0 / t1

#### runtimes ####
pool = Pool(cpu_count() - 1)
brute = []
smart = []
absolute = []
relative = []
for x in list(tqdm(pool.imap(getTime, [15] * 500), total = 500)):
    brute.append(x[0])
    smart.append(x[1])
    absolute.append(x[2])
    relative.append(x[3])
pool.close()
pool.join()
np.min(relative)
np.median(relative)
np.max(relative)
database = shelve.open('output/benchmarks/RelativeComputionTime')
database['relative'] = relative
database['brute'] = brute
database['dagitty'] = smart
database.close()