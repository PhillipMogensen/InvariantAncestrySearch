from InvariantAncestralPrediction.IAPfunctions import getIP_ANcheat, getMIP, get_MIP_dagitty
from InvariantAncestralPrediction.utils import SampleDAG
from time import time
import random
import networkx as nx

def getTime(d):
    g = SampleDAG(d, random.choice([i + 1 for i in range(d)]), 0.75)
    g_ = nx.Graph.subgraph(g, set.union({'Y'}, nx.ancestors(g, 'Y')))

    t0 = time() ; list(getMIP(getIP_ANcheat(g_))) ; t0 = time() - t0
    t1 = time() ; get_MIP_dagitty(g_) ; t1 = time() - t1

    return t0, t1

#### runtimes ####
brute, smart = zip(*[getTime(15) for i in range(100)])