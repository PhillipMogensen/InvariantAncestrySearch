#from sklearn.testing import ignore_warnings
#from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import Lasso, LassoCV, lasso_path
from pygam import LinearGAM, s
from pygam.terms import TermList
import numpy as np
import networkx as nx
import random
from scipy.stats import ttest_ind, levene, bartlett, cauchy, kstest
from scipy.linalg import inv
from itertools import combinations, product, islice
from math import pi
from scipy.special import comb
from rpy2.robjects.packages import importr
dagitty = importr('dagitty')

def powerset(l):
    """Convenience function to sample all subsets"""
    for sl in product(*[[[], [i]] for i in l]):
        yield {j for i in sl for j in i}

def CombSet(l, m):
  """Convenience function to generate all combinations up size m"""
  for i in range(m + 1):
    for j in combinations(l, i):
      yield set(j)


def getIP(g):
    """Generate all invariant sets"""
    num_nodes = len(g.nodes) - 2
    for S in powerset(range(num_nodes)):
        if nx.d_separated(g, {"E"}, {'Y'}, S):
            yield S


def getIP_restricted(g, m):
    """Generate all invariant sets of size less than m"""
    num_nodes = len(g.nodes) - 2
    for S in CombSet(range(num_nodes), m):
        if nx.d_separated(g, {"E"}, {'Y'}, S):
            yield S

def getIP_restricted_ANcheat(g, m):
    """Generate all invariant sets of size less than m"""
    # num_nodes = len(g.nodes) - 2
    AN_Y = list(nx.ancestors(g, 'Y') - {'E'})
    for S in CombSet(AN_Y, m):
        if nx.d_separated(g, {"E"}, {'Y'}, S):
            yield S

def getIP_ANcheat(g):
    """Generate all invariant sets among ancestors"""
    AN_Y = list(nx.ancestors(g, 'Y') - {'E'})
    for S in powerset(AN_Y):
        if nx.d_separated(g, {"E"}, {'Y'}, S):
            yield S

def getMIP(ip):
    """
    From a sequence of invariant sets, generate all minimally invariant sets
    """
    ip = list(ip)
    ip.sort(key=len)
    while len(ip) > 0:
        s = ip.pop(0)
        for s_ in reversed(ip):
            if s <= s_: ip.remove(s_)
        yield s

def to_dagitty(G):
    """
    Convert graph G to a dagitty string, "dag {E Y 0 1 E->1 Y->0}"
    Input: 
        - G: networx DAG, with nodes "E" and "Y"
    """
    return "\n".join(['dag', '{'] + [f'{v}' for v in G.nodes] + [f"{i} -> {j}" for (i,j) in G.edges] + ['}'])

def get_MIP_dagitty(G):
    """
    Get all minimal invariant predictors, by converting to the R dagitty package in R. 
    Input: 
        - G: networx DAG, with nodes "E" and "Y"
    """
    dagitty_mips = dagitty.adjustmentSets(dagitty.as_dagitty(to_dagitty(G)), 'E', 'Y', type = 'minimal', effect = 'direct')
    return [{int(j) for j in dagitty_mips.rx2(i)} for i in dagitty_mips.names]

def MarkovBlanket(g):
    PA_Y = set(g.predecessors('Y'))
    CH_Y = set(g.successors('Y'))
    PACH = list()
    for i in CH_Y:
        PACH.extend(list(g.predecessors(i)))
    
    return set.union(PA_Y, CH_Y, set(PACH)) - {'Y', 'E'}

def getICP(g):
    PA_Y = set(g.predecessors('Y'))
    CH_E = set(g.successors('E'))
    AN_Y = set(nx.ancestors(g, 'Y'))
    CH_AN = set.intersection(CH_E, AN_Y)
    PA_CH_AN = set.union(*[set(g.predecessors(i)) for i in CH_AN])

    RHS = set.union(CH_E, PA_CH_AN)

    return set.intersection(PA_Y, RHS)

def getIP_from_MarkovBlanket(g, MB):
    for S in powerset(list(set.intersection(set(MB), nx.ancestors(g, 'Y')))):
        if nx.d_separated(g, {'E'}, {'Y'}, S):
            yield S


def jaccard(A, B):
  """
  Calculates the Jaccard similarity, J = |A cap B| / |A cup B|, for sets A and B
  """
  numerator = len(set.intersection(A, B))
  denominator = len(set.union(A, B))
  if denominator <= 0:
    return 1
  else:
    return len(set.intersection(A, B)) / len(set.union(A, B))


def JP_test(E, X, Y, covar):
    """
    Tests the specified set of covariates for invariance, as described in the
    original ICP paper (method II)
    """
    if covar is None or covar == []:
        resid = Y
    else:
        if len(covar) == 1:
            X2 = X[:, covar].reshape(-1, 1)
        else:
            X2 = X[:, covar]
        
        b = inv(X2.T.dot(X2)).dot(X2.T).dot(Y)
        resid = Y - X2.dot(b)

    N_e = 2
    E_unique = [0, 1]

    out = np.min(
        np.multiply(
            [
                ttest_ind(resid[E == E_unique[0]], resid[E == E_unique[1]])[1],
                levene(resid[E == E_unique[0]], resid[E == E_unique[1]])[1]
            ],
            [2]
        )
    )

    return out

def RestrictedSearch(E, X, Y, alpha, MaxSize, alpha0 = None):
    """
    Same as ExhaustiveSearch (see below), but only runs up to a preset MaxSize set
    """
    d = X.shape[1]
    out_Adjust = []

    AdjustFactor = np.sum([comb(d, i) for i in range(MaxSize + 1)])

    if alpha0 is None:
        alpha0 = alpha / AdjustFactor

    EmptyInvariant = JP_test(E, X, Y, None) >= alpha0
    if EmptyInvariant:
        return set()
    
    for x in CombSet(range(d), MaxSize):
        pval = JP_test(E, X, Y, list(x))
        if pval >= alpha / AdjustFactor:
            out_Adjust.append(set(x))
            if len(x) == 0:
                return out_Adjust
    
    return out_Adjust

def ExhaustiveSearch(E, X, Y, alpha, alpha0 = None):
    d = X.shape[1]
    out_Adjust = []
    AdjustFactor = 2**d
    if alpha0 is None:
        alpha0 = alpha / AdjustFactor

    EmptyInvariant = JP_test(E, X, Y, None) >= alpha0
    if EmptyInvariant:
        return set()
    for x in powerset(range(d)):
        if len(out_Adjust) >0:
            if np.sum([s <= x for s in out_Adjust]) >= 1:
                continue
        pval = JP_test(E, X, Y, list(x))
        if pval >= (alpha / AdjustFactor):
            out_Adjust.append(set(x))
    
    return out_Adjust

#@ignore_warnings(category=ConvergenceWarning)
# def EstimateMarkovBoundary(X, Y, CV = True):
#     if CV:
#         m = LassoCV(n_alphas=50, fit_intercept=False, cv = 3, tol = 0.01).fit(X, Y)
#     else:
#         m = Lasso(alpha = 0.5, fit_intercept=False).fit(X, Y)
#     NonZero = np.sum(np.abs(m.coef_) > 0)
#     if NonZero <= 10:
#         out = np.where(np.abs(m.coef_) > 0)[0].tolist()
#     else:
#         out = np.argsort(np.abs(m.coef_))[-10:]

#     return out
def getblanket(X, Y):
    c = lasso_path(X, Y, n_alphas = 100, tol = 0.01)[1]
    cz = np.apply_along_axis(np.sum, 0, c != 0)
    def tmp(m):
        if np.sum(cz == m):
            return np.where(cz == m)
        else:
            return tmp(m - 1)
    mm = tmp(10)[0][0]
    return np.where(c[:, mm] != 0)[0].tolist()

def ICP_MB(E, X, Y, alpha):
    MB = getblanket(X, Y)
    IP = []
    for S in powerset(MB):
        if len(IP) >0:
            if np.sum([s <= S for s in IP]) >= 1:
                continue
        if JP_test(E, X, Y, list(S)) >= alpha:
            IP.append(set(S))
    ICP = set.intersection(*IP) if len(IP) > 0 else set()

    return ICP


def ICP(E, X, Y, alpha):
    if X.shape[1] > 10:
        return ICP_MB(E, X, Y, alpha, CV = True)
    
    EmptyInvariant = JP_test(E, X, Y, None) >= 0.05
    if EmptyInvariant:
        return set()

    IP = []
    for S in powerset(range(X.shape[1])):
        if len(IP) >0:
            if np.sum([s <= S for s in IP]) >= 1:
                continue
        if JP_test(E, X, Y, list(S)) >= alpha:
            IP.append(set(S))
    ICP = set.intersection(*IP) if len(IP) > 0 else set()

    return ICP


def EstimateMaxMinSets(d, B, Interventionsrange = None, p_range = None, MaxUpdate = 100000):
    """
    Estimates the largest number of possible 
    """

    if Interventionsrange is None:
        IntSampler = lambda: random.choice([i + 1 for i in range(d)])
    else:
        IntSampler = lambda: random.choice(Interventionsrange)
    if p_range is None:
        p_sampler = lambda: random.uniform(0, 1)
    else:
        p_sampler = lambda: random.uniform(*p_range)


    def SampleDAG_ForMaxMin(d, N_Interventions, p_conn):
        """
        Copy of the function SampleDAG, but changed to always have Y last in a causal ordering
        """

        if (N_Interventions >= d + 1) | (N_Interventions < 1):
            raise ValueError('N_interventions must be in {1, ..., d}')

        PossibleY = set()
        try_counter = 0
        try_max = 50000

        while len(PossibleY) == 0 & try_counter <= try_max:
            A = np.random.choice(
                [0, 1],
                (d + 1, d + 1),
                p=[1 - p_conn, p_conn]
            )
            A[np.tril_indices(d + 1)] = 0
            IntPos = np.random.choice(
                [i + 1 for i in range(d + 1)],
                N_Interventions,
                replace=False
            )
            Interventions = [1 if i in IntPos else 0 for i in range(d + 2)]
            A_ = (
                np.append(
                    np.array(Interventions).reshape(1, -1),
                    np.append(
                        np.zeros((d + 1, 1)),
                        A,
                        axis=1
                    ),
                    axis=0
                )
            )
            g = nx.convert_matrix.from_numpy_array(A_, create_using=nx.DiGraph)
            PossibleY = nx.descendants(g, 0) - set(g.successors(0))

        if try_counter >= try_max:
            print("Graph could not be sampled in ", try_max, " attempts.")
            return

        # Ynode = random.choice(list(PossibleY))
        Ynode = d + 1

        dict1 = {0: 'E', Ynode: 'Y'}
        dict2 = dict(zip(
            [x + 1 for x in range(d + 1) if x + 1 != Ynode],
            sorted([x for x in range(d + 1)])
        ))
        dict1.update(dict2)

        g_ = nx.relabel_nodes(g, dict1)
        return g_

    def MaxMinSet(i = None):
        g = SampleDAG_ForMaxMin(d, IntSampler, p_sampler)
        MIP = get_MIP_dagitty(g)
        out = len(MIP)
        return out

    currentmax = 0
    LastUpdate = 0
    for i in range(B):
        print("\ri = ", i, end = " ", flush = True)
        tmp = MaxMinSet()
        LastUpdate += 1
        if tmp > currentmax:
            currentmax = tmp
            LastUpdate = 0
            print("Current max = ", currentmax)
        if LastUpdate > MaxUpdate:
            print('\nNo updates found in the last ', MaxUpdate, ' steps, ending search')
            break
    
    return currentmax