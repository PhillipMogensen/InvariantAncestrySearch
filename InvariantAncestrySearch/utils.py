import numpy as np
import networkx as nx
import random
from scipy.interpolate import splrep, BSpline

class DataGenerator(object):
    def __init__(self, d, N_interventions, p_conn):
        self.d = d								# Number of predictor nodes
        self.N_interventions = N_interventions  # Number of nodes to intervene on
        self.p_conn = p_conn					# Probability of connecting to edges in the graph of (X, Y)
        self.sd = 1
 

    def SampleDAG(self):
        """
        Samples a dag with d + 2 nodes (d predictors (X), E and Y) by first sampling the
        d + 1 sized subgraph of (X, Y) where each node is connected with probability p_conn.
        Then, we sample N_Interventions edges from a new node E to nodes in the graph (X, Y)
        and these edges and the root node E are added. Then, we sample the position of Y among
        all nodes that 1) are not children of E and 2) are descendants of E.
        """
        d = self.d
        N_Interventions = self.N_interventions
        p_conn = self.p_conn

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

        Ynode = random.choice(list(PossibleY))

        dict1 = {0: 'E', Ynode: 'Y'}
        dict2 = dict(zip(
            [x + 1 for x in range(d + 1) if x + 1 != Ynode],
            sorted([x for x in range(d + 1)])
        ))
        dict1.update(dict2)

        g_ = nx.relabel_nodes(g, dict1)
        self.g = g_
        return 
       
    def BuildCoefMatrix(self):
        CM = np.multiply(
            np.random.uniform(0.5, 2, (self.d + 2, self.d + 2)),
            np.random.choice([-1, 1], (self.d + 2, self.d + 2))
        )
                
        self.CM = CM
        return
        
    def MakeData(self, N):
        g = self.g
        CoefficientMatrix = self.CM 
        sd = self.sd
        noise = lambda n: np.random.normal(0, 1, n)

        d = g.order()
        A = np.array(nx.linalg.adjacency_matrix(g).todense())
        Ynode = [x for x, y in enumerate(g.nodes()) if y == 'Y'][0]

        X = np.zeros((N, d))
        X[:, 0] = np.random.choice([0, 1], N)  # For equal sized observation/intervention splits
        E = X[:, 0]

        for i in range(1, d):  # This is the main component. The data is sampled by looping over the causal order and assigning nodes one by one
            parents = list(np.where(A[:, i] == 1)[0])
            if len(parents) == 0:  # Root nodes
                X[:, i] += noise(N)
            else:
                if 0 in parents:  # Nodes which are children of E
                    tmp = 0
                    for j in [x for x in parents if x != 0]:
                        tmp += CoefficientMatrix[j, i] * X[:, j]
                    tmp2 = tmp + noise(N)
                    tmp2 *= 1 / np.std(tmp2)
                    X[:, i] += (E == 0) * (tmp2) + (E == 1) * 1
                else:  # Remaning nodes
                    tmp = 0
                    for j in parents:
                        tmp += CoefficientMatrix[j, i] * X[:, j]
                    X[:, i] += tmp + noise(N)
                    X[:, i] *= 1 / np.std(X[:, i])
            
        Y = X[:, Ynode]

        return E, np.delete(X, [0, Ynode], axis=1), Y

def SampleDAG(d, N_Interventions, p_conn):
    """
    Samples a dag with d + 2 nodes (d predictors (X), E and Y) by first sampling the
    d + 1 sized subgraph of (X, Y) where each node is connected with probability p_conn.
    Then, we sample N_Interventions edges from a new node E to nodes in the graph (X, Y)
    and these edges and the root node E are added. Then, we sample the position of Y among 
    all nodes that 1) are not children of E and 2) are descendants of E.
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
            p = [1 - p_conn, p_conn]
        )
        A[np.tril_indices(d + 1)] = 0
        IntPos = np.random.choice(
            [i + 1 for i in range(d)], 
            N_Interventions, 
            replace = False
        )
        Interventions = [1 if i in IntPos else 0 for i in range(d + 2)]
        A_ = (
            np.append(
                np.array(Interventions).reshape(1, -1),
                np.append(
                    np.zeros((d + 1, 1)),
                    A,
                    axis = 1
                ),
                axis = 0
            )
        )
        g = nx.convert_matrix.from_numpy_array(A_, create_using = nx.DiGraph)
        PossibleY = nx.descendants(g, 0) - set(g.successors(0))
    
    if try_counter >= try_max:
        print("Graph could not be sampled in ", try_max, " attempts.")
        return 

    Ynode = random.choice(list(PossibleY))
    
    dict1 = {0: 'E', Ynode: 'Y'}
    dict2 = dict(zip(
        [x + 1 for x in range(d + 1) if x + 1 != Ynode], 
        sorted([x for x in range(d + 1)])
    ))
    dict1.update(dict2)

    g_ = nx.relabel_nodes(g, dict1)

    return g_
