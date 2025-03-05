'''
Base functions to generate certain random 
directed graphs.
'''
import random
from random import sample
from itertools import product, combinations
import numpy as np
import networkx as nx
import random_graph


def configuration_model_from_directed_multigraph(graph, seed=None):
    '''
    Returns a directed_random multigraph with the same degree sequences
    as the given directed multigraph.

    Parameters
    ----------
    graph : a MultiDiGraph object  
    '''
    nodes = list(graph.nodes)
    in_deg_seq = [graph.in_degree(n) for _, n in enumerate(nodes)] #[d[1] for d in graph.in_degree]
    out_deg_seq = [graph.out_degree(n) for _, n in enumerate(nodes)] # [d[1] for d in graph.out_degree]

    g = nx.directed_configuration_model(in_deg_seq, out_deg_seq, seed=seed)
    edges = list(g.edges)

    G = nx.MultiDiGraph()
    G.add_nodes_from(nodes)
    G.add_edges_from([(nodes[e[0]], nodes[e[1]]) for e in edges])

    return G


def random_sample_from_deg_seq(graph, n_iter=100, seed=None):
    '''Generates a sample of n_iter random directed multigraph all with 
     the same degree sequence as the given graph.

    Parameters
    ----------

    graph : a MultiDiGraph networkx object
    n_iten : int, default: 100
        Number of iterations, which is the number of random graphs
        to be generated.

    Returns
    -------
    rand_sample : dict 
        Dictionary whose keys are integers and the value of i  is the i-th random graph generated by the process.
     '''
    
    rand_sample = {}
    for iter in range(n_iter):
        rand_sample[iter] = configuration_model_from_directed_multigraph(graph, seed=seed)
        rand_sample.copy()

    return rand_sample


def conditional_random_multi_digraph(n, p, r, seed=0):
    '''
    Generates an Erdos-Rényi-like random directed graph of size n.
    
    Parameters
    ----------
    n : int
       Number of nodes
    p, r : float, must be between 0 and 1

    Notes
    -----
    The algorithm first generates a simple random directed graph without reciprocal 
    connections. In this graph, an edge exists from node v to node w with probability *p/2*.
    Next, it looks at the produced edges and add reciprocal edges
    with probability *r*.
    '''
    V = range(1, n+1)

    V_comb = combinations(V,2)

    # VV = [p for p in product(V,V) if p[0] != p[1]]

    G = nx.MultiDiGraph()
    G.add_nodes_from(V)

    E = []

    seed = np.random.default_rng(seed)

    for n0, n1 in V_comb:
        # e = random.choice([(n0, n1), (n1, n0)])
        
        if seed.random() < p:
            E += [random.choice([(n0, n1), (n1, n0)]),]
    E1 = []       
    for e in E:
        # if ((e[1], e[0]) not in E):
        if seed.random() < r:
            E1 += [(e[1], e[0]),]

    E += E1
     
    G.add_edges_from(E)

    return G

def scale_free_digraph(n, alpha=0.41, beta=0.54, gamma=0.05, delta_in=0.2, delta_out=0, seed=None):
    '''Returns a scale free directed graph
    without self-loops and no parallel edges.'''

    S = nx.scale_free_graph(n, alpha=alpha, beta=beta, gamma=gamma, delta_in=delta_in, delta_out=delta_out, seed=seed)
    
    V = S.nodes
    E = S.edges

    
    
    # initialize new directed graph 
    G = nx.MultiDiGraph()
    G.add_nodes_from([v for v in V])

    # for e in E1:
    #     n1 = e[0]
    #     n2 = e[1] 
    #     if (n1, n2) not in E:
    #         E.append((n1, n2))

    G.add_edges_from(list(set([(e[0], e[1]) for e in E if (e[0] != e[1])])))

    return G

def random_density_and_reciprocity_digraph(n, rho, r):
    '''Generates a random graph with same number nodes, same
    density and same reciprocity.

    Parameters
    ----------
    n : int
       Number of nodes.
    rho : float between 0 and 1
       Density of the directed graph.
    r : float between 0 and 1
       Probability of reciprocal connections.
    '''
    
    p = (2 * rho) / (1 + r)

    return conditional_random_multi_digraph(n, p, r)


def random_sample_from_graph(G, n_iter=int(1e6)):
    ''' Generates a directed graph from samples of random graphs
    with same degree sequence as the directed graph G. 

    Parameters
    ----------
    G : a Networkx DiGraph object
    '''
    in_deg_seq = {d[0]: d[1] for d in G.in_degree()}
    out_deg_seq = {d[0]: d[1] for d in G.out_degree()}

    deg_seq = []
    for k in in_deg_seq:
        deg_seq += [(out_deg_seq[k], in_deg_seq[k]),]
        # Here I switched the in and out degrees as they were switched over 
        # in the external code of the package `random_graph` (link below)
        # `https://github.com/jamesross2/random_graph/blob/master/src/random_graph/sample.py` 
    n = len(deg_seq)

    edges = random_graph.sample_directed_graph(degree_sequence=deg_seq, n_iter=n_iter)
    graph = nx.DiGraph()
    graph.add_nodes_from(range(n))
    graph.add_edges_from(edges)

    return graph
