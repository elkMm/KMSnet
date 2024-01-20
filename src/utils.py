import numpy as np
from scipy import stats
import networkx as nx


def spectral_radius(matrix):
    eigenvalues, _ = np.linalg.eig(matrix)
    return max(np.abs(eigenvalues))


def adjacency_matrix(graph, nodes=None):
    """
    Compute the adjacency matrix of a directed multigraph.

    Parameters:
    -----------
    graph: NetworkX directed multigraph

    Returns:
    A: NumPy array representing the adjacency matrix
    """
    if nodes is None:
        nodes = list(graph.nodes())
    # edge_list = list(graph.edges)
    A = nx.to_numpy_array(graph, nodelist=nodes, dtype=int)
    # adjacency_matrix = nx.to_pandas_adjacency(graph).reindex(index=nodes, columns=nodes)
    # N = len(nodes)
    # A = np.zeros((N, N))
    # for i, start_node in enumerate(nodes):
    #     for j, end_node in enumerate(nodes):
    #         edges = all_edges(edge_list, start_node, end_node)
    #         A[i][j] = len(edges)

    return A


def all_edges(edge_list, start_node, end_node):
    '''Returns all the edges starting from the given node to another end node
    in a directed multigraph.
    '''
    # E = list(graph.edges)

    try:
        edges = [e for e in edge_list if (e[0] == start_node) and (e[1] == end_node)]
    except:
        edges = []

    return edges

def get_out_edges(graph, start_node):
    '''Returns all the edges from a given node.'''
    try:
        edges = [edge for edge in graph.edges if edge[0] == start_node]
    except:
        edges = []

    return edges


def all_paths(graph, start_node):
    '''
    Returns all the paths from a node in a directed multigraph.

    Parameters
    ----------
    graph : NetworkX directed multigraph

    start_node : int or str
    
    visited: None or set
        Nodes visited.
    traveled: None or set
        Set representing the traveled edges.
    path : list of 3-tuples

    Returns
    -------
    paths : list

    Notes
    -----
    A path is represented as a list of edges, where an edge
    is represented by a 3-tuple in order to take into account
    parallel edges. 
    '''
    
    out_edges = get_out_edges(graph, start_node)

    paths = []
    for out_e in out_edges:
        paths += all_paths_starting_from_edge(graph, out_e)

    return paths



def all_paths_starting_from_edge(graph, start_edge, traveled=None, path=None):

    if traveled is None:
        traveled = set()
    if path is None:
        path = []

    traveled.add(start_edge)

    path = path + [start_edge]

    paths = [path]
    next_node = start_edge[1]

    for out_edge in get_out_edges(graph, next_node):
        if out_edge not in traveled:
            new_paths = all_paths_starting_from_edge(graph, out_edge, traveled.copy(), path.copy())
            paths.extend(new_paths)

    return paths


def vector_norm(vector):
    
    s = sum([x*x for x in vector])
    return np.sqrt(s)

    


def cdf(sample, x, sort = False):
    # Sorts the sample, if unsorted
    if sort:
        sample.sort()
    # Counts how many observations are below x
    cdf = sum(sample <= x)
    # Divides by the total number of observations
    cdf = cdf / len(sample)
    return cdf


def ks_2samp(sample1, sample2):
    # Gets all observations
    observations = np.concatenate((sample1, sample2))
    observations.sort()
    # Sorts the samples
    sample1.sort()
    sample2.sort()
    # Evaluates the KS statistic
    D_ks = [] # KS Statistic list
    for x in observations:
        cdf_sample1 = cdf(sample = sample1, x  = x)
        cdf_sample2 = cdf(sample = sample2, x  = x)
        D_ks.append(abs(cdf_sample1 - cdf_sample2))
    ks_stat = max(D_ks)
    # Calculates the P-Value based on the two-sided test
    # The P-Value comes from the KS Distribution Survival Function (SF = 1-CDF)
    m, n = float(len(sample1)), float(len(sample2))
    en = m * n / (m + n)
    p_value = stats.kstwo.sf(ks_stat, np.round(en))
    return {"ks_stat": ks_stat, "p_value" : p_value}