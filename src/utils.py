import numpy as np
from scipy import stats
from scipy.stats import entropy
import networkx as nx

TOL = 1e-2

def conjugate_graph(G):
    '''
    Returns the conjugate of a directed multigraph.

    Notes
    -----
    The conjugate of a directed graph *G=(V,E)* has the same
    vertex set *V* and the edges are the elements of *E* with all the directions reversed. 
    '''
    V = G.nodes
    E = G.edges

    E_bar = []
    for v, u, n in E:
        E_bar += [(u, v, n),]

    G_bar = nx.MultiDiGraph()
    G_bar.add_nodes_from(V)
    G_bar.add_edges_from(E_bar)

    return G_bar


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


def out_deg_ratio_matrix(graph, nodes=None):
    '''
    Returns the matrix whose (i,j)-entry is 
    the ratio of the number of edges from j to i out of 
    the out-degree of j. 
    '''
    if nodes is None:
        nodes = list(graph.nodes)
    A = adjacency_matrix(graph, nodes=nodes)
    A = A.T
    N = len(nodes)
    R = np.zeros((N, N))
    
    for j, _ in enumerate(nodes):
        a = sum(A[:, j])
        x = float(a)
        if a == 0.:
            x = 1.
        for i, u in enumerate(nodes):
            R[i][j] = float(A[i][j]) / x
        ss = sum(R[:, j])
        if ss == 0.:
            ss = 1.
        
        R[:, j] = R[:, j] / float(ss)

    return R


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


def js_divergence(x,y):
    '''Calculate the Jensen-Shannon divergence between two probability distributions'''

    n = len(x)
    m = [(x[i] + y[i]) / 2. for i in range(n)]

    jsd = entropy(m) - (entropy(x) + entropy(y)) / 2. 

    return jsd


def fidelity(x, y):
    '''Calculate the fidelity of two probability distributions.

    Given two probability distributions x and y, their fidelity if given by

    .. math::

       F(x,y) = \\left(\\sum\\sqrt{x_iy_i}\\right)^2
    
    '''
    F = 0.
    for i in range(len(x)):
        F += np.sqrt(x[i] * y[i])

    return F ** 2



def get_threshold_from_ratio(array, ratio=.8):
    '''Finds threshold from an array based on a percentage.'''

    array = sorted(array, reverse=True)
    n = len(array)
    m = int(ratio * n)
    v1 = array[m]
    v2 = array[m+1]
    return (v1 + v2)/2.

def matrix_thresholded_from_ratio(matrix, ratio=.8, rescale_columns=True):
    '''Returns thresholded a mtrix from a ratio.'''
    n, m = matrix.shape
    array = []
    for l in np.matrix(matrix).tolist():
        array += l
    
    threshold = get_threshold_from_ratio(array, ratio=ratio)
    A = np.zeros((n, m))

    for i in range(n):
        for j in range(m):
            el = matrix[i][j]
            if (i != j) and (el >= threshold):
                A[i][j] = el 
            else:
                A[i][j] = 0.

    # normalize columns
    if rescale_columns:
        for k in range(m):
            d = sum(list(A[:, k]))
            if d > 0:
                numer = d
            else:
                numer = 1.
            for i in range(n):
                A[:, k][i] = A[i][k] / numer

    return threshold, A


def column_stochastic_matrix_thresholded(matrix, entropy_ratio=.8, weight_ratio=.2):
    '''Returns a spare matrix obtained by first 
    thresholding the cloumn stochastic matrix based on the Shannon entropy  
    of the columns, then threshold of the weights of the resulted matrix.'''

    n, m = matrix.shape

    col_entr = [entropy(matrix[:, i]) for i in range(m)]
    entropy_thresh = get_threshold_from_ratio(col_entr, ratio=entropy_ratio)

    # Select columns based on entropy
    A1 = np.zeros((n, m))
    
    for j in range(m):
        c = matrix[:, j]
        if entropy(c) >= entropy_thresh:
            A1[:, j] = c 
        else:
            A1[:, j] = np.zeros(n)
    
    # threshold A1 with ratio weight_ratio
    w_thresh, A = matrix_thresholded_from_ratio(A1, ratio=weight_ratio)

    return entropy_thresh, w_thresh, A



def is_qual(val1, val2, tol=TOL):
    v = abs(val1 - val2)
    if float(v) <= tol:
        return True
    else:
        return False


def get_true_val(val, tol=1e-5):
    x = 0.
    if val > tol:
        x = val
    return x    

def nonzero_sum(x):
    '''Returns sum of the array or 1 if the sum is zero.'''
    s = sum(x)
    if s == 0.:
        s = 1.
    return s

def remove_ith(x, i):
    '''Remove ith element in array and then normalize.'''
    x[i] = 0.
    s = nonzero_sum(x)
    x = [a / float(s) for _, a in enumerate(x)]

    return x

def temperature_range(beta_min, beta_max, num=50):
    if beta_min == beta_max:
        interval = [1./beta_min]
    else:
        interval = list(np.linspace(1./beta_max, 1./beta_min, num=num))
    return interval
       