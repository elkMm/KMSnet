'''Base functions to construct the KMS directed weighted graphs
from a directed multigraph.'''
import networkx as nx
from itertools import product
from .states import *
from .utils import out_deg_ratio_matrix, column_stochastic_matrix_thresholded

def beta_kms_digraph(graph, beta, entropy_ratio=.3, w_ratio=.2):
    '''Returns the beta-KMS digraph associated to the multigraph at
    inverse temperature beta. 

    This is a the weighted digraph whose adjacency matrix is the (thresholded)
    beta-KMS emittance matrix.'''

    nodes, KMS = kms_matrix(graph, beta)

    # thresholding the KMS emittance matrix
    entropy_thresh, w_thresh, A = column_stochastic_matrix_thresholded(KMS, entropy_ratio=entropy_ratio, weight_ratio=w_ratio)

    # Construct the simple digraph from the thresholded KMS matrix
    G = nx.DiGraph()
    weighted_edges = []
    for j, v in enumerate(nodes):
        for i, u in enumerate(nodes):
            if (A[i][j] > 0.) and (u != v):
                weighted_edges += [(v, u, A[i][j]),]

    G.add_weighted_edges_from(weighted_edges)

    return entropy_thresh, w_thresh, G


def kms_subgraph_adj_mat(graph, beta, nodelist=None, with_feedback=False):
    '''Returns the sub-matrix corresponding to the KMS emittance profile of the subnetwork
    defined by the given nodelist.'''
    nodes, Z = kms_matrix(graph, beta, with_feedback=with_feedback)
    if nodelist == None:
        nodelist = nodes
    
    N = len(nodelist)
    W = np.zeros((N, N))

    for i, u in enumerate(nodelist):
        ii = nodes.index(u)
        for j, v in enumerate(nodelist):
            jj = nodes.index(v)
            W[i][j] = Z[ii][jj]

    # Normalize the columns of W
    # for j in range(N):
    #     col = W[:, j]
    #     W[:, j] = remove_ith(col, j)

    return nodelist, W


def kms_weighted_subgraph(graph, beta, nodelist=None, with_feedback=False, tol=0.):
    '''Returns the weighted directed subgraph obtained from the kms_subgraph_adj_mat.'''
    nodes, W = kms_subgraph_adj_mat(graph, beta, nodelist=nodelist, with_feedback=with_feedback)

    K = nx.DiGraph()
    K.add_nodes_from(nodes)
    E = []
    

    for i, target in enumerate(nodes):
        for j, source in enumerate(nodes):
            w = W[i][j]
            if w > tol:
            # if (i != j) and  (w > 0.):
                E += [(source, target, w),]
                # K.add_edge(source, target, weight=w)
    K.add_weighted_edges_from(E)

    return K

def node_kms_emittance_connectivity(graph, node, beta, tol=TOL):
    '''Returns the weighted directed subgraph obtained from the beta-KMS state defined by the specified node.'''
    nodes, Z = kms_matrix(graph, beta)
    i = nodes.index(node)
    Zv = remove_ith(Z[:, i], i)

    # # transform the range into [0,1]
    # a = float(min(Zv))
    # b = float(max(Zv))
    # d = 1./(b - a)
    # con = [(x - a) * d for _, x in enumerate(Zv)]
    con = [get_true_val(x, tol=tol) for _, x in enumerate(Zv)]

    # K = nx.DiGraph()
    # E = []
    C = {}
    for j, u in enumerate(nodes):
        w = con[j]
        if  w > 0.:
            # E += [(node, u, w),]
            C[u] = w
            C.copy()
    
    # K.add_weighted_edges_from(E)

    return C




def group_kms_emittance_connectivity(graph, beta, nodelist, P, tol=TOL):
    nodes, Z = kms_matrix(graph, beta)
    C = {}
    vec = np.zeros(len(nodes))
    for ind, node in enumerate(nodelist):
        i = nodes.index(node)
        Zv = remove_ith(Z[:, i], i)
        # Zv = Z[:, i]
        # # transform the range into [0,1]
        # a = float(min(Zv))
        # b = float(max(Zv))
        # d = 1./(b - a)
        p = P[ind]
        # con = [(x - a) * d for _, x in enumerate(Zv)]
        con = [float(p) * x for _, x in enumerate(Zv)]

        vec += np.array(con)


    for j, u in enumerate(nodes):
        w = vec[j]
        if  w > tol:
            # E += [(node, u, w),]
            C[u] = w
            C.copy()

    return C
    



def structural_subgraph_adj_matrix(graph, nodelist=None):
    '''Returns the adjacency matrix of the subgraph.
    
    This matrix is obtained from the adjacency matrix of the whole graph by selecting only the entries involving the nodes in the specified subgraph, then removing the diagonal and normalizing the columns of the resulting matrix.
    '''

    nodes = list(graph.nodes)
    N = len(nodes)

    if nodelist == None:
        nodelist = nodes

    R = out_deg_ratio_matrix(graph, nodes=nodes)
    # Normalize the columns of R
    for j in range(N):
        col = R[:, j]
        R[:, j] = remove_ith(col, j)

    M = len(nodelist)
    W = np.zeros((M, M))

    for i, u in enumerate(nodelist):
        ii = nodes.index(u)
        for k, v in enumerate(nodelist):
            kk = nodes.index(v)
            W[i][k] = R[ii][kk]

    return nodelist, W


def weighted_structural_subgraph(graph, nodelist=None):
    '''Returns the weighted directed subgraph obtained from the function structural_subgraph_adj_matrix.'''
    nodes, W = structural_subgraph_adj_matrix(graph, nodelist=nodelist)

    K = nx.DiGraph()
    K.add_nodes_from(nodes)
    E = []
    

    for i, target in enumerate(nodes):
        for j, source in enumerate(nodes):
            w = W[i][j]
            if w > 0.:
                E += [(source, target, w)]
                
    K.add_weighted_edges_from(E)

    return K

def structural_connectivity_adj_list(graph, sources):
    '''Returns theweighted  edge list of structural connectivity from 
     the nodes in sources to the all their direct neighbors. '''
    
    nodes, S = node_structural_connectivity(graph)
    adj_list = []

    for v in sources:
        Kv = S[v]
        v_list = [(v, u, round(Kv[i], 6)) for i, u in enumerate(nodes) if (Kv[i] > 0.)]
        adj_list += v_list

    return adj_list



    
    














