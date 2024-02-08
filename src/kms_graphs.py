'''Base functions to construct the KMS directed weighted graphs
from a directed multigraph.'''
import networkx as nx
from .states import *
from .utils import matrix_thresholded_from_ratio, column_stochastic_matrix_thresholded

def beta_kms_digraph(graph, beta, entropy_ratio=.3, w_ratio=.2):
    '''Returns the beta-KMS digraph associated to the multigraph at
    inverse temperature beta. 

    This is a the weighted digraph whose adjacency matrix is the (thresholded)
    beta-KMS emittance matrix.'''

    nodes, KMS = kms_emittance(graph, beta)

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




