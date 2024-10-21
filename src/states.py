'''Base functions to study KMS states on the C*-algebra of directed networks.'''
from itertools import product, combinations
import numpy as np
from numpy.linalg import matrix_power, norm, inv
from scipy.stats import entropy
from src.utils import *

TOL = 1e-5

def critical_inverse_temperature(graph):
    matrix = adjacency_matrix(graph)
    rc = spectral_radius(matrix)
    return np.log(rc)


def emittance_matrix(graph, beta, nodes=None):

    A = adjacency_matrix(graph, nodes=nodes)
    N = len(A)
    I = np.identity(N)

    A = np.matrix.transpose(A)
    
    B = I -  A / np.exp(1. * beta)

    B_inv = inv(B)

    return B_inv



def node_emittance(graph, node, beta):
    '''Returns the node emittance at inverse temperature beta.'''
    nodes = list(graph.nodes)
    B_inv = emittance_matrix(graph, beta, nodes=nodes)
    
    i = nodes.index(node)
    emittance = sum(B_inv[:, i])
    # for j, _ in enumerate(nodes):
    #     emittance += B_inv[i][j]
    

    return emittance

def node_emittance_variation(graph, nodelist, beta_min, beta_max, num=100):
    '''Compute the emittance of each node in the 
    nodelist and for each value of beta ranging from beta_min to beta_max.'''
    nodes = list(graph.nodes)

    interval = temperature_range(beta_min, beta_max, num=num)
    
    # if beta_min == beta_max:
    #     interval = [beta_min]
    # else:
    #     interval = list(np.linspace(beta_min, beta_max, num=num))
    
    emittance = {'range': interval} | {node: [] for node in nodelist}

    for _, T in enumerate(interval):
        matrix = emittance_matrix(graph, 1. / T, nodes=nodes)
        for u in nodelist:
            i = nodes.index(u)
            emittance[u] += [sum(matrix[:, i]),]
        emittance.copy()    

    return emittance


def emittance_vector(graph, beta):
    '''Returns the emittances of all nodes.'''

    nodes = list(graph.nodes)
    vec = [node_emittance(graph, node, beta) for _, node in enumerate(nodes)]

    # vec_normalized = vec / vector_norm(vec)
    return [nodes, vec]



def kms_emittance(graph, beta, with_feedback=True):
    '''Returns the KMS emittance matrix of the directed multigraph at inverse temperature beta.'''
    nodes = list(graph.nodes)
    N = len(nodes)
    B = emittance_matrix(graph, beta, nodes=nodes)
    # eMat = eMat.T


    Z = np.zeros((N,N))

    for j, _ in enumerate(nodes):
        Y = B[:, j]
        yv = nonzero_sum(Y)
        Zv = [x / yv for _, x in enumerate(Y)]
        Zv = [get_true_val(x, tol=TOL) for _, x in enumerate(Zv)]
        s = nonzero_sum(Zv)
        Z[:, j] = [x / s for _, x in enumerate(Zv)]

    if with_feedback == False:
        for j in range(N):
            col = Z[:, j]
            Z[:, j] = remove_ith(col, j)
    
    return nodes, Z

def kms_emittances_entropy(graph, beta):
    '''Returns the entropies of all the node KMS emittances.'''

    nodes, X = kms_emittance(graph, beta)

    return [entropy(X[:, j]) for j in range(len(nodes))]

def node_kms_emittance_profile(graph, node, beta):
    '''Gibbs profile of a node.

    Returns
    -------
    node_profile: array like
    '''

    nodes, Z = kms_emittance(graph, beta)
    i = nodes.index(node)
    node_profile = Z[:, i]

    return node_profile

def node_kms_emittance_profile_variation(graph,  nodelist, beta_min, beta_max, num=50, with_feedback=True):
    '''Return KMS emittances of nodes within a given interval.'''

    interval = temperature_range(beta_min, beta_max, num=num)
    
    KMSProfiles = {'range': interval} | {node: [] for node in nodelist}
    
    for _, T in enumerate(interval):
        beta = 1./T
        nodes, Z = kms_emittance(graph, beta, with_feedback=with_feedback)
        for u in nodelist:
            i = nodes.index(u)
            profile = Z[:, i]
            KMSProfiles[u] += [profile,]
            KMSProfiles.copy()

    return KMSProfiles


def node_kms_emittance_profile_entropy(graph, node, beta):
    node_profile = node_kms_emittance_profile(graph, node, beta)

    return entropy(node_profile)



def node_kms_emittance_profile_entropy_range(graph, nodelist, beta_min, beta_max, num=50, with_feedback=True):
    '''Compute the entropy of Gibbs profiles for each node in the 
    nodelist and for each value of beta in the range.'''
    # if beta_min == beta_max:
    #     interval = [1./beta_min]
    # else:
    #     interval = list(np.linspace(1./beta_max, 1./beta_min, num=num))
    
    # H = {'range': interval} | {node: [] for node in nodelist}
    
    # for _, T in enumerate(interval):
    #     beta = 1./T
    #     nodes, Z = kms_emittance(graph, beta)
    #     for u in nodelist:
    #         i = nodes.index(u)
    #         profile = Z[:, i]
    #         # profile[i] = 0.
    #         # s = sum(profile)
    #         # if s == 0.:
    #         #     s = 1.
    #         # profile = [x / float(s) for _, x in enumerate(profile)]
    #         H[u] += [entropy(profile),]
    #         H.copy()
    KMS = node_kms_emittance_profile_variation(graph, nodelist, beta_min, beta_max, num=num, with_feedback=with_feedback)
    interval = KMS['range']

    H = {'range': interval} | {node: [] for node in nodelist}

    for i, _ in enumerate(interval):
        for u in nodelist:
            profile = KMS[u][i]
            H[u] += [entropy(profile),]
            H.copy()
    return H


def node_kms_emittance_profile_diversity_range(graph, nodelist, beta_min, beta_max, num=50):
    '''Compute the diversity of Gibbs profiles for each node in the 
    nodelist and for each value of beta in the range.'''
    H = node_kms_emittance_profile_entropy_range(graph, nodelist, beta_min, beta_max, num=num)

    diversity = {'range': H['range']} | {node: [np.exp(x) for _, x in enumerate(H[node])] for node in nodelist}

    return diversity

def avg_node_kms_emittance(graph, beta):
    V = graph.nodes
    N = len(V)
    Z = kms_emittance(graph, beta)[1]
    X_avg = np.sum(Z[:, i] for i, _ in enumerate(V)) / N
    
    return V, X_avg


def avg_node_kms_emittance_profile_variation(graph, beta_min, beta_max, num=50):
    '''Returns the variation average Gibbs node profile for each beta in the range.'''
    interval = temperature_range(beta_min, beta_max, num=num)
    
    V = list(graph.nodes)
    N = len(V)

    variation = {
        'nodes': V,
        'range': interval,
        'avg_node_profiles': []
    }

    for _, T in enumerate(interval):
        beta = 1./T
        Z = kms_emittance(graph, beta)[1]
        X_avg = np.sum(Z[:, i] for i, _ in enumerate(V)) / N

        variation['avg_node_profiles'] += [X_avg,]
        variation.copy()

    return variation

def avg_reception_probability_variation(graph, nodelist, beta_min, beta_max, num=50):

    profile_variation_avg = avg_node_kms_emittance_profile_variation(graph, beta_min, beta_max, num=num)
    interval = profile_variation_avg['range']
    V = profile_variation_avg['nodes']
    profile_avgs = profile_variation_avg['avg_node_profiles']

    reception_prob = {'range': interval} | {'avg_reception': {node: [] for node in nodelist}}

    for step, _ in enumerate(interval):
        profile = profile_avgs[step]
        for n in nodelist:
            i = V.index(n)
            recep = profile[i]
            reception_prob['avg_reception'][n] += [recep,]
            reception_prob.copy()

    return reception_prob 


def node_reception_profile(graph, node, beta):
    '''Probabilities of reception of the node 
    at inverse temperature beta.'''

    nodes, Z = kms_emittance(graph, beta)
    i = nodes.index(node)
    reception_profile = Z[i, :] # /(1. * len(nodes))

    return reception_profile

def node_kms_receptance_profile_variation(graph,  nodelist, beta_min, beta_max, num=50):
    '''Return KMS receptances of nodes within a given interval.'''

    interval = temperature_range(beta_min, beta_max, num=num)
    
    KMSRecep = {'range': interval} | {node: [] for node in nodelist}
    
    for _, T in enumerate(interval):
        beta = 1./T
        nodes, Z = kms_emittance(graph, beta)
        for u in nodelist:
            i = nodes.index(u)
            profile = Z[i, :]
            KMSRecep[u] += [profile,]
            KMSRecep.copy()

    return KMSRecep


def KMS_emittance_dist(graph, beta):
    '''Computes the KMS states defined by the the emittance distribution at a given temperature.

    Parameters
    ----------
    graph : a networkx directed multigraph
    beta : float

    Returns
    -------
    state : array
    
    '''
    emittance = emittance_vector(graph, beta)

    nodes = emittance[0]
    vec = emittance[1]

    vec = vec / sum(x*x for x in vec)

    # normalized_vec = [x / vector_norm(vec) for _, x in enumerate(vec)]

    # A = adjacency_matrix(graph, nodes=nodes)
    # N = len(A)
    # I = np.identity(N)
    
    # B = I - A / np.exp(beta)

    # B_inv = inv(B)

    B_inv = emittance_matrix(graph, beta)

    X = np.dot(B_inv, vec)

    return [nodes, X]


def KMS_emittance_dist_entropy_variation(graph, beta_min, beta_max, num=50):
    interval = temperature_range(beta_min, beta_max, num=num)

    H = []    
    
    for _, T in enumerate(interval):
        beta = 1./T
        emittance = emittance_vector(graph, beta)

        nodes = emittance[0]
        vec = emittance[1]
        vec = vec / sum(x*x for x in vec)
        B_inv = emittance_matrix(graph, beta)

        X = np.dot(B_inv, vec)

        H += [entropy(X), ]

    return {'range': interval, 'entropy': H}


def node_structural_connectivity(graph, nodelist=None):
    nodes = list(graph.nodes)

    if nodelist == None:
        nodelist = nodes
    R = out_deg_ratio_matrix(graph, nodes=nodes)
    SS = {}
    for i, v in enumerate(nodelist):
        SS[v] = remove_ith(R[:, i], i)
        SS.copy()

    return nodes, SS
    
def node_structural_entropy(graph, nodelist=None):
    '''
    Returns the entropy of the column corresponding to 
    each node of nodelist in the out-degree-ratio matrix
    obrained by the function out_deg_ratio_matrix of the graph.
    '''
    # nodes = list(graph.nodes)

    # if nodelist == None:
    #     nodelist = nodes
    # R = out_deg_ratio_matrix(graph, nodes=nodes)
    SS = node_structural_connectivity(graph, nodelist=nodelist)[1]
    H = {}
    for _, v in enumerate(nodelist):
        d = SS[v]
        H[v] = entropy(d)
        H.copy()

    return H


    
    


