'''Base functions to study KMS states on the C*-algebra of directed networks.'''
from itertools import product, combinations
import numpy as np
from numpy.linalg import matrix_power, norm, inv
from scipy.stats import entropy
from src.utils import *

def critical_inverse_temperature(graph):
    matrix = adjacency_matrix(graph)
    rc = spectral_radius(np.matrix.transpose(matrix))
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
    if beta_min == beta_max:
        interval = [beta_min]
    else:
        interval = list(np.linspace(beta_min, beta_max, num=num))
    
    emittance = {'range': interval} | {node: [] for node in nodelist}

    for _, beta in enumerate(interval):
        matrix = emittance_matrix(graph, beta, nodes=nodes)
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



def gibbs_profile(graph, beta):
    '''Returns the Gibbs profile of the directed multigraph at inverse temperature beta.'''
    nodes = list(graph.nodes)
    N = len(nodes)
    eMat = emittance_matrix(graph, beta, nodes=nodes)
    eMat = np.matrix.transpose(eMat)

    Z = np.zeros((N,N))
    
    for i, v in enumerate(nodes):
        yv = sum([eMat[i][j] for j, _ in enumerate(nodes)])
        for k, _ in enumerate(nodes):
            Z[i][k] = eMat[i][k] / (1. * yv)

    return nodes, np.matrix.transpose(Z)

def node_gibbs_profile(graph, node, beta):
    '''Gibbs profile of a node.

    Returns
    -------
    node_profile: array like
    '''

    nodes, Z = gibbs_profile(graph, beta)
    i = nodes.index(node)
    node_profile = Z[:, i]

    return node_profile


def subnet_gibbs_profile(graph, nodelist, beta):
    '''Returns the sub-matrix correspond to the Gibbs profile of the subnetwork
    defined by the given nodelist.'''
    nodes, Z = gibbs_profile(graph, beta)
    N = len(nodelist)
    W = np.zeros((N, N))

    for i, u in enumerate(nodelist):
        ii = nodes.index(u)
        for j, v in enumerate(nodelist):
            jj = nodes.index(v)
            W[i][j] = Z[ii][jj]

    return nodelist, W

def node_profile_entropy(graph, node, beta):
    node_profile = node_gibbs_profile(graph, node, beta)

    return entropy(node_profile)

def node_profile_entropy_range(graph, nodelist, beta_min, beta_max, num=50):
    '''Compute the entropy of Gibbs profiles for each node in the 
    nodelist and for each value of beta in the range.'''
    if beta_min == beta_max:
        interval = [1./beta_min]
    else:
        interval = list(np.linspace(1./beta_max, 1./beta_min, num=num))
    
    H = {'range': interval} | {node: [] for node in nodelist}
    
    for _, T in enumerate(interval):
        beta = 1./T
        nodes, Z = gibbs_profile(graph, beta)
        for u in nodelist:
            i = nodes.index(u)
            profile = Z[:, i]
            H[u] += [entropy(profile),]
            H.copy()

    return H


def node_profile_diversity_range(graph, nodelist, beta_min, beta_max, num=50):
    '''Compute the diversity of Gibbs profiles for each node in the 
    nodelist and for each value of beta in the range.'''
    H = node_profile_entropy_range(graph, nodelist, beta_min, beta_max, num=num)

    diversity = {'range': H['range']} | {node: [np.exp(x) for _, x in enumerate(H[node])] for node in nodelist}

    return diversity

def avg_node_profile(graph, beta):
    V = graph.nodes
    N = len(V)
    Z = gibbs_profile(graph, beta)[1]
    X_avg = np.sum(Z[:, i] for i, _ in enumerate(V)) / N
    
    return V, X_avg


def avg_node_profile_variation(graph, beta_min, beta_max, num=50):
    '''Returns the variation average Gibbs node profile for each beta in the range.'''
    if beta_min == beta_max:
        interval = [1./beta_min]
    else:
        interval = np.linspace(1./beta_max, 1./beta_min, num=num)
    
    V = list(graph.nodes)
    N = len(V)

    variation = {
        'nodes': V,
        'range': interval,
        'avg_node_profiles': []
    }

    for _, T in enumerate(interval):
        beta = 1./T
        Z = gibbs_profile(graph, beta)[1]
        X_avg = np.sum(Z[:, i] for i, _ in enumerate(V)) / N

        variation['avg_node_profiles'] += [X_avg,]
        variation.copy()

    return variation

def avg_reception_probability_variation(graph, nodelist, beta_min, beta_max, num=50):

    profile_variation_avg = avg_node_profile_variation(graph, beta_min, beta_max, num=num)
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




    



    
    

def states_kl_variation(graph, nodelist, beta_min, beta_max, num=50):
    '''Return the Kulback-Liebler distance (relative entropy) between the KMS states
    corresponding to each node in nodelist.'''
    if beta_min == beta_max:
        interval = [1./beta_min]
    else:
        interval = list(np.linspace(1./beta_max, 1./beta_min, num=num))

    kl_distances = {}
    for _, T in interval:
        nodes, W = subnet_gibbs_profile(graph, nodelist, 1./T)



def node_reception_profile(graph, node, beta):
    '''Probabilities of reception of the node 
    at inverse temperature beta.'''

    nodes, Z = gibbs_profile(graph, beta)
    i = nodes.index(node)
    reception_profile = Z[i, :]/(1. * len(nodes))

    return reception_profile





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
    if beta_min == beta_max:
        interval = [1./beta_min]
    else:
        interval = list(np.linspace(1./beta_max, 1./beta_min, num=num))


    H = []    
    
    for i, T in enumerate(interval):
        beta = 1./T
        emittance = emittance_vector(graph, beta)

        nodes = emittance[0]
        vec = emittance[1]
        vec = vec / sum(x*x for x in vec)
        B_inv = emittance_matrix(graph, beta)

        X = np.dot(B_inv, vec)

        H += [entropy(X), ]

    return {'range': interval, 'entropy': H}





    
    


