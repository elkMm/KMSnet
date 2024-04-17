'''Coeffients based on KMS states'''
from itertools import combinations
import numpy as np
from distinctipy import get_colors
from src.utils import *
from src.states import kms_emittance, node_structural_state

def beta_feedback_coef_variation(graph, nodelist, beta_min, beta_max, num=50):
    '''Calculate the beta-feedback coefficient of each node in nodelist and for
    every value of beta in the specified interval between beta_min and beta_max.'''

    interval = temperature_range(beta_min, beta_max, num=num)

    # if beta_min == beta_max:
    #     interval = [1./beta_min]
    #     # interval = [beta_max]
    # else:
    #     interval = list(np.linspace(1./beta_max, 1./beta_min, num=num))
    #     # interval = list(np.linspace(beta_min, beta_max, num=num))
    
    Coef = {'range': interval} | {node: [] for node in nodelist}
    
    for _, T in enumerate(interval):
        beta = 1./T 
        nodes, Z = kms_emittance(graph, beta)
        for u in nodelist:
            i = nodes.index(u)
            recep_profile = Z[i, :]
            Zuu = recep_profile[i]
            d = nonzero_sum(recep_profile)
            Coef[u] += [float(Zuu) / d, ]
            Coef.copy()

    return Coef

def beta_kms_feedback_coef_ranking(graph, beta, nodelist=None):
    '''Ranking of nodes by their KMS feedback coefficients. 
    
    The output is as a dictionary with nodes in nodelist as keys
    and the KMS feedback coefficient at inverse temperature beta as values.'''
    if nodelist == None:
        nodelist = list(graph.nodes)

    nodes, Z = kms_emittance(graph, beta)
    coefs = {}

    for u in nodelist:
        i = nodes.index(u)
        profile = Z[:, i]
        Zuu = profile[i]
        d = nonzero_sum(Z[i, :])
        coefs[u] = round(d / float(Zuu) / d, 6)
        coefs.copy()
    
    # sort the dict by values
    coefs = sorted(coefs.items(), key=lambda x : x[1], reverse=True)

    return {c[0]: c[1] for c in coefs}


def node_kms_receptance_variation(graph, nodelist, beta_min, beta_max, num=50):
    '''Return the incoming weight for each node in nodelist and 
    for the KMS state matrix for each beta in the interval.'''

    interval = temperature_range(beta_min, beta_max, num=num)

    weights = {'range': interval} | {node: [] for node in nodelist}
    # N = len(list(graph.nodes))
    
    for _, T in enumerate(interval):
        beta = 1./T 
        nodes, Z = kms_emittance(graph, beta)
        for u in nodelist:
            i = nodes.index(u)
            Z[i, :][i] = 0.
            # weight = sum(Z[i, :])
            weights[u] += [sum(Z[i, :]),]
            weights.copy()

    return weights


def beta_kms_emittance_ranking(graph, beta, nodelist=None):
    '''Ranking of the nodes in nodelist based on the beta-KMS emittances.'''
    if nodelist == None:
        nodelist = list(graph.nodes)

    nodes, Z = kms_emittance(graph, beta)
    weights = {}
    for u in nodelist:
        i = nodes.index(u)
        Z[i, :][i] = 0.
        weights[u] = float(round(sum(Z[:, i]), 6))
        weights.copy()

    # Sort the weights by values in descending order
    weights = sorted(weights.items(), key=lambda x : x[1], reverse=True)


    return {w[0]: w[1] for w in weights}


def beta_kms_receptance_ranking(graph, beta, nodelist=None):
    '''Ranking of the nodes in nodelist based on the beta-KMS inweights.'''
    if nodelist == None:
        nodelist = list(graph.nodes)

    nodes, Z = kms_emittance(graph, beta)
    weights = {}
    for u in nodelist:
        i = nodes.index(u)
        Z[i, :][i] = 0.
        weights[u] = float(round(sum(Z[i, :]), 6))
        weights.copy()

    # Sort the weights by values in descending order
    weights = sorted(weights.items(), key=lambda x : x[1], reverse=True)


    return {w[0]: w[1] for w in weights}
    



def node_to_node_kms_flow_stream(graph, node, nodelist, beta_min, beta_max, num=50):
    '''Calculate the variation of the KMS emittance of the node to each node in nodelist.'''
    interval = temperature_range(beta_min, beta_max, num=num)

    streams = {'range': interval} | {u: [] for u in nodelist}

    for _, T in enumerate(interval):
        beta = 1./T 
        nodes, Z = kms_emittance(graph, beta)
        i = nodes.index(node)
        Zv = remove_ith(Z[:, i], i)
        for u in nodelist:
            j = nodes.index(u)
            streams[u] += [Zv[j],]
            streams.copy()

    return streams



