'''Coeffients based on KMS states'''
from itertools import combinations
import numpy as np
from distinctipy import get_colors
from src.utils import *
from src.states import kms_emittance, node_structural_state

def beta_feedback_coef(graph, nodelist, beta_min, beta_max, num=50):
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
            profile = Z[:, i]
            Zuu = profile[i]
            d = nonzero_sum(Z[i, :])
            Coef[u] += [float(Zuu) / d, ]
            Coef.copy()

    return Coef

def node_kms_in_weight(graph, nodelist, beta_min, beta_max, num=50):
    '''Return the incoming weight for each node in nodelist and 
    for the KMS state matrix for each beta in the interval.'''

    interval = temperature_range(beta_min, beta_max, num=num)

    weights = {'range': interval} | {node: [] for node in nodelist}
    N = len(list(graph.nodes))
    
    for _, T in enumerate(interval):
        beta = 1./T 
        nodes, Z = kms_emittance(graph, beta)
        for u in nodelist:
            i = nodes.index(u)
            weight = sum(Z[i, :]) / float(N)
            weights[u] += [weight,]

    return weights



def node_to_node_receptance_ratio(graph, nodelist, beta_min, beta_max, num=50):
    '''Calculate the ratio of the KMS emittance of node u to node v by the sum of all the 
    receptance of node v.'''
