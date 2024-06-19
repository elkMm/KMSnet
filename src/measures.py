'''Coeffients based on KMS states'''
from itertools import combinations
import numpy as np
from distinctipy import get_colors
from src.utils import *
from src.states import kms_emittance, node_structural_connectivity
from src.generators import random_sample_from_deg_seq



def neural_integration_coefficient(graph, nodelist, beta_min, beta_max, num=50):
    '''Calculate the Neural Integration Coefficient (NIC) of each node in nodelist and for
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
            Coef[u] += [1. - float(Zuu)/d, ]
            Coef.copy()

    return Coef

def NIC_ranking(graph, beta, nodelist=None):
    '''Ranking of nodes by their NIC. 
    
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
        # d = nonzero_sum(Z[i, :])
        coefs[u] = round(1. - float(Zuu), 6)
        coefs.copy()
    
    # sort the dict by values
    coefs = sorted(coefs.items(), key=lambda x : x[1], reverse=True)

    return {c[0]: c[1] for c in coefs}


def node_total_kms_receptance_variation(graph, nodelist, beta_min, beta_max, num=50):
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


def kms_receptance_ranking(graph, beta, nodelist=None, averaging=False, with_feedback=None):
    '''Ranking of the nodes in nodelist based on the beta-KMS receptance.'''
    if nodelist == None:
        nodelist = list(graph.nodes)

    nodes, Z = kms_emittance(graph, beta)
    N = len(nodes)
    weights = {}
    for u in nodelist:
        i = nodes.index(u)
        recep_prof = Z[i, :]
        if with_feedback == None:
            recep_prof[i] = 0.
        s = sum(recep_prof)
        if averaging == True:
            s = s / N
        weights[u] = float(round(s, 6))
        weights.copy()

    # Sort the weights by values in descending order
    weights = sorted(weights.items(), key=lambda x : x[1], reverse=True)

    return {w[0]: w[1] for w in weights}


def NEP_entropy_ranking(graph, beta, nodelist=None, with_feedback=True):
    '''Ranking of the nodes in nodelist based on the entropy of the beta-KMS emittances.'''
    if nodelist == None:
        nodelist = list(graph.nodes)

    nodes, Z = kms_emittance(graph, beta)
    weights = {}
    for u in nodelist:
        i = nodes.index(u)
        profile = Z[:, i]
        if with_feedback == False:
            profile = remove_ith(profile, i)
        weights[u] = float(round(entropy(profile), 6))
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

def structure_function_divergence(graph, nodelist, beta):
    '''Calculate the divergence between the structural connectivity vector of each node in nodelist and the corresponding KMS state at the given inverse temperature beta.

    This divergence is defined as 1 minus the Fidelity of both distributions.
    '''

    vertices, SS = node_structural_connectivity(graph)
    nodes, Z = kms_emittance(graph, beta)
    DIV = []
    for v in nodelist:
        v_ind = vertices.index(v)
        i = nodes.index(v)
        struc = remove_ith(SS[v], v_ind)
        prof = remove_ith(Z[:, i], i)
        d = 1. - fidelity(struc, prof)
        d = d * 100

        DIV.append((v, round(d, 4)))

    return DIV



def kms_connectivity_significance(graph, sources, targets, beta, n_iter=100):
    '''Calculate statistical significance of connections between nodes in sources and nodes in targets given by KMS states
    for each node in the nodelist. 

    The statistical significance is given by the p-value considering n_iter generated r
    random directed multigraphs and computing the KMS states for each of them.

    Parameters
    ----------
    graph : a MultiDiGraph object
    sources :  array
    targets : array
    beta : float
    n_iter : int, default : 100
    '''
    nodes, Z = kms_emittance(graph, beta)
    rand_sample = random_sample_from_deg_seq(graph, n_iter=n_iter)

    rand_sample_kms = {k: kms_emittance(rand_sample[k], beta)[1] for k in rand_sample}
    
    conn_sig = {n: [] for n in sources}

    for node in sources:
        i = nodes.index(node)
        Zi = remove_ith(Z[:, i], i)
        for v in [n for n in targets if n != node]:
            j = nodes.index(v)
            con_val = Zi[j]
            if con_val > 0.0:
                node_to_v_sample_con = []

                ## Select connectivities from node to v in the sample of random graphs
                for k in rand_sample_kms:
                    fZ = rand_sample_kms[k]
                    fZi = remove_ith(fZ[:, i], i)
                    node_to_v_sample_con += [fZi[j], ]
                
                p_value = bootstrap_p_value(node_to_v_sample_con, con_val, n_iter=n_iter)

                conn_sig[node] += [(v, con_val, p_value), ]

                conn_sig.copy()

    return conn_sig


def connectivity_class_symmetry_coefficient(edges, node_pairs, nodes_by_classes):
    '''Calculates the symmetry coefficient of the connectivity of the node pairs.

    Parameters
    ----------
    edges : list of 2-tuple
        All the connections over which the out-neighbors of the nodes will be selected.
    node_pairs : list of 2-tuple
        The list of pairs of nodes in which the symmetry coefficient will be computed for each pair.
    nodes_by_classes : dict
        Dictionary classifying the nodes by pre-defined classes.
    
    Definition
    ----------
    This coefficient is calculated as the Cohen Kappa coefficient on the connectivity of
    the node pair by node classes; that is, given a pair (n1, n2) of nodes, we consider the out-neighbors of n1 and n2 respectively, and classify them according the node classes given by the dictionary *nodes_by_classes*. Then ther Kappa coefficient is computed as follows. 
    
    Let C1 and C2 be the list of all classes containing the out-neighbors of n1 and n2, respectively. Let a be the number of common classes, b the number of classes in C1 not belonging to C2, c the number of elements in C2 not in C1, and d  = a + b + c. 
    Now, define p0 = a / d, and pe = ((a + b)/d) * ((a + c)/d). Then 
    kappa = (p0 - pe) / (1 - pe).
    '''

    coefs = []

    for _, node_pair in enumerate(node_pairs):
        ngb_cls = {}

        for node in node_pair:
            out_neigh = [e[1] for e in edges if e[0] == str(node)]
            out_cls = []
            for n in out_neigh:
                c = [cl for cl in nodes_by_classes if str(n) in nodes_by_classes[cl]][0]
                out_cls += [c,]
            out_cls = list(set(out_cls))

            ngb_cls[node] = out_cls
            ngb_cls.copy()

        left_cls = ngb_cls[node_pair[0]]
        right_cls = ngb_cls[node_pair[1]]

        common_cls = list(set(left_cls) & set(right_cls))
        # common_cls = [cl for cl in left_cls if cl in right_cls]
        all_cls = left_cls + right_cls
        a = len(common_cls)
        d = len(all_cls) * 1.
        # b = len([cl for cl in left_cls if cl not in common_cls])
        # c = len([cl for cl in right_cls if cl not in common_cls])

        p0 = a / d

        pe = (len(left_cls) / d) * (len(right_cls) / d)

        kappa = (p0 - pe) / (1. - pe)

        coefs += [(node_pair, round(kappa, 4)),]

    return coefs
        

            









