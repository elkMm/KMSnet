'''Utility functions to generate KMS connectivity datasets 
and read the generated files.'''
import numpy as np 
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt 
import csv
from src.measures import kms_connectivity_significance
from src.states import critical_inverse_temperature

NITER = 1000

def generate_kms_connectivity(graph, beta_factor, sources, targets, s_name='', t_name='', removed_nodes=None, removed_edges=None, removed_edges_name='', n_iter=NITER, fname=None):
    '''Generate KMS directed connections from nodes in sources to nodes in targets, with their respective p-values.
    
    Parameters
    ----------
    graph :  a MultiDiGraph object
    beta_factor :  float
        Factor to be multiplied with the critical inverse temperature of the graph.
    sources, targets : array
        List of nodes between which the KMS connections are to be computed
    s_name, t_name : str
        Names of the sources and targets list.
    removed_nodes : array of str of int, optional, default: None
        If given, the nodes in the list will be removed from the original graph, and all computations will be made on the resulting graph.
    removed_edges : array of tuples, optional, default: None
        If given, every edge in the list will be removed from the original graph, and all computations will be made on the resulting graph.
    removed_edges_name : str, optional
    n_iter : int, default : 1000
    fname : str
        File name and path to store the csv file


    Returns
    -------
    A csv file with 4 columns: neuron1, neuron2, kms_weight, p-value
    '''

    # bc = critical_inverse_temperature(graph)
    
    file_name = f'{s_name}_to_{t_name}_kms_connnect_'

    if removed_nodes != None:
        graph.remove_nodes_from(removed_nodes)
        targets = [n for n in targets if n in list(graph.nodes)]
        sources = [n for n in sources if n not in removed_nodes]
        file_name += f'({removed_nodes}-rmvd)_'

    if removed_edges != None:
        graph.remove_edges_from(removed_edges)
        file_name += f'(edges-{removed_edges_name}-rmvd)_'

    bc = critical_inverse_temperature(graph)

    beta = beta_factor * bc 
    

    file_name += f'{beta_factor}xbeta_c_{n_iter}-iters.csv'

    if fname == None:
        fname = file_name

    con_sig = kms_connectivity_significance(graph, sources, targets, beta, n_iter=n_iter)

    kms_conn_f = open(fname, 'w', newline='')
    writer = csv.writer(kms_conn_f, delimiter=',')
    writer.writerow([f'# KMS connectivity from {s_name} to {t_name} nodes at inverse temperature of {beta_factor}xbeta_c = {beta}; the p-values were calculated over {n_iter} randomly generated graphs with same degree sequence. Surgery: removed nodes: {removed_nodes}, Removed edges: {removed_edges_name}'])
    writer.writerow(['source','target','kms_weight','p-value'])

    for k in con_sig:
        con_list = con_sig[k]
        for link in con_list:
            row = [str(k),str(link[0]),float(link[1]),float(link[2])]
            writer.writerow(row)



        





