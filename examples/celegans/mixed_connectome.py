import os
import sys
import re
import networkx as nx
import matplotlib.pyplot as plt 
import numpy as np 
import scipy.stats as stats
import pandas as pd
import csv
from openpyxl import load_workbook


ex_dir = os.path.dirname(__file__)
cn_module_dir = os.path.join(ex_dir, '../..')
sys.path.append( cn_module_dir )

from src.visuals import *
from src.utils import *
from src.states import *
from src.generators import *
from src.plotting import *
from src.kms_graphs import *
from src.measures import *
import seaborn as sns
# import fitter
# from fitter import Fitter, get_common_distributions
from src.gen_colors import *
from neuron_positions import *
from src.read_write import *


from connectome import neuron_positions, neurons, neuron_by_classes, neuron_cls, node_colors, node_labels, CIRCUITS, blue_sky


default_edge_color = '#bfb4b2'
func_edge_color = '#0394ad'

# retrieve synapses from Cook
# varsh_f = '../data/NeuronConnect.xls'

# varsh_df = pd.read_excel(varsh_f)

# cook_f = open('../data/cook/herm_somatic_connectome_updated.csv', 'r', newline='')


# cook_rows = [line.strip().split(',') for line in cook_f.readlines()[2:]]

# cook_synapses = []

# for r in cook_rows:
#     if r[2] == 'electrical':
#         cook_synapses += [(r[0], r[1], r[2]), (r[1], r[0], r[2]),]
#     elif r[2] == 'chemical':
#         cook_synapses += [(r[0], r[1], r[2]),]
         

# # Add Cook data to Varshney if they were not accounted in the latter
# varsh_synapses = []
# for i in varsh_df.index:
#     row = [str(varsh_df['Neuron 1'][i]), str(varsh_df['Neuron 2'][i]), str(varsh_df['Type'][i]), int(varsh_df['Nbr'][i])]

#     if row[2] in ['S', 'Sp']:
#         varsh_synapses += [(row[0], row[1], 'chemical'),] * row[3]
#     elif row[2] in ['EJ']:
#         varsh_synapses += [(row[0], row[1], 'electrical'),] * row[3]
   

# print(len([s for s in cook_synapses if s not in varsh_synapses]))

# mixed_data = varsh_synapses

# cook_not_varsh = [s for s in cook_synapses if s not in varsh_synapses]

# # for syn in cook_not_varsh:
# #     if syn[2] == 'electrical':
# #         cook_not_varsh += [(syn[1], syn[0], syn[2]),]
#     # else:
#     #     pass
# mixed_data = varsh_synapses + cook_not_varsh

# print(len(mixed_data))




## Save the synapses from Cook and Varsheny in a unified csv file
# mixed_conn_f = open('../data/mixed/herm_somatic_connectome.csv', 'w', newline='')
# writer = csv.writer(mixed_conn_f, delimiter=',')
# writer.writerow(['# Chemical synapses and Gap junctions in the C. elegans'])
# writer.writerow(['# From dataset used by Varshney et al. completed by 3900 unique connections Cook et al. new dataset'])
# writer.writerow(['# This updated dataset of the somatic network contains 12071 unique connections'])

# for syn in mixed_data:
#     row = [str(syn[0]),str(syn[1]),str(syn[2])]
#     writer.writerow(row)

# print(varsh_synapses[:10])


### WORKING WITH THE UPDATED DATASET
up_conn_f = open('../data/mixed/herm_somatic_connectome.csv', 'r', newline='')
syn_rows = [line.strip().split(',') for line in up_conn_f.readlines()[3:]]

updt_synapses = []

for r in syn_rows:

    updt_synapses += [(r[0], r[1]),] 

G = nx.MultiDiGraph()
G.add_edges_from(updt_synapses)

# neurons = list(G.nodes)

### CRITICAL TEMPERATURE

bc = critical_inverse_temperature(G) # 4.295757002698898

# betabar_c = critical_inverse_temperature(S_bar)

## Ground states are around: 10 * beta_c + 1.7763
# beta_gd = 10 * beta_c + (52.265/100) * beta_c
# beta_gd = 10.5 * beta_c

bmin = bc + .001

beta = 3.5*bc 

bf = 1.07 * bc
bs = 2.5 * bc # = 10.739392506747246



# #### Draw structural circuits
# name = 'ExtendedThermotaxis'
# # for name in CIRCUITS:
# V = CIRCUITS[name]
# K = weighted_structural_subgraph(G, nodelist=V)
# draw_weighted_digraph(K, pos={n: POSITIONS['thermotaxis'][n] for n in K.nodes}, node_size=2.6, node_colors={n: node_colors[n] for n in K.nodes}, node_labels={n: node_labels[n] for n in K.nodes}, edgecolor=default_edge_color, font_size=12, figsize=(10,6))
# plt.show()
# plt.savefig(f'./results/newConnectome/nets/struc/{name}_circuit.pdf', dpi=300, bbox_inches='tight')
# # plt.close()

# struc_conn_f = open(f'./results/data/struc_connectivity/{name}_structural_connectivity.csv', 'w', newline='')
# writer = csv.writer(struc_conn_f, delimiter=',')
# writer.writerow([f'# Structural connectivity of the extended Thermotaxis circuit'])
# writer.writerow(['source,target,weight'])

# for we in list(K.edges(data=True)):
#     e = [str(we[0]),str(we[1]),float(we[2]['weight'])]
#     writer.writerow(e)

# for k in con_sig:
#     con_list = con_sig[k]
#     for link in con_list:
#         row = [str(k),str(link[0]),float(link[1]),float(link[2])]
#         writer.writerow(row)

# print(K.edges(data=True))



##### Save structural connecvity adj list ########
# name = 'AIY'
# sources = CIRCUITS[name]
# adj_list = structural_connectivity_adj_list(G, sources)

# struc_conn_f = open(f'./results/data/struc_connectivity/{name}_structural_connectivity.csv', 'w', newline='')
# writer = csv.writer(struc_conn_f, delimiter=',')
# writer.writerow([f'# Structural connectivity of {name}'])
# writer.writerow(['source','target','weight'])

# for we in adj_list:
#     e = [str(we[0]),str(we[1]),float(we[2])]
#     writer.writerow(e)

# for k in con_sig:
#     con_list = con_sig[k]
#     for link in con_list:
#         row = [str(k),str(link[0]),float(link[1]),float(link[2])]
#         writer.writerow(row)



# KMS subgraphs
# b_factors = [.9, .98, 1.001, 1.01, 1.06, 1.1, 1.2, 1.5, 1.8, 1.9, 2., 2.1, 2.4, 2.6, 2.8, 3., 3.1, 3.2, 3.5, 3.8, 4.] 
# # b_factors = [1.1]
# b_factor = 1.06
# name = 'ExtendedThermotaxis'
# V = CIRCUITS[name]
# # for b_factor in b_factors:
# #     b = b_factor * beta_c
# #     # V = olfactory_thermo_neurons
# K = kms_weighted_subgraph(G, b_factor * bc, nodelist=V)

# # # # print(list(K.edges(data=True))[0][2]['weight'])
# draw_weighted_digraph(K, pos={n: POSITIONS['thermotaxis'][n] for n in K.nodes}, node_size=1.6, node_colors={n: node_colors[n] for n in K.nodes}, node_labels={n: node_labels[n] for n in K.nodes}, edgecolor=blue_sky, font_size=8, figsize=(10,6))
# #     plt.savefig(f'./results/nets/kms_graphs/func_circuits/{name}_KMS-subgraph-{b_factor}xbeta_c.pdf', dpi=300, bbox_inches='tight')
# #     plt.close()
# plt.show()




# weights and ratios
# remove_nodes = ['AWCR', 'AWCL']
# G.remove_edges_from([e for e in updt_synapses if (e[0] in remove_nodes) or (e[1] in remove_nodes)])
# name = 'Thermotaxis'
# V = CIRCUITS[name]
# # # # w = node_kms_in_weight(S, V, beta_min, beta, num=20)

# # # # print(w['RIAR'])
# plot_feedback_coef_variation(G, V, bmin, beta, num=1000, colors=COLORS2, font_size=10)
# plt.legend(bbox_to_anchor=(1.05, 1.0), fontsize='8', loc='upper left')

# # plt.savefig(f'./results/fid/{name}_fidelity.pdf', dpi=300, bbox_inches='tight')
# plt.savefig(f'./results/newConnectome/coefficients/{name}_feedback.pdf', dpi=300, bbox_inches='tight')
# plt.close()
# plt.show()


# Node to node flow streams
# remove_nodes = ['AWCR', 'AWCL']
# G.remove_edges_from([e for e in updt_synapses if (e[0] in remove_nodes) or (e[1] in remove_nodes)])
# name = 'Thermotaxis'
# sources = ['AFDR', 'AFDL']
# linestyle = ['-', '-.', ':', '--']

# targets = CIRCUITS[name]
# # targets = ['AFDR', 'AFDL', 'AWCR', 'AWCL', 'AIYR', 'AIZR', 'AIZL', 'RIAR', 'RIAL']  # 

# plot_node_kms_stream_variation(G, sources, targets, bmin, beta, linestyle=linestyle, num=1000, colors=COLORS2, node_labels=node_labels)

# plt.legend(bbox_to_anchor=(1.05, 1.0), fontsize='10', loc='upper left')
# # # plt.tight_layout()
# plt.savefig(f'./results/newConnectome/streams/{sources}_to_{name}_stream.pdf', dpi=300, bbox_inches='tight')
# plt.show()



##### RECEPTANCE

# nodes_removed = ['AFDR', 'AFDL']
# # G.remove_edges_from([e for e in updt_synapses if (e[0] in remove_nodes) or (e[1] in remove_nodes)])
# name = 'Thermotaxis'
# V = CIRCUITS[name]
# # # # w = node_kms_in_weight(S, V, beta_min, beta, num=20)

# # # # print(w['RIAR'])
# plot_kms_receptance(G, V, bmin, beta, num=1000, colors=COLORS2, font_size=10, nodes_removed=nodes_removed)
# plt.legend(bbox_to_anchor=(1.05, 1.0), fontsize='12', loc='upper left')

# # plt.savefig(f'./results/fid/{name}_fidelity.pdf', dpi=300, bbox_inches='tight')
# # plt.savefig(f'./results/newConnectome/coefficients/{name}_receptance.pdf', dpi=300, bbox_inches='tight')
# # # plt.close()
# plt.show()


### Receptance entropies
# name = 'Thermotaxis'
# NOIS = CIRCUITS[name] # Neurons of interests
# plot_kms_receptance_profile_entropy(G, NOIS, bmin, beta, num=1000, colors=COLORS2, node_labels=node_labels)
# plt.legend(bbox_to_anchor=(1.05, 1.0), fontsize='8', loc='upper left')
# # plt.tight_layout()
# plt.savefig(f'./results/newConnectome/entropy/{name}_receptance_entropy.pdf', dpi=300, bbox_inches='tight')
# # plt.savefig(f'./results/circuits/entropy/{name}_emittance_entropy.pdf', dpi=300, bbox_inches='tight')
# # plt.savefig(f'./results/circuits/indices/{name}_feedback_coef.pdf', dpi=300, bbox_inches='tight')
# # plt.show()

### Emittance entropies
# name = 'Thermotaxis'
# NOIS = CIRCUITS[name] # Neurons of interests
# plot_node_kms_emittance_profile_entropy(G, NOIS, bmin, beta, num=1000, colors=COLORS2, node_labels=node_labels)
# plt.legend(bbox_to_anchor=(1.05, 1.0), fontsize='8', loc='upper left')
# # plt.tight_layout()
# plt.savefig(f'./results/newConnectome/entropy/{name}_emittance_entropy.pdf', dpi=300, bbox_inches='tight')
# # plt.savefig(f'./results/circuits/entropy/{name}_emittance_entropy.pdf', dpi=300, bbox_inches='tight')
# # plt.savefig(f'./results/circuits/indices/{name}_feedback_coef.pdf', dpi=300, bbox_inches='tight')
# # plt.show()



#### COMPARISONS FOR RECEPTANCE


##### Misc
# graph_sources = [n for n in neurons if G.in_degree(n) == 0]

# print(graph_sources)

#####################################################################
#################### RANDOM SAMPLES #################################
#####################################################################
# rG = configuration_model_from_directed_multigraph(G)
# rbc = critical_inverse_temperature(rG)
# rbmin = rbc + .000001
# rbeta  = 3.5 * rbc


# #### Draw structural circuits
# name = 'ExtendedThermotaxis'
# # for name in CIRCUITS:
# V = CIRCUITS[name]
# K = weighted_structural_subgraph(rG, nodelist=V)
# draw_weighted_digraph(K, pos={n: neuron_positions[n] for n in K.nodes}, node_size=1.6, node_colors={n: node_colors[n] for n in K.nodes}, node_labels={n: node_labels[n] for n in K.nodes}, edgecolor='silver', font_size=7, figsize=(10,6))
# plt.show()
# plt.savefig(f'./results/newConnectome/nets/struc/{name}_circuit.pdf', dpi=300, bbox_inches='tight')
# plt.close()

# Node to node flow streams
# remove_nodes = ['AWCR', 'AWCL']
# G.remove_edges_from([e for e in updt_synapses if (e[0] in remove_nodes) or (e[1] in remove_nodes)])
# name = 'Thermotaxis'
# sources = ['AFDR', 'AFDL']
# linestyle = ['-', '-.', ':', '--']

# targets = CIRCUITS[name]
# # # targets = ['AFDR', 'AFDL', 'AWCR', 'AWCL', 'AIYR', 'AIZR', 'AIZL', 'RIAR', 'RIAL']  # 

# plot_node_kms_stream_variation(rG, sources, targets, rbc + .000001, 3.5 * rbc, linestyle=linestyle, num=1000, colors=COLORS2, node_labels=node_labels)

# plt.legend(bbox_to_anchor=(1.05, 1.0), fontsize='10', loc='upper left')
# # # # plt.tight_layout()
# # # plt.savefig(f'./results/newConnectome/streams/{sources}_to_{name}_stream.pdf', dpi=300, bbox_inches='tight')
# plt.show()

# weights and ratios
# remove_nodes = ['AWCR', 'AWCL']
# G.remove_edges_from([e for e in updt_synapses if (e[0] in remove_nodes) or (e[1] in remove_nodes)])
# name = 'Thermotaxis'
# V = CIRCUITS[name]
# # # # # w = node_kms_in_weight(S, V, beta_min, beta, num=20)

# # # # # print(w['RIAR'])
# plot_feedback_coef_variation(rG, V, bmin, beta, num=1000, colors=COLORS2, font_size=10)
# plt.legend(bbox_to_anchor=(1.05, 1.0), fontsize='8', loc='upper left')

# # # plt.savefig(f'./results/fid/{name}_fidelity.pdf', dpi=300, bbox_inches='tight')
# # plt.savefig(f'./results/newConnectome/coefficients/{name}_feedback.pdf', dpi=300, bbox_inches='tight')
# # plt.close()
# plt.show()

##### RECEPTANCE

# nodes_removed = ['AFDR', 'AFDL']
# # G.remove_edges_from([e for e in updt_synapses if (e[0] in remove_nodes) or (e[1] in remove_nodes)])
# name = 'Thermotaxis'
# V = CIRCUITS[name]
# # # # # w = node_kms_in_weight(S, V, beta_min, beta, num=20)

# # # # # print(w['RIAR'])
# plot_kms_receptance(rG, V, rbmin, rbeta, num=1000, colors=COLORS2, font_size=10, nodes_removed=None)
# # plt.legend(bbox_to_anchor=(1.05, 1.0), fontsize='12', loc='upper left')

# # # plt.savefig(f'./results/fid/{name}_fidelity.pdf', dpi=300, bbox_inches='tight')
# # # plt.savefig(f'./results/newConnectome/coefficients/{name}_receptance.pdf', dpi=300, bbox_inches='tight')
# # # # plt.close()
# plt.show()

### Receptance entropy
# name = 'Thermotaxis'
# NOIS = CIRCUITS[name] # Neurons of interests
# plot_kms_receptance_profile_entropy(rG, NOIS, rbmin, rbeta, num=1000, colors=COLORS2, node_labels=node_labels)
# plt.legend(bbox_to_anchor=(1.05, 1.0), fontsize='8', loc='upper left')
# # plt.tight_layout()
# # plt.savefig(f'./results/newConnectome/entropy/{name}_receptance_entropy.pdf', dpi=300, bbox_inches='tight')
# # plt.savefig(f'./results/circuits/entropy/{name}_emittance_entropy.pdf', dpi=300, bbox_inches='tight')
# # plt.savefig(f'./results/circuits/indices/{name}_feedback_coef.pdf', dpi=300, bbox_inches='tight')
# plt.show()

### Emittance entropies
# name = 'Thermotaxis'
# NOIS = CIRCUITS[name] # Neurons of interests
# plot_node_kms_emittance_profile_entropy(rG, NOIS, rbmin, rbeta, num=1000, colors=COLORS2, node_labels=node_labels)
# plt.legend(bbox_to_anchor=(1.05, 1.0), fontsize='8', loc='upper left')
# # plt.tight_layout()
# # plt.savefig(f'./results/newConnectome/entropy/{name}_emittance_entropy.pdf', dpi=300, bbox_inches='tight')
# # # plt.savefig(f'./results/circuits/entropy/{name}_emittance_entropy.pdf', dpi=300, bbox_inches='tight')
# # # plt.savefig(f'./results/circuits/indices/{name}_feedback_coef.pdf', dpi=300, bbox_inches='tight')
# plt.show()

### Statistical Significance #####
# nodes, fZ = kms_emittance(G, bf)

# NOI = 'AFDR' # neuron if interest
# target = 'AIYR'
# i = nodes.index(NOI)
# j = nodes.index(target)
# fZv = fZ[:, i] ## emittance profile of NOI
# com = fZv[j] # 0.08611704354455302
# NOI = 'AFDR' # neuron if interest
# target = 'AWCL'
# i = nodes.index(NOI)
# j = nodes.index(target)
# fZv = fZ[:, i] ## emittance profile of NOI
# com = fZv[j] # 0.08611704354455302
# # com = 0.08611704354455302
# rand_sample = random_sample_from_deg_seq(G, n_iter=100)

# NOI_to_target = []

# for s in rand_sample:
#     g = rand_sample[s]
#     snodes, sfZ = kms_emittance(g, bf)
#     si = snodes.index(NOI)
#     sj = snodes.index(target)
#     sfZv = sfZ[:, si]
#     s_com = sfZv[sj]
#     NOI_to_target += [s_com,]


# t_stat, p_value = stats.ttest_1samp(NOI_to_target, com)
# print(NOI_to_target)
# p_value = float(len([x for x in NOI_to_target if x >= com])) / 100.0

# print(p_value)

############### GENERATE KMS NETWORK WITH P-VALUES
# name = 'ExtendedThermotaxis'
# name = 'AFD'
# sources = CIRCUITS[name]
# # # # # targets = CIRCUITS[name]
# # targets = list(G.nodes)
# # # # # factors = [.01, .06, .07, .09, 1.1, 1.5, 1.9, 2.]
# b_factor = 1.06
# # # # # b_factor = 1.035
# NITER = 5000
# # # # # # con_sig = kms_connectivity_significance(G, V, targets, b_factor * bc, n_iter=NITER)
# fname = f'./results/data/kms_connectivity/{name}_kms_connectivity_{b_factor}xbeta_c_{NITER}-iters.csv'
# generate_kms_connectivity(G, b_factor, sources, targets, s_name=name, t_name='', n_iter=NITER, fname=fname)

# kms_conn_f = open(f'./results/data/kms_connectivity/{name}_kms_connectivity_{b_factor}xbeta_c_{NITER}-iters.csv', 'w', newline='')
# writer = csv.writer(kms_conn_f, delimiter=',')
# writer.writerow([f'# KMS connectivity of {name} neurons at inverse temperature of {b_factor}x beta_c; the p-values were calculated over {NITER} iterations of random graphs'])
# writer.writerow(['neuron1','neuron2','kms_weight','p-value'])

# for k in con_sig:
#     con_list = con_sig[k]
#     for link in con_list:
#         row = [str(k),str(link[0]),float(link[1]),float(link[2])]
#         writer.writerow(row)


####### REMOVING NEURONS 
# nodes_removed = ['AWCL', 'AWCR']
# # # G.remove_nodes_from(nodes_removed)
# # targets = list(G.nodes)
# # # bc = critical_inverse_temperature(G)
# file_dir = './results/data/kms_connectivity/surgery/'

# generate_kms_connectivity(G, b_factor, sources, targets, s_name='all', t_name=name, removed_nodes=nodes_removed, n_iter=NITER, file_dir=file_dir)
# con_sig = kms_connectivity_significance(G, V, targets, b_factor * bc, n_iter=NITER)

# kms_conn_f = open(f'./results/data/kms_connectivity/surgery/{name}_kms_connect_({nodes_removed}-removed)_{b_factor}xbeta_c_{NITER}-iters.csv', 'w', newline='')
# writer = csv.writer(kms_conn_f, delimiter=',')
# writer.writerow([f'# KMS connectivity of {name} neurons at inverse temperature of {b_factor}x beta_c; the p-values were calculated over {NITER} randomly generated graphs with same degree sequence'])
# writer.writerow(['neuron1','neuron2','kms_weight','p-value'])

# for k in con_sig:
#     con_list = con_sig[k]
#     for link in con_list:
#         row = [str(k),str(link[0]),float(link[1]),float(link[2])]
#         writer.writerow(row)





##### REMOVING NON-SIGNIFICANT EDGES
# NONSIGNIFICANT = {
#     'RIAL-RIAR': [e for e in updt_synapses if (e[0] == 'RIAL') and (e[1] == 'RIAR')],
#     'AIAL-AIAR': [e for e in updt_synapses if (e[0] == 'AIAL') and (e[1] == 'AIAR')],
#     'AIBR-AIZL': [e for e in updt_synapses if (e[0] == 'AIBR') and (e[1] == 'AIZL')],
#     'AWCR-RIAL': [e for e in updt_synapses if (e[0] == 'AWCR') and (e[1] == 'RIAL')],
#     'AIBR-RIAL': [e for e in updt_synapses if (e[0] == 'AIBR') and (e[1] == 'RIAL')],
#     'AIZR-AIAR': [e for e in updt_synapses if (e[0] == 'AIZR') and (e[1] == 'AIAR')],
#     'AIAL-AIZL': [e for e in updt_synapses if (e[0] == 'AIAL') and (e[1] == 'AIZL')],
# }
# edges_name = 'RIAL-RIAR'
# G.remove_edges_from(NONSIGNIFICANT[edges_name])

# bc = critical_inverse_temperature(G)



# # # print(bc)
# name = 'ExtendedThermotaxis'
# V = CIRCUITS[name]
# # b_factor = 1.035
# b_factor = 1.9


# K = weighted_structural_subgraph(G, nodelist=V)
# draw_weighted_digraph(K, pos={n: POSITIONS['thermotaxis'][n] for n in K.nodes}, node_size=1.8, node_colors={n: node_colors[n] for n in K.nodes}, node_labels={n: node_labels[n] for n in K.nodes}, edgecolor=default_edge_color, font_size=10, figsize=(10,6))
# plt.show()

# con_sig = kms_connectivity_significance(G, V, b_factor * bc, n_iter=NITER)

# kms_conn_f = open(f'./results/data/kms_connectivity/surgery/{name}_{edges_name}_kms_connectivity_{b_factor}xbeta_c_{NITER}-iters.csv', 'w', newline='')
# writer = csv.writer(kms_conn_f, delimiter=',')
# writer.writerow([f'# KMS connectivity of the Thermotaxis circuit at inverse temperature of {b_factor}x beta_c'])
# writer.writerow(['source,target,kms_weight,p-value'])

# for k in con_sig:
#     con_list = con_sig[k]
#     for link in con_list:
#         row = [str(k),str(link[0]),float(link[1]),float(link[2])]
#         writer.writerow(row)


# print(1 / (2.2 * bc))

############# Reading files 
#### Read and Visualize the resulting network file
# alpha = .05
# kms_conn_f = open(fname, 'r', newline='')

# print(regular_polygon_coord(10, 4.))

# kms_con_rows = [line.strip().split(',') for line in kms_conn_f.readlines()[2:]]

# kms_weighted_connections = []

# for line in kms_con_rows:
#     if float(line[3]) <= alpha:
#         kms_weighted_connections += [(str(line[0]), str(line[1]), float(line[2])),] 

# # # # w_sum = nonzero_sum([e[2] for e in kms_weighted_connections])   

# # # # kms_weighted_connections = [(e[0], e[1], e[2] / w_sum) for e in kms_weighted_connections]

# K = nx.DiGraph()
# K.add_weighted_edges_from([e for e in kms_weighted_connections if e[0] == 'AFDR'])


# node_shape = {}
# for n in [v for v in list(K.nodes) if (str(v) != 'AFDR')]:
#     if  ('AFDR', n) in G.edges:
#         node_shape[n] = 'round'
#     else:
#         node_shape[n] = 'polygon'
#     node_shape.copy()


# draw_weighted_digraph(K, pos={n: neuron_positions[n] for n in K.nodes}, node_shape=node_shape, node_size=2., node_colors={n: node_colors[n] for n in K.nodes}, node_labels={n: node_labels[n] for n in K.nodes}, edgecolor=blue_sky, font_size=12, figsize=(10,6))
# plt.show()
# plt.savefig(f'./results/newConnectome/nets/func/{name}_kms_circuit_{b_factor}xbeta_c_{NITER}-iters.pdf', dpi=300, bbox_inches='tight')

###################################################
####### Read and visualize Node KMS emittance files
# left = sources[0]
# right = sources[1]
# # HEM = {
# #     'left': left,
# #     'right': right
# # }
# thresh = .05
# kms_conn_f = open(f'./results/data/kms_connectivity/{name}_kms_connectivity_{b_factor}xbeta_c_{NITER}-iters.csv', 'r', newline='')

# kms_con_rows = [line.strip().split(',') for line in kms_conn_f.readlines()[2:]]
# left_conns = []
# right_conns = []
# # kms_weighted_connections = []

# for line in kms_con_rows:
#     if float(line[3]) <= thresh:
#         if str(line[0]) == left:
#             left_conns += [(str(line[1]), float(line[2])),]
#         elif str(line[0]) == right:
#             right_conns += [(str(line[1]), float(line[2])),]

# #### plotting the individual node emittance
# ax = plt.gca()

# for i, connect in enumerate(right_conns):
#     marker = 'o'
#     if is_out_neighbor(G, right, connect[0]) == False:
#         marker = 'D'
#     plt.scatter(i, connect[1], color=node_colors[connect[0]], s=100, edgecolors='dimgray', marker=marker, alpha=.5)
#     ax.annotate(node_labels[connect[0]], (i + .005, connect[1] + .005), fontsize=7)

# ax.get_xaxis().set_visible(False)

# plt.xlabel(f'Nodes')
# plt.ylabel(f'{right} KMS connectivity')
# plt.show()
# plt.savefig(f'./results/newConnectome/nets/func/{right}_kms_emittance_{b_factor}xbeta_c_{NITER}-iters.pdf', dpi=300, bbox_inches='tight')
# print(sorted(left_conns, key=lambda x: x[1], reverse=True))

# print([n for n in G.nodes if ('ASER', n) in G.edges])
# print(G.in_degree('ASEL'), G.in_degree('ASER'))

#         kms_weighted_connections += [(str(line[0]), str(line[1]), float(line[2])),] 

##### Read surgery kms
# thresh = 1.
# kms_conn_f = open(f'./results/data/kms_connectivity/surgery/{name}_{edges_name}_kms_connectivity_{b_factor}xbeta_c.csv', 'r', newline='')

# kms_con_rows = [line.strip().split(',') for line in kms_conn_f.readlines()[2:]]

# kms_weighted_connections = []

# for line in kms_con_rows:
#     if float(line[3]) <= thresh:
#         kms_weighted_connections += [(str(line[0]), str(line[1]), float(line[2])),] 

# # w_sum = nonzero_sum([e[2] for e in kms_weighted_connections])   

# # kms_weighted_connections = [(e[0], e[1], e[2] / w_sum) for e in kms_weighted_connections]

# K = nx.DiGraph()
# K.add_weighted_edges_from(kms_weighted_connections)

# draw_weighted_digraph(K, pos={n: POSITIONS['thermotaxis'][n] for n in K.nodes}, node_size=1.6, node_colors={n: node_colors[n] for n in K.nodes}, node_labels={n: node_labels[n] for n in K.nodes}, edgecolor=blue_sky, font_size=7, figsize=(10,6))
# plt.show()


###############################################
######## KMS STATES DIVERGENCE ################
# plot_kms_states_js_divergence(G, [('AIAL', 'AFDR'), ('AIAL', 'AFDL'), ('AIAR', 'AFDR'), ('AIAR', 'AFDL'), ('AIYL', 'AFDL'), ('AIYL', 'AFDR'), ('AIYR', 'AFDR')], bmin, 2.5 * bc, num=50, node_labels=node_labels)
# plt.legend(bbox_to_anchor=(1.05, 1.0), fontsize='8', loc='upper left')
# # plt.savefig(f'./results/newConnectome/divergence/left_right_jensen-shannon_divergence.pdf', dpi=300, bbox_inches='tight')
# plt.show()





###########################################################
############## KMS STATES ASSYMMETRY BY  NEURON CLASSES ######################

# neur_pair = ('ASEL', 'ASER')

# ngb_cls = {}

# for neuron in neur_pair:
#     out_neigh = out_neighbors(G, str(neuron))
#     out_cls = []
#     for n in out_neigh:
#         c = [cl for cl in neuron_by_classes if str(n) in neuron_by_classes[cl]][0]
#         out_cls += [c,]
#     out_cls = list(set(out_cls))

#     ngb_cls[neuron] = out_cls
#     ngb_cls.copy()

# left_cls = ngb_cls[neur_pair[0]]
# right_cls = ngb_cls[neur_pair[1]]

# # common_cls = list(set(left_cls) & set(right_cls))
# common_cls = [cl for cl in left_cls if cl in right_cls]
# all_cls = left_cls + right_cls
# a = len(common_cls)
# d = len(all_cls) * 1.
# b = len([cl for cl in left_cls if cl not in common_cls])
# c = len([cl for cl in right_cls if cl not in common_cls])

# p0 = a / d

# pe = ((a + b) / d) * ((a + c) / d)

# kappa = (p0 - pe) / (1. - pe)

# print(kappa)
# G = configuration_model_from_directed_multigraph(G)
# print(connectivity_class_symmetry_coefficient(list(G.edges), [tuple(sources)], neuron_by_classes), connectivity_class_symmetry_coefficient(list(K.edges), [tuple(sources)], neuron_by_classes))
# neuron_by_classes = {cl: neuron_by_classes[cl] for cl in neuron_by_classes if neuron_by_classes[cl] in list(G.nodes)}
# node_pairs = [('ASEL', 'ASER'), ('AFDL', 'AFDR'), ('AWCL', 'AWCR'), ('ASIL', 'ASIR'), ('AIYL', 'AIYR'), ('AIZL', 'AIZR'), ('AIBL', 'AIBR'), ('AIAL', 'AIAR'), ('RIAL', 'RIAR'), ('AVEL', 'AVER'), ('AVAL', 'AVAR'), ('IL2L', 'IL2R'), ('FLPL', 'FLPR')]
# plot_kms_states_by_classes_divergence(G, neuron_by_classes, node_pairs, bmin, 2.5 * bc, num=100, colors=COLORS, node_labels=node_labels)
# # plot_kms_states_js_divergence(G, node_pairs, bmin, 2.5 * bc, num=100, colors=COLORS, node_labels=node_labels)
# plt.legend(bbox_to_anchor=(1.05, 1.0), fontsize='8', loc='upper left')
# # plt.savefig(f'./results/newConnectome/divergence/left_right_kms_fidelity_individual.pdf', dpi=300, bbox_inches='tight')
# plt.show()
# print([c for c in neuron_by_classes if 'AFDR' in neuron_by_classes[c]][0])


###### Plot neural emittance

# plot_node_emittance(G, CIRCUITS['ExtendedThermotaxis'], bmin, beta, node_labels=node_labels)

# plt.legend(bbox_to_anchor=(1.05, 1.0), fontsize='8', loc='upper left')
# plt.show()



# autapses = [e for e in updt_synapses if e[0] == e[1]]
# RMDVR = [e[0] for e in updt_synapses if e[1] == 'RMDVR']
# # print(len(set(autapses)))
# # print(G.out_degree('RID'), G.in_degree('RID'))
# # print(len(all_paths(G,'AFDR', 'AIYR')))
# print(set(RMDVR))    
# print(set(e[0] for e in autapses))

# AFD_out = [e for e in updt_synapses if e[0] == 'AFDR']
# RMD_in = [e for e in updt_synapses if e[1] == 'RMDVR']

# AFD_out_ends = [e[1] for e in AFD_out]
# RMD_in_starts = [e[0] for e in RMD_in]

# intermed = [f for f in updt_synapses if (f[0] in AFD_out_ends) and (f[1] in RMD_in_starts)]

# one_hop = [(e, f) for e, f in product(AFD_out, RMD_in) if e[1] == f[0]]

# two_hop = [(e1, e2, e3) for e1, e2, e3 in product(AFD_out, intermed, RMD_in)]

# print(len(one_hop))

# paths = npaths(list(G.edges), 'RID', 'URXL', 3)
# paths = npaths(list(G.edges), 'AFDR', 'AIBR', 4)

# # # path6 = paths_6(list(G.edges), 'AFDR', 'RMDVR')
# # # print(set(e[0] for e in autapses))
# print(len(paths))














