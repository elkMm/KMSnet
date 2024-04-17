import os
import sys
import re
import networkx as nx
import matplotlib.pyplot as plt 
import numpy as np 
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


from connectome import neuron_positions, node_colors, node_labels, CIRCUITS, blue_sky


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


### CRITICAL TEMPERATURE

bc = critical_inverse_temperature(G) # 4.295757002698898

# betabar_c = critical_inverse_temperature(S_bar)

## Ground states are around: 10 * beta_c + 1.7763
# beta_gd = 10 * beta_c + (52.265/100) * beta_c
# beta_gd = 10.5 * beta_c

bmin = bc + .000001

beta = 3.5*bc 

bf = 1.06 * bc
# print(bc)


# #### Draw structural circuits
# name = 'Thermotaxis'
# # for name in CIRCUITS:
# V = CIRCUITS[name]
# K = weighted_structural_subgraph(G, nodelist=V)
# draw_weighted_digraph(K, pos={n: neuron_positions[n] for n in K.nodes}, node_size=1.6, node_colors={n: node_colors[n] for n in K.nodes}, node_labels={n: node_labels[n] for n in K.nodes}, edgecolor='silver', font_size=7, figsize=(10,6))
# # plt.show()
# plt.savefig(f'./results/newConnectome/nets/struc/{name}_circuit.pdf', dpi=300, bbox_inches='tight')
# plt.close()



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
# sources = ['AIYR', 'AIYL']
# linestyle = ['-', '-.', ':', '--']

# targets = CIRCUITS[name]
# # targets = ['AFDR', 'AFDL', 'AWCR', 'AWCL', 'AIYR', 'AIZR', 'AIZL', 'RIAR', 'RIAL']  # 

# plot_node_kms_stream_variation(G, sources, targets, bmin, beta, linestyle=linestyle, num=1000, colors=COLORS2, node_labels=node_labels)

# plt.legend(bbox_to_anchor=(1.05, 1.0), fontsize='10', loc='upper left')
# # # plt.tight_layout()
# # plt.savefig(f'./results/newConnectome/streams/{sources}_to_{name}_stream.pdf', dpi=300, bbox_inches='tight')
# plt.show()



##### RECEPTANCE

nodes_removed = ['AFDR', 'AFDL']
# G.remove_edges_from([e for e in updt_synapses if (e[0] in remove_nodes) or (e[1] in remove_nodes)])
name = 'Thermotaxis'
V = CIRCUITS[name]
# # # w = node_kms_in_weight(S, V, beta_min, beta, num=20)

# # # print(w['RIAR'])
plot_kms_receptance(G, V, bmin, beta, num=1000, colors=COLORS2, font_size=10, nodes_removed=nodes_removed)
plt.legend(bbox_to_anchor=(1.05, 1.0), fontsize='12', loc='upper left')

# plt.savefig(f'./results/fid/{name}_fidelity.pdf', dpi=300, bbox_inches='tight')
# plt.savefig(f'./results/newConnectome/coefficients/{name}_receptance.pdf', dpi=300, bbox_inches='tight')
# # plt.close()
plt.show()



#### COMPARISONS FOR RECEPTANCE








