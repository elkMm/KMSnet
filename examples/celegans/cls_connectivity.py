import os
import sys
import re
import networkx as nx
import matplotlib.pyplot as plt 
import numpy as np 
import pandas as pd
import csv


ex_dir = os.path.dirname(__file__)
cn_module_dir = os.path.join(ex_dir, '../..')
sys.path.append( cn_module_dir )

from connectome import neurons, neuron_cls, neuron_cl_dict, synapses, color_maps, node_colors

from src.visuals import *
from src.utils import *
from src.states import *
from src.generators import *
from src.plotting import *
from src.kms_graphs import *

import seaborn as sns

np.random.seed(5679)





cls_connections = []
for n1, n2 in synapses:
    cl1 = next(iter([cl for cl in neuron_cls if n1 in neuron_cl_dict[cl]]))
    cl2 = next(iter([cl for cl in neuron_cls if n2 in neuron_cl_dict[cl]]))
    cls_connections += [(cl1, cl2), ]

cls_labels = {cl: cl for cl in neuron_cls}

cls_colors = {}
for cl in neuron_cls:
    n = next(iter(neuron_cl_dict[cl]))
    cls_colors[cl] = node_colors[n]
    cls_colors.copy()

C = nx.MultiDiGraph()
# C.add_nodes_from(neuron_cls)
C.add_edges_from(cls_connections)

nodes = C.nodes
nodelist = ['AVA', 'AVE', 'AVF', 'AIY', 'AWB', 'RIA', 'AS', 'ADA', 'ADE', 'ALM', 'PVC', 'IL1', 'IL2', 'DB', 'VB', 'DD', 'VA', 'DA', 'DVA', 'DVB', 'DVC']

nodelist = [n for n in nodelist if n in C.nodes]

locomorory_circuit = ['AVD', 'AVA', 'PVC', 'AVB', 'VA', 'DA', 'AS', 'VB', 'DB', 'DD', 'VD', 'PDB']

# draw_multi_digraph(C, layout=nx.kamada_kawai_layout, node_shape='polygon',\
#                        node_colors={n: cls_colors[n] for n in nodes}, node_labels={n: cls_labels[n] for n in nodes})
# plt.show()

beta_c = critical_inverse_temperature(C)
beta_min = .96*beta_c
beta = 10.*beta_c + .0001
# print(beta_c)
plot_node_kms_emittance_profile_entropy(C, locomorory_circuit, beta_min, beta, num=100)
# plt.show()
# profile = subnet_gibbs_profile(C, nodelist, beta)
# data = profile[1]
# data = np.random.rand(6, 6)

# sns.clustermap(data, metric='jensenshannon', figsize = (10, 6),  # Figure sizes
#               dendrogram_ratio = 0.1)

# e_thresh, w_thresh, KMSE = beta_kms_digraph(C, 1.1*beta_c, entropy_ratio=.9, w_ratio=.1)

# print(f'thresh: {thresh}')
# # print([(v, u, KMSemit.get_edge_data(v, u)['weight']) for v, u in KMSemit.edges])
# A = nx.to_numpy_array(KMSemit, nodelist=KMSemit.nodes, dtype=np.float32)
# deg = dict(KMSE.out_degree)
# KMSAbs = nx.DiGraph()
# KMSAbs.add_edges_from([(u, v) for (v, u) in KMSemit.edges])
# pos = nx.random_layout(KMSE)
# nx.draw_networkx(KMSE, pos=pos, arrows=True, nodelist=list(deg.keys()), node_size=[(v+.3) * 40 for v in deg.values()], width=.4, with_labels=True, labels={n: n for n in list(deg.keys())}, node_color = [cls_colors[n] for _, n in enumerate(list(deg.keys()))], alpha=.6,horizontalalignment='center', verticalalignment='center', edge_color="tab:gray", font_size=5)
# # K = nx.MultiDiGraph()
# # K.add_edges_from(KMSemit.edges)
# # draw_multi_digraph(K, layout=nx.kamada_kawai_layout, node_colors={n: node_colors[n] for n in K.nodes})

plt.legend(bbox_to_anchor=(1.05, 1.0), fontsize='8', loc='upper left')
plt.tight_layout()
# plt.savefig('./results/neuronClass/entropy.pdf', dpi=300, bbox_inches='tight')

plt.show()