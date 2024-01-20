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
nodelist = ['AVA', 'AVE', 'AS', 'ADA', 'ADE', 'ALM', 'PVC', 'IL1', 'IL2', 'DB', 'VB', 'DD', 'VA', 'DA']

# draw_multi_digraph(C, layout=nx.kamada_kawai_layout, node_shape='polygon',\
#                        node_colors={n: cls_colors[n] for n in nodes}, node_labels={n: cls_labels[n] for n in nodes})
# plt.show()

beta_c = critical_inverse_temperature(C)
# beta_min = beta_c + .000001
beta = beta_c + .0001
# print(beta_c)
# plot_node_profile_entropy(C, nodelist, beta_min, beta, num=1000)
# plt.show()
profile = subnet_gibbs_profile(C, nodelist, beta)
data = profile[1]
# data = np.random.rand(6, 6)

sns.clustermap(data, metric='jensenshannon', figsize = (10, 6),  # Figure sizes
              dendrogram_ratio = 0.1)
plt.show()