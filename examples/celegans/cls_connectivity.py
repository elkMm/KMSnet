import os
import sys
import re
import networkx as nx
import matplotlib.pyplot as plt 
import numpy as np 
import pandas as pd
import csv
from scipy.spatial.distance import jensenshannon
from scipy.special import rel_entr

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

np.random.seed(675679)





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

nodes = list(C.nodes)
nodelist = ['AVA', 'AVE', 'AVF', 'AIY', 'AWB', 'RIA', 'AS', 'ADA', 'ADE', 'ALM', 'PVC', 'IL1', 'IL2', 'DB', 'VB', 'DD', 'VA', 'DA', 'DVA', 'DVB', 'DVC']

nodelist = [n for n in nodelist if n in C.nodes]

locomorory_circuit = ['AVD', 'AVA', 'PVC', 'AVB', 'VA', 'DA', 'AS', 'VB', 'DB', 'DD', 'VD', 'PDB']
olfactory_circuit = ['AFD', 'AWA', 'AWB', 'AWC', 'ADL', 'ASH', 'AVH']
thermosensory_circuit = ['AIY', 'AIZ', 'AIA', 'AIB', 'RIF', 'AVD', 'AVA', 'AVB']
olfac_thermo_circuit = olfactory_circuit + thermosensory_circuit + ['RIA']
thermo_subcircuit = ['AFD', 'AWC', 'AIY', 'AIZ', 'RIA']

# draw_multi_digraph(C, layout=nx.kamada_kawai_layout, node_shape='polygon',\
#                        node_colors={n: cls_colors[n] for n in nodes}, node_labels={n: cls_labels[n] for n in nodes})
# plt.show()

# C.remove_nodes_from(locomorory_circuit)

beta_c = critical_inverse_temperature(C) # = 5.286973453649968
beta_min = beta_c - .0001
beta = 20.*beta_c + .0001

beta_afd = 1.498 * beta_c # beta_afd = 7.919886233567652, T = 0.12626443998167539
beta_awc = 1.548 * beta_c # beta_awc = 8.18423490625015, T = 0.12218613119673756
beta_aiy = 1.5999999 * beta_c + .000001 # beta_aiy = 8.459157997142603, T =  0.11821507534648099
beta_ria = 1.59 * beta_c # beta_ria = 8.40628779130345, T = 0.11895857301418221
beta_ava = 1.7279999 * beta_c  + .02# beta_ava = 9.155889599209798, T = 0.1092193160658365
beta_aiz = 1.59 * beta_c + .0004 # beta_aiz = 8.406687791303451, T = 0.1189529128266759
# print(f'beta_c: {beta_c}')
# print(f'beta_afd: {beta_afd}')
# print(f'beta_ava: {beta_ava}')
# print(f'beta: {beta_aiz}')
# print(f'temperature: {1./beta_aiz}')
# plot_node_kms_emittance_profile_entropy(C, olfac_thermo_circuit, beta_min, beta, num=10000)
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

# plt.legend(bbox_to_anchor=(1.05, 1.0), fontsize='8', loc='upper left')
# plt.tight_layout()
# plt.savefig('./results/new/thermoOlfac.pdf', dpi=300, bbox_inches='tight')
# node_prof = node_kms_emittance_profile(C, 'AWC', 2.1*beta_c)
# xs = list(np.linspace(1./(2.8*beta_c), 1./beta_min, num=1000))
# vertices, SS = node_structural_state(C)
# centers = thermo_subcircuit

# center = 'AVA'
# inde = vertices.index(center)
# struc = SS[center]
# struc = remove_ith(struc, inde)
# struc[inde] = 0.
# ss = sum(struc)
# if ss == 0.:
#     ss = 1.
# struc = [x / float(ss) for _, x in enumerate(struc)]

# ys = []
# for _, b in enumerate(xs):
#     vertices2, Z = kms_emittance(C, 1./b)
#     prof = Z[:, inde]
#     # prof[inde] = 0.

#     # # # Remove numbers close to zero 
#     # prof = [get_true_val(x) for _, x in enumerate(prof)]
#     # s = sum(prof)
#     # if s == 0.:
#     #     s = 1.
#     # prof = [x / float(s) for _, x in enumerate(prof)]
#     prof = remove_ith(prof, inde)
#     ys += [fidelity(struc, prof),]


# plt.plot(xs, ys)

# print(f'JSD: {jsd}')
# print(f'nodes: {vertices2}\n {struc}\n')
# print(f'nodes: {vertices}\n {prof}')
# A = adjacency_matrix(C, nodes=nodes)
# A = A.T
# i = nodes.index('ADF')

# print(f'{vertices2}\n{prof}\n')


# plot_sates_fidelity(C, olfac_thermo_circuit, beta_min, 2.8*beta_c, num=1000)
# fid = fidelity(struc, prof)
# print([(n, struc[vertices2.index(n)]) for n in vertices2 if struc[vertices2.index(n)] != 0], '\n')

# print([(n, prof[vertices2.index(n)]) for n in vertices2 if prof[vertices2.index(n)] != 0])
# print(fid)
# prof = [get_true_val(x, tol=1e-03) for _, x in enumerate(prof)]
# transform the range into [0,1]
# a = float(min(prof))
# b = float(max(prof))
# d = 1./(b - a)
# prof = [(x - a) * d for _, x in enumerate(prof)]


# ax = plt.gca()
# # Select only the neurons with non-zero receptance
# new_v = []
# for p, nv in enumerate(vertices):
#     if prof[p] > 0:
#         plt.scatter(nv, prof[p], color=cls_colors[nv])
#         ax.annotate(cls_labels[nv], (nv, prof[p]), fontsize=7)


# for i, n in enumerate(vertices):
#     ax.annotate(cls_labels[n], (vertices[i], prof[i]), fontsize=7)



# ax.get_xaxis().set_visible(False)

# plt.xlabel(f'Nodes')
# plt.ylabel(f'{center} KMS intensity')
# plot_kms_simplex_volume(C, beta_min, beta, num=100)
# plot_node_kms_emittance_profile_entropy(C, thermo_subcircuit, 1.001*beta_c, beta, num=10000)
# plt.legend(bbox_to_anchor=(1.05, 1.0), fontsize='8', loc='upper left')
# # plt.tight_layout()
# plt.savefig(f'./results/new/{center}_KMS_circuit_1.1xbeta_c+.001.pdf', dpi=300, bbox_inches='tight')
# plt.savefig(f'./results/new/OlfacThermo_KMS_entropy.pdf', dpi=300, bbox_inches='tight')

# plt.show()
# print([e for e in cls_connections if 'AFD' == e[0]])
# print(beta_c)


plot_feedback_coef(C, locomorory_circuit, beta_min, beta, num=1000)
plt.legend(bbox_to_anchor=(1.05, 1.0), fontsize='10', loc='upper left')

# plt.savefig('./results/coefficients/ThermoSensory_feedback.pdf', dpi=300, bbox_inches='tight')
plt.show()
