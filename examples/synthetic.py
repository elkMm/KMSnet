import os
import sys
import networkx as nx
import matplotlib.pyplot as plt 
import numpy as np 
import pandas as pd
import csv

import seaborn as sns


ex_dir = os.path.dirname(__file__)
cn_module_dir = os.path.join(ex_dir, '..')
sys.path.append( cn_module_dir )

from src.visuals import *
from src.utils import *
from src.states import *
from src.generators import *
from src.plotting import *
from src.kms_graphs import *

np.random.seed(67989)


# V = range(1,6)
# E = [(1,2),]*100000 + [(2,1),]*100000
# E += [(2,1), (2,3), (3, 4), (4,4), (5,3)]
# E = [(1, 2), (1, 2), (1, 3), (1, 5), (2, 2), (2, 3), (2, 4), (2, 5), (3, 4), (3, 6), (3, 6), (3, 6), (4, 4), (4, 6), (5, 2), (5, 3), (5, 4), (5, 6)]
# # V = ['u', 'v', 'w', 'z', ]
# # E = [('u', 'u'),('u', 'v'), ('u', 'v'), ('w', 'v'), ('w', 'u'), ('z', 'w')]
# G = nx.complete_graph(20, create_using=nx.DiGraph())
# G = nx.random_k_out_graph(300, 3, 4, self_loops=True)
# E1 = list(G1.edges)
# E1b = [(e[1], e[0]) for e in E1]

# V = range(1, 5)
# E = [(1, 1), (1, 2), (1, 2), (1, 2), (1, 3)]

## stochastic block model
sizes = [15, 10, 30]
probs = [[0.5, 0.5, 0.02], [0.5, .2, 0.07], [0.2, 0.07, .02]]
g = nx.stochastic_block_model(sizes, probs, directed=True, seed=0)

V = list(g.nodes)
E = list(g.edges)

G = nx.MultiDiGraph()
G.add_nodes_from(V)
G.add_edges_from(E)
G.add_edges_from([(2,3),]*100000)
# G_bar = conjugate_graph(G)
# G = conditional_random_multi_digraph(20, .35, .12)
# G = nx.fast_gnp_random_graph(20, .2, directed=True, seed=56789)
# G = scale_free_digraph(130)
# draw_multi_digraph(G)
# plt.show()
# A = directed_multigraph_adjacency_matrix(G)
# print(critical_inverse_temperature(G))
# print(KMS_states(G, 3.))


# KMS = KMS_states(G, 2*critical_inverse_temperature(G))

# # sns.histplot(data=KMS)
# # plt.show()

# x = KMS[0] #[i for i, _ in enumerate(KMS[0])]
# y = KMS[1]
# plt.scatter(x, y)
# plt.show()
beta_c = critical_inverse_temperature(G) # 3.39842857192179
# print(beta_c)
# ## Ground states are around: 10 * beta_c + 1.7763
# beta_gd = 10 * beta_c + (52.265/100) * beta_c
# beta_gd = 10.5 * beta_c

beta_min = beta_c + .000001

beta = 10*beta_c + .1
print(beta_c)
# ## Reference beta: 1.8

# epsilon = (1.7/100) * beta 




# KMS1 = KMS_states(G, beta)
# KMS2 = KMS_states(G, 1.5*beta)


# sns.histplot(data=KMS1[1])

# plt.show()
# x = KMS1[0]
# y = KMS1[1]

# plt.scatter(x, y)
# plt.ylabel('Emittance')
# ax = plt.gca()
# for i, n in enumerate(x):
#     ax.annotate(node_labels[n], (x[i], y[i]), fontsize=7)



# ax.get_xaxis().set_visible(False)
# plt.show()
# print(beta_c)

# x = KMS1[0] #[i for i, _ in enumerate(KMS[0])]
# y = KMS1[1]

# sample1 = KMS1[1] / sum(KMS1[1])
# sample2 = KMS2[1] / sum(KMS2[1])
# print(stats.cramervonmises_2samp(sample1, sample2))
# print(stats.cramervonmises_2samp(KMS1[1], KMS2[1]))
# print(beta)
# entropy_thresh, w_thresh, KMSemit = beta_kms_digraph(G, 20*beta)
plot_sates_fidelity(G, list(G.nodes), beta_min, beta, num=1000)
# print(f'thresh: {thresh}')
# print([(v, u, KMSemit.get_edge_data(v, u)['weight']) for v, u in KMSemit.edges])
# print(entropy_thresh, w_thresh)
# deg = dict(KMSemit.out_degree)
# pos = nx.spring_layout(KMSemit)
# nx.draw_networkx(KMSemit, pos=pos, arrows=True, nodelist=list(deg.keys()), node_size=[(v+.2) * 100 for v in deg.values()], labels={n: n for n in list(deg.keys())}, alpha=.6,horizontalalignment='left', verticalalignment='bottom', edge_color="tab:gray")
# H = node_structural_entropy(G, nodelist=G.nodes)
# xs = [n for n in H]
# ys = [H[n] for _, n in enumerate(xs)]
# plt.scatter(xs, ys)
# plot_node_kms_emittance_profile_entropy(G, G.nodes, beta_min, 10*beta, num=10000)
# plt.legend(bbox_to_anchor=(1.05, 1.0), fontsize='10', loc='upper left')
# plot_kms_simplex_volume(G, beta_min, beta, num=50)
plt.show()
