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

np.random.seed(567989)


# V = range(1,7)
# E = [(1, 2), (1, 2), (1, 3), (1, 5), (2, 2), (2, 3), (2, 4), (2, 5), (3, 4), (3, 6), (3, 6), (3, 6), (4, 4), (4, 6), (5, 2), (5, 3), (5, 4), (5, 6)]
# # V = ['u', 'v', 'w', 'z', ]
# # E = [('u', 'u'),('u', 'v'), ('u', 'v'), ('w', 'v'), ('w', 'u'), ('z', 'w')]

# G = nx.MultiDiGraph()
# G.add_nodes_from(V)
# G.add_edges_from(E)
# G = conditional_random_multi_digraph(20, .35, .12)
G = nx.fast_gnp_random_graph(100, .02, directed=True, seed=56789)
# G = scale_free_digraph(20)
# draw_digraph(G)
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

## Ground states are around: 10 * beta_c + 1.7763
beta_gd = 10 * beta_c + (52.265/100) * beta_c
beta_gd = 10.5 * beta_c

beta_min = beta_c + .0000001

beta = beta_c + 10.6

## Reference beta: 1.8

epsilon = (1.7/100) * beta 




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

# Gibbs = gibbs_profile(G, beta)
# x = Gibbs[0]
# ax = plt.gca()
# plt.imshow(Gibbs[1], cmap='seismic')
# ax.set_xticks(range(len(x)), labels=x)
# ax.set_yticks(range(len(x)), labels=x)
# plt.colorbar()
# plt.show()
# node_profile = node_gibbs_profile(G, 'c', beta)
# profile_entropy = node_profile_entropy(G, 'a', beta)
# entropies = node_profile_entropy_range(G, V, beta, beta + 3.)

# print(beta_c)
# # plot_node_emittance(G, ['u', 'v'], beta_min, beta, num=10000)
# plot_node_profile_entropy(G, list(G.nodes)[:6], beta_min, beta, num=10000)
# kms = KMS_emittance_dist_entropy_variation(G, beta_min, beta, num=100)
# plt.plot(kms['range'], kms['entropy'])
# plt.xlabel(f'1/ß')
# plt.ylabel('Emittance distribution entropy')

# plt.show()
reception = avg_reception_probability_variation(G, G.nodes, beta_min, beta, num=50)

data = pd.DataFrame(reception['avg_reception'], columns=G.nodes, index=[round(x, 4) for _, x in enumerate(reception['range'])])
sns.clustermap(data,
               metric='seuclidean',
               row_cluster=False,
               cmap='hot',
    dendrogram_ratio=(.1, .2),
    cbar_pos=(0, .2, .03, .4), figsize=(8, 6), annot_kws={'fontsize':12})
plt.show()
