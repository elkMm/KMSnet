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

from src.visuals import *
from src.utils import *
from src.states import *
from src.generators import *
from src.plotting import *
import seaborn as sns
# import fitter
from fitter import Fitter, get_common_distributions


np.random.seed(5679)



neurons_f = '../data/NeuronType.xls'
dataf = '../data/NeuronConnect.xls'
name_neurons = '../data/name_neurons.txt'



df_neurons = pd.read_excel(neurons_f)

df = pd.read_excel(dataf)



f = open(name_neurons, 'r', newline='')
# data = re.sub('[ ]+', ',', f.read())

n_lines = [re.sub('[ ]+', ',', line) for line in f.readlines()]
n_rows = [line.strip().split(',') for line in n_lines]



# neuron types
SE = 'se' # sensory
MO = 'mo' # motor neurons
IN = 'in' # interneurons
SEMO = 'se/mo'
SEIN = 'se/in'
MOSE = 'mo/se'
MOIN = 'mo/in'
INMO = 'in/mo'
INSE = 'in/se'

SE_color = 'tab:orange'
MO_color = 'tab:olive'
IN_color = 'tab:blue'

color_codes = {'se': SE_color, 'mo': MO_color, 'in': IN_color}

# SEMO_color = 'tab:magenta'
# SEIN_color = '


# # print(df.columns)
# neurons = [str(df_neurons['Neuron'][i]) for i in df_neurons.index]

neurons = []
node_colors = {}
color_maps = []
node_labels = {}

neuron_cls = []

for _, row in enumerate(n_rows):
    node = str(row[0])
    node_class = str(row[1])
    t = str(row[2])
    color = str([color_codes[k] for k in color_codes if t.startswith(k)][0])
    neurons.append(node)
    neuron_cls.append(node_class)
    color_maps.append(color)
    node_colors[node] = color
    node_labels[node] = node 
    node_colors.copy()
    node_labels.copy()

neuron_cls = list(set(neuron_cls))

# Put neurons in their corresponding neuron classes
neuron_cl_dict = {}
for cl in neuron_cls:
    neuron_cl_dict[cl] = [str(r[0]) for r in n_rows if str(r[1]) == str(cl)]
    neuron_cl_dict.copy()



# print(node_colors)

interNeurons = [row[0] for row in n_rows if row[2].startswith('in')]   
motorNeurons = [row[0] for row in n_rows if row[2].startswith('mo')]  
sensoryNeurons = [row[0] for row in n_rows if row[2].startswith('se')]   

communities = dict()
for neuron in neurons:
    if neuron in sensoryNeurons:
        communities[neuron] = 0

    if neuron in interNeurons:
        communities[neuron] = 1

    if neuron in motorNeurons:
        communities[neuron] = 2

synapses = []
# EJ_net = []

for i in df.index:
    row = [str(df['Neuron 1'][i]), str(df['Neuron 2'][i]), str(df['Type'][i]), int(df['Nbr'][i])]
    if row[2] in ['S', 'Sp']:
        new_edges = [(row[0], row[1]),] * row[3]
        synapses += new_edges



    # if df['Type'][i] =='EJ':
    #     EJ_net.append(edge)


S = nx.MultiDiGraph()
# S.add_nodes_from(neurons)
S.add_edges_from(synapses)


# pos = nx.kamada_kawai_layout(S)
# pos['AVAL'] = [-1,0]
# pos['AVAR'] = [1,0]

# draw_multi_digraph(S, pos=pos, node_colors=node_colors, node_labels={}, figsize=(10,10))
# plt.savefig('polySp_connectom.pdf', dpi=300, bbox_inches='tight')
# plt.show()

G = configuration_model_from_directed_multigraph(S)

beta_c = critical_inverse_temperature(G) # 3.39842857192179

## Ground states are around: 10 * beta_c + 1.7763
# beta_gd = 10 * beta_c + (52.265/100) * beta_c
# beta_gd = 10.5 * beta_c

beta_min = beta_c - .0001

beta = 10.1*beta_c 

# ## Reference beta: 1.8

# # epsilon = (1.7/100) * beta 
# beta2 = np.log(np.exp(-10.5 * beta_c) - np.exp(-beta_c))
# epsilon = - np.log(1. - np.exp( - 0.94*beta_c))



# KMS1 = KMS_emittance_dist(S, beta_min)
# KMS2 = KMS_states(S, beta2)

# sns.histplot(data=KMS1[1])
# plt.show()
# dist_fitter = Fitter(KMS1[1], distributions=get_common_distributions())
# dist_fitter.fit()
# summary = dist_fitter.summary()
# # print(summary)
# print(dist_fitter.fitted_param["powerlaw"])

# x = KMS1[0] #[i for i, _ in enumerate(KMS[0])]
# y = KMS1[1]
# y = y / sum(y)

# plt.subplot(1, 2, 1)
# sns.histplot(data=y)

# cmaps = [node_colors[n] for _, n in enumerate(x)]


# # # # # plt.subplot(1, 2, 2)
# plt.scatter(x, y, color=cmaps)
# plt.ylabel('Emittance')
# ax = plt.gca()
# for i, n in enumerate(x):
#     ax.annotate(node_labels[n], (x[i], y[i]), fontsize=7)



# ax.get_xaxis().set_visible(False)
# plt.show()
# print(beta_c)

# sample1 = KMS1[1] / sum(KMS1[1])
# sample2 = KMS2[1] / sum(KMS2[1])
# # print(stats.cramervonmises_2samp(sample1, sample2))
# # print(stats.cramervonmises_2samp(KMS1[1], KMS2[1]))
# print(epsilon)
# print(sum(y))
# Gibbs = gibbs_profile(G, beta)

# plt.imshow(Gibbs[1], cmap='seismic')
# plt.colorbar()

nodelist = ['AVAL', 'AVAR', 'ADAL', 'ADAR', 'FLPL', 'FLPR', 'IL2L', 'IL2R', 'PVCL', 'PVCR'] \
    + ['AVEL', 'AVER',\
             'AVBL', 'AVBR', 'RIML', 'RIMR', 'RIAL', 'RIAR', 'RMDVL', 'RMDVR']
# nodelist = list(S.nodes)
# plot_node_entropy(S, nodelist, beta_min, beta, num=5000)
# # draw_multi_digraph(G, layout=nx.kamada_kawai_layout)
# plot_node_profile_entropy(S, S.nodes, beta_min, beta, num=100)
# Gibbs = gibbs_profile(S, beta_c + .51)
# cols = Gibbs[0]
# mat = Gibbs[1]
# data = {}
# for i, u in enumerate(cols):
#     data[u] = list(mat[:, i])
#     data.copy()

# data = pd.DataFrame(data, columns=cols, index=cols)



# sns.clustermap(data, metric='jensenshannon', 
#             #    standard_scale=1, 
#             #    center=0,
#                cmap='hot',  # Figure sizes
#             #   dendrogram_ratio = 0.1,
#                dendrogram_ratio=(.1, .1),
#                 #    cbar_pos=(.02, .32, .03, .2),
#                 # cbar_pos=(.02, .32, .03, .2),
#                 cbar_pos=(0, .2, .006, .4),
#                     figsize=(30, 20))
# plt.imshow(Gibbs[1], cmap='hot')
# ax.set_xticks(range(len(nodelist)), labels=nodelist, rotation=45)
# ax.set_yticks(range(len(nodelist)), labels=nodelist, rotation=15)
# plt.colorbar()
# plt.xscale('symlog')
# kms = KMS_emittance_dist_entropy_variation(G, beta_min, beta, num=50)
# plt.plot(kms['range'], kms['entropy'])
# plt.xlabel(f'1/ß')
# plt.ylabel('Emittance distribution entropy')

# avg_profile = avg_node_profile(S, beta_min)[1]
# print(avg_profile)



# reception = avg_reception_probability_variation(S, S.nodes, beta_min, beta, num=40)

# data = pd.DataFrame(reception['avg_reception'], columns=S.nodes, index=[round(x, 4) for _, x in enumerate(reception['range'])])
# # sns.set(font_scale=.4)
# sns.clustermap(data,
#                metric='correlation',
#                row_cluster=False,
#             #    xticklabels=True,
#                cmap='hot',
#     dendrogram_ratio=(.1, .2),
#     cbar_pos=(0, .2, .01, .4), figsize=(20, 6))
# plt.ylabel('Temperature')
# sns.set_context("notebook", rc={"font.size": 4, 'font.family':'Helvetica'})

# plt.savefig('GibbsProfile_beta_c-.51.png', dpi=300, bbox_inches='tight')
# plt.show()

RIVL_syn = [s for s in synapses if s[0] == 'RIVL']

print(RIVL_syn)



