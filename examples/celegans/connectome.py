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
from src.kms_graphs import *
import seaborn as sns
# import fitter
from fitter import Fitter, get_common_distributions


np.random.seed(5679546)



neurons_f = '../data/NeuronType.xls'
dataf = '../data/NeuronConnect.xls'
name_neurons = '../data/name_neurons.txt'
neuron_pos_f = open('../data/neuron_positions.csv', 'r', newline='')



df_neurons = pd.read_excel(neurons_f)

df = pd.read_excel(dataf)



f = open(name_neurons, 'r', newline='')
# data = re.sub('[ ]+', ',', f.read())

n_lines = [re.sub('[ ]+', ',', line) for line in f.readlines()]
n_rows = [line.strip().split(',') for line in n_lines]

pos_rows = [l.strip().split(',') for l in neuron_pos_f.readlines()]
neuron_positions = {row[0]: (float(row[2]), float(row[3])) for row in pos_rows}


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

S_bar = conjugate_graph(S)


# pos = nx.kamada_kawai_layout(S)
# pos['AVAL'] = [-1,0]
# pos['AVAR'] = [1,0]

# draw_multi_digraph(S, pos=pos, node_colors=node_colors, node_labels={}, figsize=(10,10))
# plt.savefig('polySp_connectom.pdf', dpi=300, bbox_inches='tight')
# plt.show()

# G = configuration_model_from_directed_multigraph(S)

beta_c = critical_inverse_temperature(S) # 3.39842857192179

# betabar_c = critical_inverse_temperature(S_bar)

## Ground states are around: 10 * beta_c + 1.7763
# beta_gd = 10 * beta_c + (52.265/100) * beta_c
# beta_gd = 10.5 * beta_c

beta_min = beta_c - .0001

beta = 10.1*beta_c 

# ## Reference beta: 1.8

# # epsilon = (1.7/100) * beta 
# beta2 = np.log(np.exp(-10.5 * beta_c) - np.exp(-beta_c))
# epsilon = - np.log(1. - np.exp( - 0.94*beta_c))



# KMS1 = KMS_emittance_dist(S, 1.1*beta_c)
# # KMS2 = KMS_states(S, beta2)

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

# nodelist = ['AVAL', 'AVAR', 'ADAL', 'ADAR', 'FLPL', 'FLPR', 'RIAL', 'RIAR', 'IL2L', 'IL2R', 'AFDL', 'AFDR', 'DVA', 'DVB', 'DVC', 'DA01', 'DA09' , 'AIZL', 'AIZR'] \
#     + ['AVEL', 'AVER', 'AIYL', 'AIYR', 'AWCL', 'AWCR',\
#              'AVBL', 'AVBR', 'RIML', 'RIMR',  'RMDVL', 'RMDVR', 'PVCL', 'PVCR', 'PVDL', 'PVDR']
VA_neurons = [f'VA0{i}' for i in range(1,10)] + ['VA10', 'VA11', 'VA12']
VB_neurons = [f'VB0{i}' for i in range(1,10)] + ['VB10', 'VB11']
VD_neurons = [f'VD0{i}' for i in range(1,10)] + ['VD10', 'VD11', 'VD12', 'VD13']
DA_neurons = [f'DA0{i}' for i in range(1,10)]
AS_neurons = [f'AS0{i}' for i in range(1, 10)] + ['AS10', 'AS11']
DB_neurons = [f'DB0{i}' for i in range(1, 8)]
DD_neurons = [f'DD0{i}' for i in range(1, 7)]

other_motors = ['AVAL', 'AVAR', 'PVCL', 'PVCR', 'AVBL', 'AVBR']

thermotaxis_neurons = ['AFDL', 'AFDR', 'AWCL', 'AWCR', 'AIYL', 'AIYR', 'AIZL', 'AIZR', 'RIAL', 'RIAR']
chemotaxis_neurons = ['AIAL', 'AIAR', 'ASEL', 'ASER', 'PHAL', 'PHAR', 'PHBL', 'PHBR', 'ADFL', 'ADFR', 'ASIL', 'ASIR', 'ASKL', 'ASKR', 'ASGL', 'ASGR', 'ASJL', 'ASJR']

GABA_neurons = DD_neurons + VD_neurons + ['RMED', 'RMEV', 'RMEL', 'RMER', 'AVL', 'DVB', 'RIS']

# nodelist = list(S.nodes)
# plot_node_entropy(S, nodelist, beta_min, beta, num=5000)
# # draw_multi_digraph(G, layout=nx.kamada_kawai_layout)
plot_node_kms_emittance_profile_entropy(S, chemotaxis_neurons, .9*beta_c, beta, num=10000)
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
# e_thresh, w_thresh, KMSemit = beta_kms_digraph(S, .972*beta_c, entropy_ratio=.2, w_ratio=.01)

# print(f'thresh: {thresh}')
# # print([(v, u, KMSemit.get_edge_data(v, u)['weight']) for v, u in KMSemit.edges])
# A = nx.to_numpy_array(KMSemit, nodelist=KMSemit.nodes, dtype=np.float32)
# deg = dict(KMSemit.out_degree)
# # KMSAbs = nx.DiGraph()
# # KMSAbs.add_edges_from([(u, v) for (v, u) in KMSemit.edges])
# # pos = nx.random_layout(KMSemit)
# pos = {n: neuron_positions[n] for n in S.nodes}
# nx.draw_networkx(KMSemit, pos=pos, arrows=True, nodelist=list(deg.keys()), node_size=[(v+.3) * 20 for v in deg.values()], width=.4, with_labels=True, labels={n: n for n in list(deg.keys())}, node_color = [node_colors[n] for _, n in enumerate(list(deg.keys()))], alpha=.6,horizontalalignment='center', verticalalignment='center', edge_color="tab:gray", font_size=5)
# # K = nx.MultiDiGraph()
# # K.add_edges_from(KMSemit.edges)
# # draw_multi_digraph(K, layout=nx.kamada_kawai_layout, node_colors={n: node_colors[n] for n in K.nodes})
# # plt.savefig('./results/KMS-Absorbance_Beta_c.pdf', dpi=300, bbox_inches='tight')
# deg_data = [x[1] for x in KMSemit.out_degree]
# entro = kms_emittances_entropy(S, 1.0000001*beta_c)
# sns.histplot(entro)
# print(len(KMSemit.nodes))
# plt.imshow(A)
# plt.colorbar()
# H = node_structural_entropy(S, nodelist=thermotaxis_neurons)
# xs = [n for n in H]
# ys = [H[n] for _, n in enumerate(xs)]
# plt.scatter(xs, ys)
plt.legend(bbox_to_anchor=(1.05, 1.0), fontsize='10', loc='upper left')
plt.tight_layout()
plt.savefig('./results/neuronClass/Chemotaxis_StructureAndEmittanceEntropy.pdf', dpi=300, bbox_inches='tight')
# plt.show()

# print(sorted(KMSemit.out_degree, key=lambda x: x[1], reverse=True)[:10])

