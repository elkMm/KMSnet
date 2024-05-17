import os
import sys
import re
import networkx as nx
import matplotlib.pyplot as plt 
import numpy as np 
import pandas as pd
import csv, json


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


# np.random.seed(79546)



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

light_green = '#abf7c0'
light_purple = '#ff75ff'
light_blue = '#5abde0'
light_orange = 'tab:orange'


lightgreen = '#a4f5ea'
blue_sky = '#71aeeb'
light_red_gray = '#d67d6f'
light_gray_yellow = '#f0e086'


SE_color = light_green
IN_color = light_gray_yellow
MO_color = light_purple


color_codes = {'se': SE_color, 'mo': MO_color, 'in': IN_color}



# SEMO_color = 'tab:magenta'
# SEIN_color = '


# # print(df.columns)
# neurons = [str(df_neurons['Neuron'][i]) for i in df_neurons.index]

pharynx_neurons = ['I1L', 'I1R', 'I2L',	'I2R', 'I3', 'I4', 'I5', 'I6', 'M1', 'M2L', 'M2R', 'M3L', 'M3R', 'M4', 'M5', 'MCL', 'MCR', 'MI']

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
# neuron_cl_dict = {}
# for cl in neuron_cls:
#     neuron_cl_dict[cl] = [str(r[0]) for r in n_rows if str(r[1]) == str(cl)]
#     neuron_cl_dict.copy()

neuron_by_classes_f = open('../data/neurons_by_classes.json', 'r')
neuron_by_classes = json.load(neuron_by_classes_f)


pharyngeal = ['M1', 'M2', 'M3', 'M4', 'M5', 'I1', 'I2', 'I3', 'I4', 'I5', 'I6', 'MI', 'NSM', 'MC', 'CAN']

neuron_by_classes = {nc: neuron_by_classes[nc] for nc in neuron_by_classes if nc not in pharyngeal}
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
    if row[2] in ['S', 'Sp', 'EJ']:
        new_edges = [(row[0], row[1]),] * row[3]
        synapses += new_edges
        


additional_synapses = [('AWCR', 'AFDR')]
func_links = [('AFDR', 'AIZR'), ('AFDL', 'AIZL'), ('AWCL', 'AIZL')]

# synapses = synapses + func_links
# sub = [e for e in synapses if (e[0] == 'AFDR') and (e[1] == 'AIYR')]
# synapses = [e for e in synapses if e not in sub]

    # if df['Type'][i] =='EJ':
    #     EJ_net.append(edge)


S = nx.MultiDiGraph()
S.add_nodes_from(['VC06'])
S.add_edges_from(synapses)

# S_bar = conjugate_graph(S)


# pos = nx.kamada_kawai_layout(S)
# pos['AVAL'] = [-1,0]
# pos['AVAR'] = [1,0]

# draw_multi_digraph(S, pos=neuron_positions, node_colors=node_colors, node_labels={}, figsize=(10,10))
# # plt.savefig('polySp_connectom.pdf', dpi=300, bbox_inches='tight')
# plt.show()

# G = configuration_model_from_directed_multigraph(S)

beta_c = critical_inverse_temperature(S) # 3.998629949388744

# betabar_c = critical_inverse_temperature(S_bar)

## Ground states are around: 10 * beta_c + 1.7763
# beta_gd = 10 * beta_c + (52.265/100) * beta_c
# beta_gd = 10.5 * beta_c

beta_min = beta_c + .000001

beta = 3.5*beta_c 

beta_f = 1.06 * beta_c

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

command_interneurons = ['AVAL', 'AVAR', 'PVCL', 'PVCR', 'AVBL', 'AVBR', 'AVDL', 'AVDR']

thermotaxis_neurons = ['AFDL', 'AFDR', 'AWCL', 'AWCR', 'AIYL', 'AIYR', 'AIZL', 'AIZR', 'RIAL', 'RIAR']
chemotaxis_neurons = ['AIAL', 'AIAR', 'ASEL', 'ASER', 'PHAL', 'PHAR', 'PHBL', 'PHBR', 'ADFL', 'ADFR', 'ASIL', 'ASIR', 'ASKL', 'ASKR', 'ASGL', 'ASGR', 'ASJL', 'ASJR']

GABA_neurons = DD_neurons + VD_neurons + ['RMED', 'RMEV', 'RMEL', 'RMER', 'AVL', 'DVB', 'RIS']
thermosensory_neurons = thermotaxis_neurons + ['RMDDL', 'RMDDR', 'RMDL', 'RMDR', 'RMDVL', 'RMDVR', 'AVEL', 'AVER', 'AIBL', 'AIBR']

locomotory_neurons = VA_neurons + VB_neurons + VD_neurons + DA_neurons + DB_neurons + DD_neurons + AS_neurons + command_interneurons + ['PDB']

olfactory_thermo_neurons = ['AFDL', 'AFDR', 'AWAL', 'AWAR', 'AWBL', 'AWBR', 'AWCL', 'AWCR', 'ADLL', 'ADLR', 'ASHL', 'ASHR', 'AVHL', 'AVHR'] + ['AIYL', 'AIYR', 'AIZL', 'AIZR', 'AIAL', 'AIAR', 'AIBL', 'AIBR', 'RIFL', 'RIFR', 'AVDL', 'AVDR', 'AVAL', 'AVAR', 'AVBL', 'AVBR']

olfactory_neurons = ['AFDL', 'AFDR', 'AWAL', 'AWAR', 'AWBL', 'AWBR', 'AWCL', 'AWCR', 'ADLL', 'ADLR', 'ASHL', 'ASHR', 'AVHL', 'AVHR', 'AIBL', 'AIBR', 'AIYL', 'AIYR', 'AIZL', 'AIZR']

mechanoreceptor_neurons = ['ALML', 'ALMR', 'PLML', 'PLMR', 'AVM', 'PVM', 'PVDL', 'PVDR', 'ADEL', 'ADER', 'PDEL', 'PDER', 'ASHL', 'ASHR', 'FLPL', 'FLPR', 'OLQDL', 'OLQDR', 'OLQVL', 'OLQVR', 'CEPDL', 'CEPDR', 'CEPVL', 'CEPVR', 
'IL1L', 'IL1R', 'IL1DL', 'IL1DR', 'IL1VL', 'IL1VR']

chemorepulsion_neurons = ['AVAL', 'AVAR', 'AVDL', 'AVDR', 'AVBL', 'AVBR', 'PVCL', 'PVCR', 'PHAL', 'PHAR', 'PHBL', 'PHBR', 'ASKL', 'ASKR', 'ASHL', 'ASHR', 'AIAL', 'AIAR', 'AIBL', 'AIBR']

touch_sensitive_neurons = VA_neurons + DA_neurons + VB_neurons + DB_neurons + \
     AS_neurons + command_interneurons + ['ALML', 'ALMR', 'AVM', 'PLML', 'PLMR', 'LUAL', 'LUAR']


extended_thermotaxis = thermotaxis_neurons + ['AIAL', 'AIAR', 'AIBL', 'AIBR', 'ASIL', 'ASIR']


CIRCUITS = {
    'CommandInterneurons': command_interneurons,
    'Thermotaxis': thermotaxis_neurons,
    'ExtendedThermotaxis': extended_thermotaxis,
    'ThermoSensory': thermosensory_neurons,
    'Locomotory': locomotory_neurons,
    'OlfactoryThermo': olfactory_thermo_neurons,
    'MechanoSensory': mechanoreceptor_neurons,
    'GABA': GABA_neurons,
    'Chemotaxis': chemotaxis_neurons,
    'Olfactory': olfactory_neurons,
    'Chemorepulsion': chemorepulsion_neurons,
    'TouchInducedMovement': touch_sensitive_neurons,
    'VA': VA_neurons,
    'VB': VB_neurons,
    'VD': VD_neurons,
    'DA': DA_neurons,
    'DB': DB_neurons,
    'DD': DD_neurons,
    'AS': AS_neurons,
    'AFD': ['AFDL', 'AFDR'],
    'AWC': ['AWCL', 'AWCR'],
    'AIA': ['AIAL', 'AIAR'],
    'AIB': ['AIBL', 'AIBR'],
    'ASI': ['ASIL', 'ASIR'],
    'AIY': ['AIYL', 'AIYR'],
    'AIZ': ['AIZL', 'AIZR'],
    'RIA': ['RIAL', 'RIAR'],
    'RIB': ['RIBL', 'RIBR'],
    'ASE': ['ASEL', 'ASER']
}


# nodelist = list(S.nodes)
# plot_node_entropy(S, nodelist, beta_min, beta, num=5000)
# # draw_multi_digraph(G, layout=nx.kamada_kawai_layout)
# plot_node_kms_emittance_profile_entropy(S, thermosensory_neurons, .9*beta_c, beta, num=10000)
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


# plot_sates_fidelity(S, chemotaxis_neurons, beta_min, beta, num=1000)

# plot_kms_simplex_volume(S, beta_min, beta, num=50)
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
# plt.legend(bbox_to_anchor=(1.05, 1.0), fontsize='10', loc='upper left')
# plt.tight_layout()
# plt.savefig('./results/circuits/ThermoSensory_StructureAndEmittanceEntropy.pdf', dpi=300, bbox_inches='tight')
# plt.show()
# print(sorted(S.in_degree, key=lambda x: x[1], reverse=True)[:10])

# print(S.in_degree(['AVAL', 'AVAR', 'RIAL', 'RIAR']))
# beta_afdr = 2.1 * beta_c + .0001 # beta_afdr = 8.39722289371636, T = 0.11908699014626606
# beta_afdl = 2.1 * beta_c + .0001 # beta_afdl = 8.39722289371636, T = 0.11908699014626606
# beta_awcr = 2.1 * beta_c + .0001 # beta_awcr = 8.39722289371636, T = 0.11908699014626606
# beta_aiyr = 2.1 * beta_c + .0001 # beta_aiyr = 8.39722289371636, T = 0.11908699014626606
# beta_riar = 2.1 * beta_c + .0001 # beta_riar = 8.39722289371636, T = 0.11908699014626606
# beta_aizr = 2.1 * beta_c + .0001 # beta_aizr = 8.39722289371636, T = 0.11908699014626606
# beta_avar = 2.1 * beta_c + .0001 # beta_avar = 8.39722289371636, T = 0.11908699014626606






# print(f'beta_c: {beta_c}')
# print(f'beta_afd: {beta_afd}')
# print(f'beta_ava: {beta_ava}')
# print(f'beta: {beta_avar}')
# print(f'temperature: {1./beta_avar}')

# # Individual Neuron functional circuits
# vertices, Z = kms_emittance(S, beta_avar)
# vertices2, SS = node_structural_state(S)
# center = 'AVG'
# inde = vertices2.index(center)
# prof = Z[:, inde]
# struc = SS[center]

# print(f'JSD: {jsd}')
# print(f'nodes: {vertices2}\n {struc}\n')
# print(f'nodes: {vertices}\n {prof}')
# A = adjacency_matrix(C, nodes=nodes)
# A = A.T
# i = nodes.index('ADF')

# print(f'{vertices2}\n{prof}\n')

# prof[inde] = 0.
# struc[inde] = 0.
# # # # Remove numbers close to zero 
# # prof = [get_true_val(x) for _, x in enumerate(prof)]
# s = sum(prof)
# if s == 0.:
#     s = 1.
# prof = [x / float(s) for _, x in enumerate(prof)]

# ss = sum(struc)
# if ss == 0.:
#     ss = 1.
# struc = [x / float(ss) for _, x in enumerate(struc)]

# fid = fidelity(struc, prof)
# print([(n, struc[vertices2.index(n)]) for n in vertices2 if struc[vertices2.index(n)] != 0], '\n')

# print([(n, prof[vertices2.index(n)]) for n in vertices2 if prof[vertices2.index(n)] != 0])
# print(fid)
# # transform the range into [0,1]
# a = float(min(prof))
# b = float(max(prof))
# d = 1./(b - a)
# prof = [(x - a) * d for _, x in enumerate(prof)]


# ax = plt.gca()
# Select only the neurons with non-zero receptance
# for p, nv in enumerate(vertices):
#     if prof[p] > 0.:
#         plt.scatter(nv, prof[p], color=node_colors[nv])
#         ax.annotate(node_labels[nv], (nv, prof[p]), fontsize=7)

#
# for i, n in enumerate(vertices):
#     ax.annotate(cls_labels[n], (vertices[i], prof[i]), fontsize=7)



# ax.get_xaxis().set_visible(False)

# plt.xlabel(f'Nodes')
# plt.ylabel(f'{center} KMS intensity')


# plot_feedback_coef(S, locomotory_neurons, beta_min, beta, colors=COLORS2, num=1000)
# plt.legend(bbox_to_anchor=(1.05, 1.0), fontsize='7', loc='upper left')

# plt.savefig('./results/coefficients/Locomotory_feedback.pdf', dpi=300, bbox_inches='tight')
# plt.savefig('./results/fid/Locomotory_fidelity.pdf', dpi=300, bbox_inches='tight')
# plt.show()


#### Draw structural circuits
# name = 'ThermoSensory'
# # for name in CIRCUITS:
# V = CIRCUITS[name]
# K = weighted_structural_subgraph(S, nodelist=V)
# draw_weighted_digraph(K, pos={n: neuron_positions[n] for n in K.nodes}, node_size=1.6, node_colors={n: node_colors[n] for n in K.nodes}, node_labels={n: node_labels[n] for n in K.nodes}, edgecolor='silver', font_size=7, figsize=(10,6))
# # plt.show()
# plt.savefig(f'./results/nets/struc/{name}_circuit.pdf', dpi=300, bbox_inches='tight')
# plt.close()
# V = olfactory_thermo_neurons
# K = nx.MultiDiGraph()
# K.add_nodes_from(V)
# K.add_edges_from([e for e in synapses if ((e[0] in V) and (e[1] in V))])
# draw_multi_digraph(K, pos={n: neuron_positions[n] for n in K.nodes}, node_size=.4, node_colors={n: node_colors[n] for n in K.nodes}, node_labels={n: node_labels[n] for n in K.nodes}, edgecolor='silver', font_size=11, figsize=(10,6))
# plt.savefig('./results/nets/struc/OlfactoryThermo_circuit.pdf', dpi=300, bbox_inches='tight')
# plt.show()


# weights and ratios
# name = 'Thermotaxis'
# V = CIRCUITS[name]
# # # w = node_kms_in_weight(S, V, beta_min, beta, num=20)

# # # print(w['RIAR'])
# plot_feedback_coef_variation(S, V, beta_min, beta, num=1000, colors=COLORS2, font_size=10)
# plt.legend(bbox_to_anchor=(1.05, 1.0), fontsize='8', loc='upper left')

# # plt.savefig(f'./results/fid/{name}_fidelity.pdf', dpi=300, bbox_inches='tight')
# plt.savefig(f'./results/coefficients/{name}_feedback.pdf', dpi=300, bbox_inches='tight')
# plt.close()
# plt.show()

# rank = beta_kms_receptance_ranking(S, beta_f)
# rank_file = open('./results/data/coefs/kms-receptance-ranking_1.06xbeta_c.csv', 'w', newline='')
# writer = csv.writer(rank_file, delimiter=',')
# for key in rank:
#     row = [key,rank[key]]
#     writer.writerow(row)


# KMS subgraphs
# b_factors = [.9, .98, 1.001, 1.01, 1.06, 1.1, 1.2, 1.5, 1.8, 1.9, 2., 2.1, 2.4, 2.6, 2.8, 3., 3.1, 3.2, 3.5, 3.8, 4.] 
# # b_factors = [1.1]
# # b_factor = 1.06
# name = 'ThermoSensory'
# V = CIRCUITS[name]
# for b_factor in b_factors:
#     b = b_factor * beta_c
#     # V = olfactory_thermo_neurons
#     K = kms_weighted_subgraph(S, b, nodelist=V)

# # # print(list(K.edges(data=True))[0][2]['weight'])
#     draw_weighted_digraph(K, pos={n: neuron_positions[n] for n in K.nodes}, node_size=1.6, node_colors={n: node_colors[n] for n in K.nodes}, node_labels={n: node_labels[n] for n in K.nodes}, edgecolor=blue_sky, font_size=8, figsize=(10,6))
#     plt.savefig(f'./results/nets/kms_graphs/func_circuits/{name}_KMS-subgraph-{b_factor}xbeta_c.pdf', dpi=300, bbox_inches='tight')
#     plt.close()
#     plt.show()

### 
# beta_f = 1.06 * beta_c # = 4.238547746352069 optimal functional beta?
## T_f = 1/beta_f = 0.2359298655679073


# plt.hist(list(rank.values()), bins=10)
# plt.show()

# print(rank)
# print([e for e in synapses if e[1] == 'PVDL'])

## Node KMS weighted connectity
# b_factor = 1.01
# emitter = 'AFDR'
# Con = node_kms_emittance_connectivity(S, emitter, b_factor*beta_c)
# # # draw_weighted_digraph(K, pos={n: neuron_positions[n] for n in K.nodes}, node_size=.4, node_colors={n: node_colors[n] for n in K.nodes}, node_labels={n: node_labels[n] for n in K.nodes}, edgecolor='tan', font_size=10, figsize=(10,6))
# # # plt.show()

# ax = plt.gca()
# ys = [k for k in Con.keys()]

# # # # Select only the neurons with non-zero receptance
# for nv in ys:
#     plt.scatter(nv, Con[nv], color=node_colors[nv])
#     ax.annotate(node_labels[nv], (nv, Con[nv]), fontsize=7)

# #
# # for i, n in enumerate(vertices):
# #     ax.annotate(cls_labels[n], (vertices[i], prof[i]), fontsize=7)



# ax.get_xaxis().set_visible(False)

# plt.xlabel(f'Nodes')
# plt.ylabel(f'{emitter} KMS intensity')
# plt.show()
# plt.savefig(f'./results/neuronKMS/{emitter}_KMS-{b_factor}xbeta_c.pdf', dpi=300, bbox_inches='tight')
#     plt.close()

# print([e for e in synapses if (e[0] == e[1])])

# print('VC06' in list(S.nodes))
# print(len(S.nodes))
# print(S.in_degree('AIAL'), S.out_degree('AIAL'))


# Node to node flow streams
# remove_nodes = ['AFDR', 'AFDL']
# S.remove_edges_from([e for e in synapses if (e[0] in remove_nodes) or (e[1] in remove_nodes)])
# name = 'Thermotaxis'
# sources = ['AIYR', 'AIYL']
# linestyle = ['-', ':', '--', '-.']
# # Emit = node_kms_emittance_profile(S, )
# # targets = ['AWBL', 'AWBR', 'ADLL', 'ADLR', 'URXL', 'URXR']
# targets = CIRCUITS[name]
# # targets = ['AFDR', 'AFDL', 'AWCR', 'AWCL', 'AIYR', 'AIZR', 'AIZL', 'RIAR', 'RIAL']  # 

# plot_node_kms_stream_variation(S, sources, targets, beta_min, beta, linestyle=linestyle, num=1000, colors=COLORS2, node_labels=node_labels)
# streams = node_to_node_kms_flow_stream(S, source, targets, beta_min, beta)

# xs = streams['range']
# for u in targets:
#     ys = streams[u]
#     plt.plot(xs, ys, '--', label=f'{source}-->{u}')
# plt.legend(bbox_to_anchor=(1.05, 1.0), fontsize='8', loc='upper left')
# # plt.tight_layout()
# # plt.savefig(f'./results/circuits/streams/{sources}_to_{name}_stream.pdf', dpi=300, bbox_inches='tight')
# plt.show()


## Clustering based on Node flow streams
# name = 'Thermotaxis'
# source = 'AWCL'

# # targets = ['AWBL', 'AWBR', 'ADLL', 'ADLR', 'URXL', 'URXR']
# targets = CIRCUITS[name]
# # ax = plt.gca()


# #                 )
# plot_node_kms_streams_clustering(S, source, targets, beta_min, beta, num=100, font_size=8)
# # plt.imshow(data, cmap='hot')
# # ax.set_xticks(range(len(targets)), labels=targets, rotation=45)
# # ax.set_yticks(range(len(streams['range'])), labels=streams['range'], rotation=15)
# plt.savefig(f'./results/circuits/clustering/{source}_to_{name}_stream.pdf', dpi=300, bbox_inches='tight')
# plt.tight_layout()

# plt.show()
# print(data.head())


### Receptance entropies
# name = 'Thermotaxis'
# NOIS = CIRCUITS[name] # Neurons of interests
# plot_feedback_coef_variation(S, NOIS, beta_min, beta, num=1000, colors=COLORS2, node_labels=node_labels)
# plt.legend(bbox_to_anchor=(1.05, 1.0), fontsize='8', loc='upper left')
# plt.tight_layout()
# # plt.savefig(f'./results/circuits/entropy/{name}_receptance_entropy.pdf', dpi=300, bbox_inches='tight')
# # plt.savefig(f'./results/circuits/entropy/{name}_emittance_entropy.pdf', dpi=300, bbox_inches='tight')
# # plt.savefig(f'./results/circuits/indices/{name}_feedback_coef.pdf', dpi=300, bbox_inches='tight')
# plt.show()

# afd = ['AFDR', 'AFDL']
# AFD_links = [e for e in synapses if (e[0] in afd) or (e[1] in afd)]
# print(len(AFD_links))    

# print(len(synapses))



### Emittance entropies
# name = 'Thermotaxis'
# NOIS = CIRCUITS[name] # Neurons of interests
# plot_node_kms_emittance_profile_entropy(S, NOIS, beta_min, beta, num=1000, colors=COLORS2, node_labels=node_labels)
# plt.legend(bbox_to_anchor=(1.05, 1.0), fontsize='8', loc='upper left')
# # plt.tight_layout()
# # plt.savefig(f'./results/newConnectome/entropy/{name}_emittance_entropy.pdf', dpi=300, bbox_inches='tight')
# # # plt.savefig(f'./results/circuits/entropy/{name}_emittance_entropy.pdf', dpi=300, bbox_inches='tight')
# # # plt.savefig(f'./results/circuits/indices/{name}_feedback_coef.pdf', dpi=300, bbox_inches='tight')
# plt.show()
