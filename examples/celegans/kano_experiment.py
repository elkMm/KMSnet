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

# from connectome import neurons, neuron_cls, neuron_cl_dict, synapses, color_maps, node_colors
from cls_connectivity import cls_colors, cls_labels, C, beta_c
from src.visuals import *
from src.utils import *
from src.states import *
from src.generators import *
from src.plotting import *
from src.kms_graphs import *
from src.gen_colors import *

import seaborn as sns

np.random.seed(675679)

neuron_cls = list(C.nodes)

neuron_cls_thermo = ['AWC', 'AFD', 'AIY']

# files = {n: f'../data/kano/{n}_curve.csv' for n in neuron_cls_thermo}

# # AWC_file = '../data/kano/AWC_curve.csv'
# responses = {n: pd.read_csv(files[n]) for n in neuron_cls_thermo}

# # # AWC_coord = pd.read_csv(files['AWC'])

# xs = list(responses['AWC']['X'])

# AWC_resp = list(responses['AWC']['Y'])

# AFD_resp = list(responses['AFD']['Y'])
# AIY_resp = list(responses['AIY']['Y'])

### Save neuron responses into the same file
## Save only the somatic connectome

# calcium_sig_csv = open('../data/kano/AWC_AFD_AIY_CalciumSignals.csv', 'w', newline='')

# writer = csv.writer(calcium_sig_csv, delimiter=',')
# writer.writerow(['# Mean calcium signals in AFD, AWC, and AIY'])
# writer.writerow(['# Data extracted from fig 6-G in Kano et al. 2023'])
# writer.writerow(['temperature','AFD','AWC','AIY'])

# for i, x in enumerate(xs):
#     row = [float(x),float(AFD_resp[i]),float(AWC_resp[i]),float(AIY_resp[i])]
#     writer.writerow(row)


# plt.plot(xs, AFD_resp, label='AFD', color=COLORS[neuron_cls.index('AFD')])
# plt.plot(xs, AWC_resp, label='AWC', color=COLORS[neuron_cls.index('AWC')])
# plt.plot(xs, AIY_resp, label='AIY', color=COLORS[neuron_cls.index('AIY')])
# plt.xlabel(f'Physical Temperature')
# plt.ylabel(f'Calcium signals')
# plt.legend(bbox_to_anchor=(1.05, 1.0), fontsize='7', loc='upper left')
# plt.savefig('./results/kano/AFD_AWC_AIY_CaSignals.pdf', dpi=300, bbox_inches='tight')

## plot KMS receptances based on the Ca signals ibtained from Kano experimental data
calcium_sig_csv = open('../data/kano/AFD_AWC_AIY_CalciumSignals.csv', 'r', newline='')
data_rows = [l.strip().split(',') for l in calcium_sig_csv.readlines()[3:]]

physical_temps = []
AFD_AWC_calc_dist = []

for _, r in enumerate(data_rows):
    physical_temps += [float(r[0]),]
    AFD_y = float(r[1])
    AWC_y = float(r[2])
    AIY_y = float(r[3])
    q = AFD_y + AWC_y
    AFD_AWC_calc_dist += [[AFD_y , AWC_y, AIY_y],]

# # # print(AFD_AWC_calc_dist[:4])
target_categories = {
    'Motor': ['AVA', 'RMD', 'SMD', 'RIM', 'RIF', 'PVC', 'AVB', 'ASG', 'VA', 'VD', 'DA', 'DD','AS'],
    'AFDvsAWC': ['AFD', 'AWC'],
    'AIY': ['AIY'],
    'AIZ': ['AIZ'],
    'VD': ['VD'],
    'Thermo': ['AIA', 'AIB', 'AIZ'],
    'AIY-AIZ': ['AIY', 'AIZ'],
    'Integr': ['AIA', 'AIB', 'AIN'],
    'ASE': ['ASE'],
    'AVA': ['AVA'],
    'AFD': ['AFD'],
    'AWC': ['AWC'],
    'ThermoTaxi': ['AFD', 'AWC', 'AIY', 'AIZ', 'RIA'],
    'RIA': ['RIA'],
    'AIB': ['AIB'],
    'RMD': ['RMD'],
    'SMD': ['SMD']
}

# cat_name = 'AIY'
    
# targeted_neurons = target_categories[cat_name]
# YS = {n: [] for n in targeted_neurons}

# b_factor = 1.06

# emitters='AIY'

# beta_fmin = 1.01 * beta_c
# beta_fmax = 1.1 * beta_c
# theoret_temp = list(np.linspace(1/ beta_fmax, 1 / beta_fmin, num=90))

# randP = list(np.random.rand(90, 3))
# dists = AFD_AWC_calc_dist
# # randP = np.random.normal(0, .1, size=(90, 3))
# for k, t in enumerate(theoret_temp):
# # for _, d in enumerate(dists):
#     b = 1 / t
#     P = dists[k]
#     Con = group_kms_emittance_connectivity(C, b, ['AWC'], [P[1]])
#     for n in targeted_neurons:
#         recep = 0
#         if n in Con:
#             recep = Con[n]
#         YS[n] += [recep,]
#         YS.copy()

# for tn in targeted_neurons:
#     tn_color = COLORS[neuron_cls.index(tn)]
#     plt.plot(theoret_temp, YS[tn], label=tn, color=tn_color)
# plt.xlabel(f'Physical Temperature')
# plt.ylabel(f'KMS receptance from {emitters}')
# plt.legend(bbox_to_anchor=(1.05, 1.0), fontsize='7', loc='upper left')
# # plt.savefig(f'./results/kano/receptance/{cat_name}_ReceptanceFromAFD-AWC_{b_factor}xbeta_c.pdf', dpi=300, bbox_inches='tight')
# plt.show()




