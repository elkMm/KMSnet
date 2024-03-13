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

from cls_connectivity import *

from src.visuals import *
from src.utils import *
from src.states import *
from src.generators import *
from src.plotting import *
from src.kms_graphs import *
from src.ca import *

import seaborn as sns





# vertices, Z = kms_emittance(C, 1.1*beta_c+.001)
# anchors = locomorory_circuit

# cols = {}
# for v in anchors:
#     inde = vertices.index(v)
#     prof = Z[:, inde]
#     prof[inde] = 0.
    
#     # Remove numbers close to zero 
#     prof = [get_true_val(x) for _, x in enumerate(prof)]
#     s = sum(prof)
#     if s == 0.:
#         s = 1.
#     cols[v] = [x / float(s) for _, x in enumerate(prof)]
#     cols.copy()

# save data in csv
fname = f'./results/data/Locomotory_1.1xbeta_c+.001.csv'    
# keys = list(cols.keys())
# with open(fname, 'w', newline='') as csvfile:
#     table = csv.writer(csvfile, delimiter=',')
#     table.writerow(['',] + [str(key) for _, key in enumerate(keys)])
#     for i, u in enumerate(vertices):
#         nums = [float(cols[key][i]) for _, key in enumerate(keys)]
#         if sum(nums) > 0.:
#             row = [str(u),] + nums
#             table.writerow(row) 



transfer = CA()

emitts = pd.read_csv(fname, index_col=0)

# print(emitts.head(10))

transfer.fit(emitts)

pcs_row, pcs_col = \
    transfer.get_princpl_coords_df(row_categories=emitts.index,
                                   col_categories=emitts.columns)
pcs_row['Dim 1'] = -pcs_row['Dim 1']
pcs_col['Dim 1'] = -pcs_col['Dim 1']
print('Principal coordinates of row variables in DataFrame:')
print(pcs_row)
print(pcs_col)

variances = transfer.principal_inertias_
percent_explnd_var = (variances / variances.sum()) * 100

fig, ax = plt.subplots()
sns.barplot(x=np.arange(1, 5), y=percent_explnd_var[:4], ax=ax)
ax.set_xlabel('Dimensions')
ax.set_ylabel('Percentage of explained variances')
var_text = ['{:.1f}%'.format(pers) for pers in percent_explnd_var[:4]]
for i, txt in enumerate(var_text):
    ax.annotate(txt, (i, percent_explnd_var[i]),
                horizontalalignment='center',
                verticalalignment='center')
plt.show()

fig, ax = plt.subplots()
sns.regplot(x='Dim 0', y='Dim 1', data=pcs_row, fit_reg=False, ax=ax)
sns.regplot(x='Dim 0', y='Dim 1', data=pcs_col, fit_reg=False, ax=ax)
# plt.plot([0]*len(pcs_row), [0]*len(pcs_row), color='b')
for i, txt in enumerate(list(emitts.index)):
    ax.annotate(txt, (pcs_row.iloc[i]['Dim 0'], pcs_row.iloc[i]['Dim 1']))
for i, txt in enumerate(list(emitts.columns)):
    ax.annotate(txt, (pcs_col.iloc[i]['Dim 0'], pcs_col.iloc[i]['Dim 1']))
ax.set_xlabel('Dim 0')
ax.set_ylabel('Dim 1')
plt.show()

