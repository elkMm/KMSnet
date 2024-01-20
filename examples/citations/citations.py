import os
import sys
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
import seaborn as sns
# import fitter
from fitter import Fitter, get_common_distributions


np.random.seed(5679)



cit_DBLP = '../data/cit-DBLP.edges'


f = open(cit_DBLP, 'r', newline='')
lines = f.readlines()
rows = [line.strip().split(' ') for line in lines[2:]]

E = []

for row in rows:
    e = (int(row[0]), int(row[1]))
    E.append(e)

G = nx.MultiDiGraph()

G.add_edges_from(E)

# draw_multi_digraph(G, layout=nx.kamada_kawai_layout)
beta_c = critical_inverse_temperature(G) # 3.39842857192179


beta = 5*beta_c

KMS = KMS_states(G, beta)
# KMS2 = KMS_states(S, beta2)
x = KMS[0]
y = KMS[1]

# print(beta_c)
sns.histplot(data=y)
plt.show()