import os
import sys
import re
import networkx as nx
import matplotlib.pyplot as plt 
import numpy as np 
import scipy.stats as stats
import pandas as pd
import csv
from openpyxl import load_workbook


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
from src.read_write import *


up_conn_f = open('../data/cook/herm_whole_connectome.csv', 'r', newline='')
syn_rows = [line.strip().split(',') for line in up_conn_f.readlines()[3:]]

synapses = []

for r in syn_rows:

    synapses += [(r[0], r[1]),] 

G = nx.MultiDiGraph()
G.add_edges_from(synapses)

print(len(G.nodes))