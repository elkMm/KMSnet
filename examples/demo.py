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


matrix = np.array([
    [.1, .2, 0, 4],
    [.001, .0123, 1e-10, 3],
    [2e-3, 1e-4, .092, 2],
    [.002, .092, -3e-13, 1.2]
])

# epsilon, A = matrix_thresholded_from_ratio(matrix, .1)

# print(f'threshold: {epsilon}\n')
# print(A)
print(is_qual(.2, .199999))