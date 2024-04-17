import os
import sys
import re
import csv


ex_dir = os.path.dirname(__file__)
cn_module_dir = os.path.join(ex_dir, '../..')
sys.path.append( cn_module_dir )

files = {
    'chemorep': '../data/chemorepulsion_pos.csv',
    'touchSens': '../data/touch_circuit_pos.csv'
}

POSITIONS = {

}

for f_name in files:
    pos_f = open(files[f_name], 'r', newline='')
    pos_rows = [l.strip().split(',') for l in pos_f.readlines()]
    f_positions = {row[0]: (float(row[2]), float(row[3])) for row in pos_rows}
    POSITIONS[f_name] = f_positions

