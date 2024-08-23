import os
import sys
import re
import csv


ex_dir = os.path.dirname(__file__)
cn_module_dir = os.path.join(ex_dir, '../..')
sys.path.append( cn_module_dir )

files = {
    'all3d': '../data/neuron_positions.csv',
    'skuhersky': '../data/skuhersky_high_resol_3d_positions.csv',
    'chemorep': '../data/chemorepulsion_pos.csv',
    'touchSens': '../data/touch_circuit_pos.csv',
    'thermotaxis': '../data/thermotaxis_pos.csv'
}

POSITIONS = {}

for f_name in files:
    pos_f = open(files[f_name], 'r', newline='')
    pos_rows = [l.strip().split(',') for l in pos_f.readlines()]
    f_positions = {row[0]: (float(row[2]), float(row[3])) for row in pos_rows}
    if f_name in ['all3d', 'skuhersky']:
         f_positions = {row[0]: (float(row[1]), float(row[2]), float(row[3])) for row in pos_rows}
    POSITIONS[f_name] = f_positions
    POSITIONS.copy()



anat_pos_f = open('../data/anatlas_neuron_positions.txt', 'r', newline='')

lines = [re.sub(r'[^\S\n\t]+', ',', line) for line in anat_pos_f.readlines()]
rows = [l.strip().split(',') for l in lines]

nlist = rows[0]

pos_lines = lines[1:]
nrows = [l.strip().split('\t') for l in pos_lines]
d3_pos = {}

for i, n in enumerate(nlist):
    row = nrows[i]
    d3_pos[n] = [float(row[0]),float(row[1]),float(row[2])]
    d3_pos.copy()


d3_pos = {n: d3_pos[n] for _, n in enumerate(sorted(nlist))}


POSITIONS.update({
    '3d': d3_pos
})