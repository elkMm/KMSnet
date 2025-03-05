import json 
from scipy.cluster.hierarchy import linkage, dendrogram, to_tree, fcluster, fclusterdata
from mixed_connectome import *

neurons = list(G.nodes)
# interNeurons = [n for n in interNeurons if n in neurons]
# sensoryNeurons = [n for n in sensoryNeurons if n in neurons]
# motorNeurons = [n for n in motorNeurons if n in neurons]


GANGLIA =  json.load(open('../data/randi_aconnect_ids_ganglia.json'))


#### neuron types in json
# neuron_types = {
#     'sensory': sensoryNeurons,
#     'inter': interNeurons,
#     'motor': motorNeurons
# }
NEURONTYPES = json.load(open('../data/neuron_types.json'))


#### Divide the list of neurons into 28 groups of 10 neurons
# L = sorted(list(G.nodes))
# GROUP = {}
# for i in range(0, 28):
#     GROUP[i] = L[i * 10:(i+1) * 10]
#     GROUP.copy()

#### SFD groups 




##### Structural connecvity adj list ########
# name = 'all'
# sources = CIRCUITS[name]
# sources = sorted(list(G.nodes))
# adj_list = structural_connectivity_adj_list(G, sources)

# struc_conn_f = open(f'./results/data/struc_connectivity/{name}_structural_connectivity.csv', 'w', newline='')
# writer = csv.writer(struc_conn_f, delimiter=',')
# writer.writerow([f'# Structural connectivity of {name}'])
# writer.writerow(['source','target','weight'])

# for we in adj_list:
#     e = [str(we[0]),str(we[1]),float(we[2])]
#     writer.writerow(e)

######## Save connectome as adg list for a directed multigraph

# fname = f'../data/connectivity/Supplementary-Data1_connectome_multidigraph.txt'
# data1 = open(fname, 'w')
# data1.write("source,target\n")
# for e in G.edges:
#     data1.write(f'{str(e[0])},{str(e[1])}\n')
################### Struct connectivity
# name = 'AFD'
# name = 'all'
# # # # source = 'AFDR'
# # # # NOI = 'RMDVR'
# fname = f'./results/data/struc_connectivity/{name}_structural_connectivity.csv'


# struc_con_f = open(fname, 'r', newline='')
# struc_con_rows = [line.strip().split(',') for line in struc_con_f.readlines()[2:]]

# struc_weighted_connections = []

# for line in struc_con_rows:
#     struc_weighted_connections += [(str(line[0]), str(line[1]), float(line[2])),]



# connect = [e for e in struc_weighted_connections if e[0] == source]

# struc = [e[1] for e in connect]

# k1 = len(struc) + 1
# # k2 = len(func)
# radius = 16.
# # r2 = 20.  

# # con_list = [(k1, struc, r1), (k2, func, r2)]

# node_coords = [(source, 0.0, 0.0)]

# angle = radius * (np.pi / float(k1)) 
# for i, node in enumerate(struc, start=1):
#     x = radius * np.cos(i * angle)
#     y = radius  * np.sin(i * angle)
#     node_coords.append((node, x, y))

# pos = {str(co[0]): (float(co[1]), float(co[2])) for co in node_coords}
# pos[NOI] = (radius, 0.)

# K = nx.DiGraph()
# K.add_weighted_edges_from(connect)

# node_shape={n: 'polygon' for n in K.nodes}
# node_shape[source] = 'round'

# draw_weighted_digraph(K, pos=pos, node_size=4.1, node_colors={n: node_colors[n] for n in K.nodes}, node_shape=node_shape , node_labels={n: node_labels[n] for n in K.nodes}, arrowstyle='anat', arrow_shrink=56, edgecolor='silver', font_size=20, figsize=(10,10))
# plt.show()
# plt.savefig(f'./results/newConnectome/nets/struc/{source}_structural_connectivity.png', transparent=True, dpi=300, bbox_inches='tight')


####### Thermotaxis
# name = 'ExtendedThermotaxis'
# # for name in CIRCUITS:
# V = CIRCUITS[name]
# K = weighted_structural_subgraph(G, nodelist=V)
# draw_weighted_digraph(K, pos={n: POSITIONS['thermotaxis'][n] for n in K.nodes}, node_shape={n: 'polygon' for n in K.nodes}, node_size=4., node_colors={n: node_colors[n] for n in K.nodes}, node_labels={n: node_labels[n] for n in K.nodes}, arrowstyle='anat', arrow_shrink=42., edgecolor='silver', font_size=20, figsize=(12,12))
# # plt.show()
# plt.savefig(f'./results/newConnectome/nets/struc/{name}_circuit.pdf', dpi=300, bbox_inches='tight')
# # # plt.close()

######## KMS SUBGRAPHS 
# name = 'ExtendedThermotaxis'
# name = 'RIA'
# # sources = ['RMDVR']
# targets = CIRCUITS[name]
# sources = list(G.nodes)
# # rmv_edges = [('AFDR', 'RMDVR')]

# # # # # # # # factors = [.01, .06, .07, .09, 1.1, 1.5, 1.9, 2.]
# b_factor = 1.05
# # # # # # # # b_factor = 1.035
# NITER = 5000
# fname = f'./results/data/kms_connectivity/{name}_kms_connectivity_{b_factor}xbeta_c_{NITER}-iters.csv'
# generate_kms_connectivity(G, b_factor, sources, targets, s_name='all', t_name='', n_iter=NITER, fname=fname)

# ### surgery
# fname = f'./results/data/kms_connectivity/surgery/{name}_kms_connect_E{rmv_edges}-rmvd_{b_factor}xbc_{NITER}-iters.csv'
# generate_kms_connectivity(G, b_factor, sources, targets, s_name=name, t_name='', removed_edges=rmv_edges, removed_edges_name='(AFDR,RMVDR)', n_iter=NITER, fname=fname)


# alpha = .05
# kms_conn_f = open(fname, 'r', newline='')

# # # print(regular_polygon_coord(10, 4.))

# kms_con_rows = [line.strip().split(',') for line in kms_conn_f.readlines()[2:]]

# kms_weighted_connections = []

# for _, line in enumerate(kms_con_rows):
#     if float(line[3]) <= alpha:
#         kms_weighted_connections += [(str(line[0]), str(line[1]), float(line[2])),] 


# K = nx.DiGraph()
# K.add_weighted_edges_from(kms_weighted_connections)

# # # print([e for e in kms_weighted_connections if e[0] == 'AFDR'])
# node_shape = {n: 'polygon' for n in K.nodes}




# draw_weighted_digraph(K, pos={n: POSITIONS['thermotaxis'][n] for n in K.nodes}, node_shape=node_shape, node_size=4., node_colors={n: node_colors[n] for n in K.nodes}, node_labels={n: node_labels[n] for n in K.nodes}, arrow_shrink=48, edgecolor=func_edge_color, font_size=20, figsize=(12,12))
# plt.show()
# plt.savefig(f'./results/newConnectome/nets/func/{name}_kms_circuit_{b_factor}xbeta_c_{NITER}-iters.pdf', dpi=300, bbox_inches='tight')

######### KMS CONNECTIVITY ######
# from datetime import datetime
# name = 'AFD'
# s = 'AFDR'
# NOI = 'RMDVR'
# # # exceptions = ['ADLL','ADLR','AWBL', 'AWBR', 'URXR']
# # # # # source = CIRCUITS[name]
# # # # G.add_edges_from([('RID', 'ALA'), ('ALA', 'ADEL'), ('ADEL', 'URXL')]*10)
# # # G.add_edges_from([('RID', 'URXL')])
# # # # sources = ['RID']
# # sources = sorted(list(G.nodes))
# # targets = sources
# # # # # # # # # # # # # factors = [.01, .06, .07, .09, 1.1, 1.5, 1.9, 2.]
# b_factor = 1.05
# # # # # # # # # # # # # b_factor = 1.035
# NITER = 5000
# fname = f'./results/data/kms_connectivity/{name}_kms_connectivity_{b_factor}xbeta_c_{NITER}-iters.csv'
# # # ts1 = datetime.now()
# # # print(ts1)
# # # generate_kms_connectivity(G, b_factor, sources, targets, s_name=name, t_name='all', n_iter=NITER, fname=fname)
# # # ts2 = datetime.now()
# # # print(ts2)


# alpha = .05
# kms_conn_f = open(fname, 'r', newline='')

# # # # print(regular_polygon_coord(10, 4.))

# kms_con_rows = [line.strip().split(',') for line in kms_conn_f.readlines()[2:]]

# kms_weighted_connections = []

# for _, line in enumerate(kms_con_rows):
#     if (float(line[3]) <= alpha):
#         kms_weighted_connections += [(str(line[0]), str(line[1]), float(line[2])),] 



# # if len(exceptions) > 0:
# #     for l in [r for r in kms_con_rows if str(r[1]) in exceptions]:
# #         kms_weighted_connections += [(str(l[0]), str(l[1]), float(l[2])),]
# # # # w_sum = nonzero_sum([e[2] for e in kms_weighted_connections])   

# # # # kms_weighted_connections = [(e[0], e[1], e[2] / w_sum) for e in kms_weighted_connections]

# # ### Generate coordinates ######## 
# edges = G.edges

# connect = [e for _, e in enumerate(kms_weighted_connections) if e[0] == s]

# struc = [e[1] for _, e in enumerate(connect) if ((e[0], e[1]) in edges)]
# func = [e[1] for e in connect if ((e[0], e[1]) not in edges)]

# k1 = len(struc) + 1
# k2 = len(func) + 1
# r1 = 20.
# r2 = 30.  

# con_list = [(k1, struc, r1), (k2, func, r2)]

# node_coords = [(s, 0.0, 0.0)]

# for (k, con, radius) in con_list:
#     angle = radius * (np.pi / float(k)) 
#     con_coord = []
#     for i, node in enumerate(con, start=1):
#         x = radius * np.cos(i * angle)
#         y = radius  * np.sin(i * angle)
#         con_coord.append((node, x, y))
#     node_coords += con_coord


# pos = {str(co[0]): (float(co[1]), float(co[2])) for co in node_coords}
# if NOI in struc:
#     pos.update({NOI: (r1, 0.0)})


# K = nx.DiGraph()
# K.add_weighted_edges_from(connect)

# # # print([e for e in kms_weighted_connections if e[0] == 'AFDR'])
# node_shape = {n: 'polygon' for n in K.nodes}
# # for n in [v for v in list(K.nodes) if (str(v) != s)]:
# #     if  (s, n) in G.edges:
# #         node_shape[n] = 'polygon'
# #     else:
# #         node_shape[n] = 'polygon'
# #     node_shape.copy()
# node_shape[s] = 'round'


# draw_weighted_digraph(K, pos=pos, node_shape=node_shape, node_size=4.1, node_colors={n: node_colors[n] for n in K.nodes}, node_labels={n: node_labels[n] for n in K.nodes}, arrow_shrink=42, edgecolor=func_edge_color, font_size=20, figsize=(12,12))
# plt.show()
# plt.savefig(f'./results/newConnectome/nets/func/{s}_kms_circuit_{b_factor}xbeta_c_{NITER}-iters.svg', dpi=300, bbox_inches='tight')


#### Functional atlas ########
# name = 'all'
# b_factor = 1.05
# # # # # # # # # # # # # # # b_factor = 1.035
# NITER = 5000
# fname = f'./results/data/kms_connectivity/{name}_kms_connectivity_{b_factor}xbeta_c_{NITER}-iters.csv'
# # # ts1 = datetime.now()
# # # print(ts1)
# # # generate_kms_connectivity(G, b_factor, sources, targets, s_name=name, t_name='all', n_iter=NITER, fname=fname)
# # # ts2 = datetime.now()
# # # print(ts2)


# alpha = .05
# kms_conn_f = open(fname, 'r', newline='')

# # # # # # # # print(regular_polygon_coord(10, 4.))

# kms_con_rows = [line.strip().split(',') for line in kms_conn_f.readlines()[2:]]
# all_connects = [(str(l[0]), str(l[1]), float(l[2]), float(l[3])) for l in kms_con_rows]
# kms_weighted_connections = []

# for _, line in enumerate(kms_con_rows):
#     if (float(line[3]) <= alpha):
#         kms_weighted_connections += [(str(line[0]), str(line[1]), float(line[2])),] 




# # # proportions = {}
# # # for n in list(G.nodes):
# # #     n_outs = [e for e in all_connects if e[0] == n]
# # #     n_outs_sig = [e for e in all_connects if (e[0] == n) and (float(e[3] <= alpha))]
# # #     proportions[n] = len(n_outs_sig)
# # #     proportions.copy()

# # K = nx.DiGraph()
# # K.add_weighted_edges_from(struc_weighted_connections)

## unweighted graph 
# H = nx.DiGraph()
# H.add_edges_from([(w[0], w[1]) for w in kms_weighted_connections])

# print(number_of_paths(H, 'AFDR', 'AS08', 1))

# F = nx.DiGraph()
# # # normalize weights
# normalized_connections = []
# for neur in neurons:
#     w = nonzero_sum([e[2] for e in kms_weighted_connections if e[0] == neur])
#     normalized_connections += [(neur, e[1], e[2] / w) for e in kms_weighted_connections if e[0] == neur]




# F.add_weighted_edges_from(normalized_connections)

# v = 'ASHL'
# weights = [e[2] for e in normalized_connections if e[0] == v]
# sns.histplot(data = weights, bins=10)
# # # sns.histplot(data=list([K.out_degree(v) for v in K.nodes]), color='tab:blue', alpha=.4, label='anat')
# # # sns.histplot(data=list([F.out_degree(u) for u in F.nodes]), color='tab:red', alpha=.3, label='func', bins=10)
# plt.show()


# pos_dict = POSITIONS['all3d']
# draw_weighted_digraph(F, pos={n: (pos_dict[n][1], pos_dict[n][2]) for n in F.nodes}, node_size=.4, node_shape='polygon', node_colors=node_colors, node_labels={n:node_labels[n] for n in F.nodes}, font_size=6, edgecolor=func_edge_color, edge_factor=1.,e_alpha=.2, figsize=(12,10), font_kws = dict(fontweight='normal', fontstretch='normal'))

# font_kws = dict(fontweight='normal', fontstretch='normal')

# # ax = plt.gca()

# width = .6

# plt.text(-17.3, -10., 'Neurons', fontsize=12, zorder=10, **font_kws)
# polygonal_node(xy=(-17., -11.), width=width, facecolor=SE_color, ax=plt.gca())
# plt.text(-16.5, -11.3, 'Sensory', fontsize=12, zorder=10, **font_kws)

# polygonal_node(xy=(-17., -12.), width=width, facecolor=IN_color, ax=plt.gca())
# plt.text(-16.5, -12.3,'Inter', fontsize=12, zorder=10, **font_kws)

# polygonal_node(xy=(-17., -13), width=width, facecolor=MO_color, ax=plt.gca())
# plt.text(-16.5, -13.3,'Motor', fontsize=12, zorder=10, **font_kws)
# # plt.show()
# plt.savefig(f'./results/newConnectome/nets/func/topological_functional_connectome_{b_factor}xbeta_c_{NITER}-iters.pdf', dpi=300, bbox_inches='tight')
# p = 50
# most_senders = sorted(sorted([n for n in F.nodes], key=lambda x : F.out_degree(x), reverse=True)[:p])
# k = 85

# k = 65

# most_receivers = sorted(sorted([n for n in F.nodes], key=lambda x : F.in_degree(x), reverse=True)[:k])


# most_receivers = ['AIBL', 'AS01', 'AS02', 'AS03', 'AS04', 'AS05', 'AS06', 'AS07', 'AS08', 'AS09', 'AS10', 'AS11', 'AVAL', 'AVAR', 'AVBR', 'AVDL', 'AVDR', 'AVER', 'DA01', 'DA02', 'DA03', 'DA04', 'DA05', 'DA06', 'DA07', 'DA08', 'DA09', 'DB02', 'DB03', 'DB04', 'DB05', 'DB06', 'DB07', 'LUAL', 'LUAR', 'PHCR', 'PLMR', 'PVCL', 'PVCR', 'PVT', 'RIAL', 'RIAR', 'RIBL', 'RID', 'RIML', 'RMDDL', 'RMDDR', 'RMDL', 'RMDR', 'RMDVL', 'RMDVR', 'RMEV', 'SABD', 'SABVL', 'SABVR', 'SMDDL', 'SMDDR', 'SMDVL', 'VA01', 'VA02', 'VA03', 'VA04', 'VA05', 'VA06', 'VA07', 'VA08', 'VA09', 'VA10', 'VA11', 'VA12', 'VB04', 'VB06', 'VB07', 'VB08', 'VB09', 'VB10', 'VB11', 'VC06', 'VD08', 'VD13']

# senders = sorted(sorted([n for n in F.nodes if n != 'VD08'], key=lambda x : len([v for v in most_receivers if (x, v) in F.edges]), reverse=True)[:k])
# senders = sorted(sorted([n for n in F.nodes if n not in ['VD08']], key=lambda x : F.out_degree(x), reverse=True)[:k])

# senders = ['ADAR', 'ALA', 'ALMR', 'AS01', 'AS03', 'AS05', 'AS07', 'AS08', 'AS09', 'AS10', 'ASHR', 'AVAL', 'AVAR', 'AVBL', 'AVBR', 'AVDL', 'AVDR', 'AVEL', 'AVER', 'AVJL', 'AVJR', 'AVM', 'BDUL', 'BDUR', 'DA02', 'DA05', 'DA06', 'DA07', 'DA08', 'DB05', 'DB06', 'DB07', 'DVA', 'DVC', 'FLPL', 'FLPR', 'LUAL', 'LUAR', 'PHBL', 'PHBR', 'PLML', 'PLMR', 'PQR', 'PVCL', 'PVCR', 'PVDL', 'PVDR', 'PVNL', 'PVNR', 'PVPL', 'PVPR', 'PVR', 'PVWL', 'PVWR', 'RIS', 'RMHL', 'SAAVR', 'SABD', 'SABVL', 'SABVR', 'SDQL', 'SIADL', 'SIADR', 'SIAVR', 'SIBDL', 'SIBDR', 'SIBVL', 'SIBVR', 'VA01', 'VA05', 'VA08', 'VA10', 'VA12', 'VB07', 'VC04', 'VD05', 'VD06', 'VD08', 'VD10', 'VD11']

# most_connected = sorted(list(set([n for n in most_senders + most_receivers])))
# k = len(most_connected)
# forwards = CIRCUITS['ForwardLocomotion']
# backwards = CIRCUITS['BackwardLocomotion']
# coordination = CIRCUITS['LocomotionCoordination']

# matK = np.zeros((k,k))
# matF = np.zeros((k,k))
# for i, u in enumerate(senders):
#     for j, v in enumerate(most_receivers):
#         if (u == v) or ((u, v) not in K.edges):
#             matK[i][j] = 0.
#         else:
#             matK[i][j] = next(iter([float(w[2]) for w in struc_weighted_connections if (str(w[0]) == u) and (str(w[1]) == v)]))

# for i, u in enumerate(senders):
#     for j, v in enumerate(most_receivers):
#         if (u == v) or ((u, v) not in F.edges):
#             matF[i][j] = 0.
#         else:
#             matF[i][j] = next(iter([float(w[2]) for w in normalized_connections if (str(w[0]) == u) and (str(w[1]) == v)]))

# # # # print(list(F.edges(data=True))[:2])
# plt.rcParams.update({
#         "figure.autolayout": True,
#         "text.usetex": True,
#         "font.family": "Helvetica",
#         "figure.figsize": (12,10),
#         "font.size": 7
#     })
# # # fig, axes = plt.subplots(ncols=2)
# # # ax1, ax2 = axes
# plt.imshow(matF, cmap='turbo')
# # # # im2 = ax2.imshow(matF, cmap='hot')

# # # # plt.gca().grid(color='black', linestyle='-', linewidth=1)
# # # # for axe in [ax1,ax2]:
# plt.xticks(range(k), most_receivers, rotation=90)
# plt.yticks(range(k), senders)
# # ### labels colors
# xticks = plt.gca().get_xticklabels()
# yticks = plt.gca().get_yticklabels()

# for tick in xticks + yticks:
#     label = tick.get_text()
#     if label in forwards:
#         tick.set_color('blue')
#     elif label in backwards:
#         tick.set_color('red')
#     elif label in coordination:
#         tick.set_color('dimgray')
#     else:
#         tick.set_color('black')

# plt.xlabel(f'Target neurons', size=12)
# plt.ylabel(f'Source neurons', size=12)
# # # # plt.xlabel(f'Post-synaptic neurons', size=12)
# # # # plt.ylabel(f'Pre-synaptic neurons', size=12)
# cbar = plt.colorbar(shrink=.2) 
# # # # cbar.set_label("Weight", loc='center', size=12)
# # # cbar.ax.tick_params(labelsize=10)

# # plt.show()
# plt.savefig(f'./results/newConnectome/matrices/func_connect_mat_motor_{b_factor}xbc.svg', dpi=300, bbox_inches='tight')
# # print(len(K.edges))
# print(len(F.edges))
# print(sorted([(n, F.out_degree(n), F.in_degree(n)) for n in F.nodes], key=lambda x : x[2], reverse=True)[:50])

# print(sorted([(n, K.out_degree(n), K.in_degree(n)) for n in K.nodes], key=lambda x : x[1], reverse=True)[:30])
# print(K.in_degree('AS08'), K.in_degree('DA06'))
# n = 'PQR'
# print(F.out_degree(n), F.in_degree(n))

# print(print(sorted([(n, nx.clustering(K, n)) for n in F.nodes], key=lambda x : x[1], reverse=True)[:10]))
# v = 'LUAR'
# print(sorted([e for e in kms_weighted_connections if e[1] == v],key=lambda x : x[2], reverse=True)[:30])

# print(len(struc_weighted_connections))
# print(len(kms_weighted_connections))
# print(len(NEURONTYPES['motor']))
# print(most_receivers)
# print(senders)

### Save degree ranking in latex
# k = 60
# func_rank = {
#     'k-in': {a: i+1 for i, a in enumerate(sorted([n for n in F.nodes], key=lambda x : F.in_degree(x), reverse=True)[:k])},
#     'k-out': {a: i+1 for i, a in enumerate(sorted([n for n in F.nodes], key=lambda x : F.out_degree(x), reverse=True))},
#     'kw-in': {a: i+1 for i, a in enumerate(sorted([n for n in F.nodes], key=lambda x : F.in_degree(x, weight='weight'), reverse=True))},
#     'kw-out': {a: i+1 for i, a in enumerate(sorted([n for n in F.nodes], key=lambda x : F.out_degree(x, weight='weight'), reverse=True))}
# }

# struc_rank = {
#     'k-in': {a: i+1 for i, a in enumerate(sorted([n for n in K.nodes], key=lambda x : K.in_degree(x), reverse=True))},
#     'k-out': {a: i+1 for i, a in enumerate(sorted([n for n in K.nodes], key=lambda x : K.out_degree(x), reverse=True))},
#     'kw-in': {a: i+1 for i, a in enumerate(sorted([n for n in K.nodes], key=lambda x : K.in_degree(x, weight='weight'), reverse=True))},
#     'kw-out': {a: i+1 for i, a in enumerate(sorted([n for n in K.nodes], key=lambda x : K.out_degree(x, weight='weight'), reverse=True))}
# }

# latex_table = ''

# for n in func_rank['k-in']:
#     kin = func_rank['k-in'][n]
#     kout = func_rank['k-out'][n]
#     kwin = func_rank['kw-in'][n]
#     kwout = func_rank['kw-out'][n]

#     skin = struc_rank['k-in'][n]
#     skout = struc_rank['k-out'][n]
#     skwin = struc_rank['kw-in'][n]
#     skwout = struc_rank['kw-out'][n]

#     # latex_table += f'{n} & {kout} & {kwout} & {skout} & {skwout} \\\ \n'
#     latex_table += f'{n} & {kin} & {kwin} & {skin} & {skwin} \\\ \n'

# # print(K.out_degree('DA07'))
# print(latex_table)
     




######## DIVERGENCE/FIDELITY METRIC ########
# name = 'Thermotaxis'
# name = 'all'
# # # V = CIRCUITS[name]
# # # # V = NEURONTYPES['motor']
name = 'connectome'
V = sorted(list(G.nodes))
# # # # # # # # # beta_factor = 1.035
# # # # # # # # # b = beta_factor * bc
# # # # # # # # # print(1. / (b))
# ts = 1./ bs
# plot_sfd(G, V, bmin, 2.6*bc, num=100, node_colors=NEURONCOLORS, node_labels=node_labels, font_size=30, figsize=(10,6))

# # # # # plt.axvline(x=1./(2.5 * bc), linestyle='--', color='gray', label='')
# plt.axvline(ts, linestyle='--', color='gray', label='')
# plt.text(ts + .001, 60., r'$1/\beta_s$', fontsize=20, zorder=10)
# # plt.legend(bbox_to_anchor=(1.05, 1.0), fontsize='20', loc='upper left')
# # plt.show()
# plt.savefig(f'./results/newConnectome/divergence/{name}_sfd.pdf', dpi=300, bbox_inches='tight')
# plt.close()
# # print(1. / (1.5 * bc))
# beta_factor = 1.9
# b = beta_factor * bc
# # # # # #### SAVE divergence data 
# DIV = structure_function_divergence(G, V, b)

# DIV = sorted(DIV, key=lambda x : x[1], reverse=True)

# div_f = open(f'./results/data/divergence/{name}_sfd_{beta_factor}xbc.csv', 'w', newline='')
# writer = csv.writer(div_f, delimiter=',')
# writer.writerow([f'# Divergence between structural connectivity and KMS state of {name} at inverse temperature {beta_factor}xbeta_c={b}'])
# writer.writerow(['neuron','sfd'])

# for _, div in enumerate(DIV):
#     row = [str(div[0]),float(div[1])]
#     writer.writerow(row)

# print(deviations)


#### Scatter plot neurons by sfd values
# plt.rcParams.update({
#         "figure.autolayout": True,
#         "text.usetex": True,
#         "font.family": "Helvetica",
#         "figure.figsize": (16,8),
#         "font.size": 30
#     })
# ax = plt.gca()
# for i, div in enumerate(DIV):
#     plt.scatter(i, div[1], color='white', s=0, edgecolors='gray', marker='o', alpha=.5)
#     ax.annotate(node_labels[div[0]], (i + .005, div[1] + .005), fontsize=20)
# ax.get_xaxis().set_visible(False)
# plt.ylabel(r'{\bf sfd} (\%)')
# plt.savefig(f'./results/newConnectome/divergence/scplot{beta_factor}xbc.pdf', dpi=300, bbox_inches='tight')
# plt.show()
# print(1.01 * bc)


################### OTHER MEASURES : Volume  ################
# rG = configuration_model_from_directed_multigraph(G)

# xs = temperature_range(bmin, beta, num=50)
# Y1 = []
# Y2 = []
# for _, T in enumerate(xs):
#         b = 1./T 
#         Z1 = kms_emittance(G, b)[1]
#         Z2 = kms_emittance(rG, b)[1]
#         V1 = linalg.det(Z1)
#         V2 = linalg.det(Z2)
#         Y1 += [V1,]
#         Y2 += [V2,]
# plt.plot(xs, Y1, '-o', color='tab:red')
# plt.plot(xs, Y2, '-o', color='tab:blue')
# xlabel = r'Temperature $1/\beta$'
# plt.xlabel(xlabel)
# plt.ylabel('KMS Simplex volume')        

# plot_kms_simplex_volume(G, bmin, 1.2*bc, num=50, font_size=24)
# # # plt.legend(bbox_to_anchor=(1.05, 1.0), fontsize='12', loc='upper left')
# plt.show()
# print(1/(.19 * bc))
# plt.savefig(f'./results/newConnectome/simplex/kms_simplex_volume.pdf', dpi=300, bbox_inches='tight')


#### ENTROPY 

# plot_kms_states_js_divergence(G, [('RIAL', 'RIAR')], bmin, bs, num=50)
# plt.show()
######## STREAMS ####

# plt.rcParams.update({
#         "figure.autolayout": True,
#         "text.usetex": True,
#         "font.family": "Helvetica",
#         "figure.figsize": (16,10),
#         "font.size": 40
#     })
# name = 'AIY(AFD-)'
# # # # sources = ['AWCL', 'AWCR']
# # # linestyles = ['-', '--']
# # # # V = CIRCUITS[name]
# V = ['AIYL', 'AIYR']
# aiy_color = {'AIYL':'tab:blue', 'AIYR':'tab:red'}
# node_rm = ['AFDR', 'AFDL']
# plot_IC(G, V, bmin, bs, num=60, node_colors=aiy_color, node_labels=node_labels, nodes_removed=node_rm, font_size=40)
# # # # # plot_kms_simplex_volume(G, bmin, beta, num=50)
# # # # # plot_feedback_coef_variation(G, V, bmin, beta, colors=COLORS2, node_labels=node_labels, num=50, font_size=20)
# # # # plot_kms_receptance(G, V, bmin, beta, colors=COLORS2, node_labels=node_labels, num=50, font_size=20)
# # plt.axvline(x=1./(1.07 * bc), linestyle='--', color='gray', label='')
# plt.legend(bbox_to_anchor=(1.05, 1.0), fontsize='30', loc='upper left')
# plt.savefig(f'./results/newConnectome/streams/{name}_streams.pdf', dpi=300, bbox_inches='tight')

# plt.show()
# _, Z = kms_emittance(G, 1.62 * bc)
# det = linalg.det(Z)
# print(det)
#######################################################################
######## Croisements entropiques RIA vs AIB vs AIZ ####################
#######################################################################
### RIAL - AIBL : T = 0.20534 ---> b_factor = 1.13367
### RIAL - AIBR: T = 0.21107 ---> b_factor = 1.102894
### RIAL - AIZL : T = 0.214826 ---> b_factor = 1.083611115
### RIAR - AIBL : T = 0.19256 ---> b_factor = 1.208911
### RIAR - AIBR: T = 0.19813 ---> b_factor = 1.17492475
### RIAR - AIZR: T = 0.215804 ---> b_factor = 1.07870031
### RIAR - AIZL: T = 0.200392 ---> b_factor = 1.16166235

### 

# name = 'RIARvsAIZR'
# sources = ['RIAR', 'AIZR']
# # sources = CIRCUITS[name]
# # # sources = ['RMDVR']
# targets = list(G.nodes)
# # # # # # # # factors = [.01, .06, .07, .09, 1.1, 1.5, 1.9, 2.]
# b_factor = 1.07870031
# # # # # # # # b_factor = 1.035
# NITER = 5000
# fname = f'./results/data/kms_connectivity/{name}_kms_connectivity_{b_factor}xbeta_c_{NITER}-iters.csv'
# generate_kms_connectivity(G, b_factor, sources, targets, s_name=name, t_name='', n_iter=NITER, fname=fname)

# G = configuration_model_from_directed_multigraph(G)
# ######## ENTROPY ############
# name = 'AmphidSensila'
# V = CIRCUITS[name]
# # # name ='RID'
# # # V = ['RID']
# # b_factor = 1.07
# plot_nep_entropy(G, V, bmin, bs, num=10, base=2, with_feedback=False, node_colors=NEURONCOLORS, node_labels=node_labels)
# plt.legend(bbox_to_anchor=(1.05, 1.0), fontsize='12', loc='upper left')
# plt.show()

# # rank = NEP_entropy_ranking(G, b_factor * bc, V, with_feedback=False)

# # ### Save data 
# fname = f'./results/data/entropy/{name}_nep_entropy_{b_factor}xbc.csv'

# nep_f = open(fname, 'r', newline='')
# # writer = csv.writer(nep_f, delimiter=',')
# # writer.writerow([f'# Neural Emittance Profiles (with no feedback) of neurons in the {name} circuit '])
# # writer.writerow(['neuron','entropy'])
# # for nep in rank:
# #     writer.writerow([str(nep),float(rank[nep])])
# nep_rows = [line.strip().split(',') for line in nep_f.readlines()[2:]]

# vals = sorted([float(r[1]) for r in nep_rows], reverse=True)
# # ## Associate color to each neuron based on its NEP entropy
# import matplotlib.cm as cm
# from matplotlib.colors import Normalize

# cmap = cm.get_cmap('hot')


# cnorm = Normalize(vmin=min(vals), vmax=max(vals), clip=True)
# mapper = cm.ScalarMappable(norm=cnorm, cmap=cmap)

# thermo_color = {}
# for r in nep_rows:
#     thermo_color[r[0]] = mapper.to_rgba(float(r[1]))
#     thermo_color.copy()

# # cmap_list = []


# K = weighted_structural_subgraph(G, nodelist=V)
# draw_weighted_digraph(K, pos={n: POSITIONS['thermotaxis'][n] for n in K.nodes}, node_shape={n: 'polygon' for n in K.nodes}, node_size=3.4, node_colors=thermo_color, node_labels={n: node_labels[n] for n in K.nodes}, arrowstyle='fancy', arrow_shrink=40., edgecolor='silver', font_size=30, figsize=(12,10))
# plt.subplots_adjust(bottom=0.01, right=.85, top=1.)
# cax = plt.axes((.9, 0.1, 0.02, 0.5))
# cbar = plt.colorbar(mappable=mapper, cax=cax)
# cbar.ax.tick_params(labelsize=24)
# cbar.set_label(label=r'NEP entropy at optimal $\beta$', size=24)
# plt.show()
# plt.savefig(f'./results/newConnectome/entropy/{name}_nep_colormap.svg', transparent=True, dpi=300, bbox_inches='tight')
# np.random.seed(79546)
# # G = configuration_model_from_directed_multigraph(G)
# # print(thermo_color)
# plot_neural_emittance_profile_entropy(G, V, bmin, 2.5*bc, num=10, node_labels=node_labels, node_colors=NEURONCOLORS, with_feedback=False, font_size=60, figsize=(12,10))    
# # # # # plot_feedback_coef_variation(G, V, bmin, beta, colors=COLORS2, node_labels=node_labels, num=50, font_size=20)
# # # # plot_kms_receptance(G, V, bmin, beta, colors=COLORS2, node_labels=node_labels, num=50, font_size=20)
# # # plt.axvline(x=1./(1.07 * bc), linestyle='--', color='gray', label='')
# plt.legend(bbox_to_anchor=(1.05, 1.0), fontsize='20', loc='upper left')
# plt.show()
# plt.savefig(f'./results/newConnectome/entropy/{name}_entropy.pdf', dpi=300, bbox_inches='tight')

# colorlist = get_colors(len(list(G.nodes)), pastel_factor=.5)

# cols = {u: colorlist[i] for i, u in enumerate(list(G.nodes))}

# print(cols)

##### RECEPTANCE And FEEDBACK #######

# N = len(list(G.nodes))

# name = 'ExtendedThermotaxis'
# # V = CIRCUITS[name]
# # V = ['ASHL', 'ASHR', 'HSNL', 'HSNR', 'PHAL', 'PHAR', 'PHBL', 'PHBR', 'RIH']
# # V = sorted(list(G.nodes))
# # V = ['RID']
# # V = ['RMGL', 'RMGR', 'FLPL', 'FLPR', 'OLQDL', 'OLQDR', 'OLQVL', 'OLQVR', 'ASHL', 'ASHR', 'ASKL', 'ASKR', 'ASJL', 'ASJR']
# # V = ['SMBDL', 'SMBDR', 'SMBVL', 'SMBVR']
# tf = 1./ (1.07 * bc)
# tfmin = 1./ (1.085 * bc)
# tfmax = 1. / (1.02 * bc)
# recep_f = f'./results/newConnectome/receptance/integration_capacity_all.csv'

# IC_shape_cls = json.load(open('./results/newConnectome/receptance/neuron_IC_shape_classes.json'))

# cls_latex = 'Group 1 & Group 2 & Group 3 & Group 4 \\\\ \n'

# grp1 = IC_shape_cls['group-1']['neurons']
# grp2 = IC_shape_cls['group-2']['neurons']
# grp3 = IC_shape_cls['group-3']['neurons']
# grp4 = IC_shape_cls['group-4']['neurons']
# # for i, k in enumerate(IC_shape_cls.keys()):
# #     els = ''
# #     for n in IC_shape_cls[k]['neurons']:
# #         els += n + ', '
# #     cls_latex += f'{i + 1} & {els} & \\\\ \n'

# cls_latex += ', '.join(grp1) + ' & ' + ', '.join(grp2) + ' & ' + ', '.join(grp3) + ' & ' + ', '.join(grp4) 

# print(cls_latex)

# # # IC_corr_cls = json.load(open('./results/newConnectome/receptance/neuron_classification_by_IC__log-correlation-10-classes.json'))


# # # recep = node_total_kms_receptance_variation(G, sorted(list(G.nodes)), bmin, bs, num=200)
# # # df = pd.DataFrame.from_dict(recep)
# # # df.to_csv(recep_f, encoding='utf-8', index=False)
# df = pd.read_csv(recep_f, index_col=False)
# # df = pd.DataFrame(df, index=df['temperature'])
# xs = df['temperature']

# plt.rcParams.update({
#         "figure.autolayout": True,
#         "text.usetex": True,
#         "font.family": "Helvetica",
#         "figure.figsize": (12,10),
#         "font.size": 40
#     })

# y = df['AVEL']

# print(riemann_sum(xs, y))

# # a = []
# # for c in recept_classes:
# #     a += recept_classes[c]['neurons']

# # print(len(a))


##### Plot from IC_shape_classes
# for cl in IC_shape_cls:
#     nodelist = IC_shape_cls[cl]['neurons']
# # for gp_id in GROUP:
# #     name = f'group-{gp_id}'
# #     V = GROUP[gp_id]
# nodelist = IC_corr_cls['cl-9']
# cl = 'group-1'
# nodelist = IC_shape_cls[cl]['neurons']
# for n in nodelist:
#     plt.plot(xs, (df[n] * 100.) / (N - 1), color=NEURONCOLORS[n], label=n, linewidth=2.)
# # plt.axvline(x=tf, linestyle='--', color='gray', label='', alpha=0.5)
# # plt.axvline(x=tfmax, linestyle='--', color='gray', label='', alpha=0.5)
# # plt.axvspan(tf, tfmax, color='green', alpha=0.08)

# # plt.axvline(x=tf, linestyle='--', color='gray', label='')
# # plt.text(tf, -.06, r'$1/\beta_o$', fontsize=20, zorder=10)

# plt.xlabel(r'Temperature $1/\beta$')
# plt.ylabel(r'IC ($\%$)')
# # plt.legend(bbox_to_anchor=(1.05, 1.0), fontsize='20', loc='upper left')
# # plt.show()
# plt.savefig(f'./results/newConnectome/receptance/IC_shape_classes/{cl}_IC.pdf', dpi=300, bbox_inches='tight')
#     plt.close()
#     plt.close()
# print((1. / .2236) / bc)

# plt.savefig(f'./results/newConnectome/receptance/{name}_receptance.svg', dpi=300, bbox_inches='tight')

# V = ['RIAL','RIAR', 'DA06', 'DA08', 'DD03', 'AVAL', 'AVAR', 'PVCL', 'PVCR']
# plot_feedbac_coef(G, V, bmin, bs, num=30, node_colors=NEURONCOLORS, node_labels=node_labels, font_size=60, figsize=(12,10))
# plt.axvline(x=tf, linestyle='--', color='gray', label='')
# plt.axvline(x=tfmax, linestyle='--', color='gray', label='')
# plt.text(tf, -.004, r'$1/\beta_o$', fontsize=20, zorder=10)
# plt.legend(bbox_to_anchor=(1.05, 1.0), fontsize='10', loc='upper left')
# plt.show()
# # print((1. / .2236) / bc)

# # plt.savefig(f'./results/newConnectome/receptance/{name}_receptance.svg', dpi=300, bbox_inches='tight')
# plt.savefig(f'./results/newConnectome/coefficients/{name}_nic.pdf', dpi=300, bbox_inches='tight')
### Clustering wrt integration capacity ####
# index = df['temperature']
# V = IC_shape_cls['group-1']['neurons']
# V = sorted(V)
# for v in list(G.nodes):
#     df[v] = [np.log(df[v][i] + .001) for i, _ in enumerate(df[v])]
# # V = IC_shape_cls['group-1']['neurons']
# data = df[V].transpose()

# data = pd.DataFrame(data, index=index)

# sns.clustermap(data, metric='chebyshev', 
#             #    standard_scale=1, 
#             #    center=0,
#             row_cluster=False,
#             cmap='hot',  # Figure sizes
#             #   dendrogram_ratio = 0.1,
#             dendrogram_ratio=(.1, .1),
#                 #    cbar_pos=(.02, .32, .03, .2),
#                 # cbar_pos=(.02, .32, .03, .2),
#             cbar_pos=(0, .2, .006, .4),
#             figsize=(10, 6)
#             )
# metric = 'correlation'
# c_link = linkage(data,  metric=metric, method='complete')# computing the linkage

# B=dendrogram(c_link, labels=V)
# # print(B)

# num_clusters = 9
# cluster_labels = fcluster(c_link, num_clusters, criterion='maxclust')
# # print(cluster_labels)
# # cluster_labels = fclusterdata(data, t=4, metric='chebyshev', criterion='maxclust', method='complete')
# index = list(data.index)
# clusters = {f'cl-{i}':[] for i in range(1,num_clusters + 1)}
# for idx, label in enumerate(cluster_labels):
#     clusters[f'cl-{label}'].append(index[idx])
#     clusters.copy()

# clusters_json = f'./results/newConnectome/receptance/grou1_classification_by_IC__log-{metric}-{num_clusters}-classes.json'
# with open(clusters_json, "w") as outfile: 
#     json.dump(clusters, outfile)
# print(cluster_labels)
# print(list(data.index))
# print(data.tail(20))
# plt.show()







##### Receptance and IC ranking

# name = 'connectome'
# # # V = CIRCUITS[name]
# # V = list(G.nodes)
# b_factor = 1.02

# rank = IC_ranking(G, b_factor * bc, nodelist=sorted(list(G.nodes)))
# # rank = feedback_coef_ranking(G, b_factor*bc, nodelist=sorted(list(G.nodes)))
# # print(feed_rank)
# # # ### Save data 
# fname = f'./results/newConnectome/receptance/{name}_total_kms_receptance_rank_{b_factor}xbc.csv'
# # fname = f'./results/newConnectome/feedback/feedback_coefficients_all_1.05xbc.csv'

# # print(rank)

# rank_f = open(fname, 'w', newline='')
# writer = csv.writer(rank_f, delimiter=',')
# writer.writerow([f'# Ranking neurons by Integration Capacity at beta = {b_factor}xbc '])
# writer.writerow(['neuron','IC'])
# for _, c in enumerate(rank):
#     writer.writerow([str(c[0]),float(c[1])])

# riemanns = []

# for v in IC_shape_cls['group-2']['neurons']:
#     riemanns += [(v, riemann_sum(xs, df[v]))]
    

# riemanns = sorted(riemanns, key=lambda x : x[1], reverse=True)


# IC_rank_fname = f'./results/data/receptance/ranking_by_IC_riemann_sum.csv'
# IC_rank_f = open(IC_rank_fname, 'w', newline='')
# writer = csv.writer(IC_rank_f, delimiter=',')
# writer.writerow([f'# Ranking all neurons by the value of the area under the curve represented by the IC function'])
# for _, s in enumerate(riemanns):
#     writer.writerow([s[0],s[1]])


##### Demo #####
# from scipy.stats import skewnorm
# xs = temperature_range(bmin, 10.*bc, num=100)
# ys = []
# for _, T in enumerate(xs):
#     ys += [skewnorm.pdf(1. / T, 4.5),]

# plt.plot(xs, ys)
# plt.show()

# print(sorted(list(G.nodes), key=lambda x: G.out_degree(x), reverse=True))

############## 3D PLOTTING #####
##### Plotting neurons with color according to their IC sum
# from mpl_toolkits.mplot3d import Axes3D

# # neuronlist = IC_shape_cls['group-1']['neurons']
# neuronlist = list(G.nodes)
# # neuronlist = GANGLIA['anterior_ganglion']
# IC_ranks_o = open(IC_rank_fname, 'r', newline='')
# IC_ranks = [line.strip().split(',') for line in IC_ranks_o.readlines()[1:]]

# ICs = {}
# for r in IC_ranks:
#     ICs[r[0]] = float(r[1])
#     ICs.copy()


# XYZ = POSITIONS['3d']
# xs = []
# ys = []
# zs = []
# cs = []
# s = []

# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# cmap = plt.get_cmap("viridis")
# ax.set_axis_off()
# for _, n in enumerate(neuronlist):
#     x, y, z = XYZ[n]
#     c = ICs[n]
#     xs += [x,]
#     ys += [y,]
#     zs += [z,]
#     cs += [c,]
   


# ax.scatter(xs, ys, zs, s=50., c=cs, cmap=cmap)
# fig.tight_layout()
# # ax.set_aspect('equal')
# plt.xlim([min(xs)-1.5, max(xs)+1.5])
# plt.ylim([min(ys)-1.5, max(ys)+1.5])
# plt.ylim([min(zs)-1.5, max(zs)+1.5])
# ax.view_init(elev=40, azim=-45) 

# plt.show()


# print({n: XYZ[n][2] for n in CIRCUITS['VA']})



##### Emittance volume and path degree
# name = 'Thermotaxis'
# V = CIRCUITS[name]
# tf = 1./ (1.07 * bc)
# # plot_node_emittance(G, V, bmin, bs, num=10, node_labels=node_labels)
# plot_avg_total_receptance(G, bmin, bs, num=100, font_size=30, figsize=(10,6))
# plt.axvline(x=tf, linestyle='--', color='red', label='')
# plt.axhline(y=1/2., linestyle='--', color='gray')
# plt.text(tf + .001, -.004, r'$1/\beta_{\rm o}$', fontsize=20, zorder=10)
# # xs = range(20,100)
# # ys = [npath_avg(G, k) for k in xs]
# # plt.plot(xs, ys)
# plt.savefig(f'./results/newConnectome/receptance/mean_total_receptance.pdf', dpi=300, bbox_inches='tight')
# plt.show()

#### Kolmogorov-Smirnov
# from scipy.stats import ks_2samp 

# def ks(data1, data2):
#     sts = ks_2samp(data1,data2)
#     return {"stat": sts.statistic, "pvalue": sts.pvalue, "stat_location": sts.statistic_location}

# nodelist = ['AIYL', 'AIYR']

# edge_rmv = [e for e in G.edges if (e[0] in nodes_rmv) or (e[1] in nodes_rmv)]
# G.remove_edges_from(edge_rmv)
# IC = IC_variation(G, nodelist, bmin, bs, num=1000)
# left = IC['AIYL']
# right = IC['AIYR']
# sts = stats.ks_2samp(left, right)

# print(sts)
# afd = ['AFDL', 'AFDR']
# awc = ['AWCL', 'AWCR']
# afdl_rm = [e for e in G.edges if (e[0] != 'AFDL') and (e[1] != 'AFDL')]
# afdr_rm = [e for e in G.edges if (e[0] != 'AFDR') and (e[1] != 'AFDR')]
# afd_rm = [e for e in G.edges if (e[0] not in afd) and (e[1] not in afd)]
# awcl_rm = [e for e in G.edges if (e[0] != 'AWCL') and (e[1] != 'AWCL')]
# awcr_rm = [e for e in G.edges if (e[0] != 'AWCR') and (e[1] != 'AWCR')]
# awc_rm = [e for e in G.edges if (e[0] not in awc) and (e[1] not in awc)]

# AFDL = nx.MultiDiGraph()
# AFDL.add_nodes_from(neurons)
# AFDL.add_edges_from(afdl_rm)


# AFDR = nx.MultiDiGraph()
# AFDR.add_nodes_from(neurons)
# AFDR.add_edges_from(afdr_rm)


# AFD = nx.MultiDiGraph()
# AFD.add_nodes_from(neurons)
# AFD.add_edges_from(afd_rm)



# AWCL = nx.MultiDiGraph()
# AWCL.add_nodes_from(neurons)
# AWCL.add_edges_from(awcl_rm)


# AWCR = nx.MultiDiGraph()
# AWCR.add_nodes_from(neurons)
# AWCR.add_edges_from(awcr_rm)

# AWC = nx.MultiDiGraph()
# AWC.add_nodes_from(neurons)
# AWC.add_edges_from(awc_rm)

# wt_IC = IC_variation(G, nodelist, bmin, bs, num=1000)
# afdl_IC = IC_variation(AFDL, nodelist, bmin, bs, num=1000)
# afdr_IC = IC_variation(AFDR, nodelist, bmin, bs, num=1000)
# afd_IC = IC_variation(AFD, nodelist, bmin, bs, num=1000)


# awcl_IC = IC_variation(AWCL, nodelist, bmin, bs, num=1000)
# awcr_IC = IC_variation(AWCR, nodelist, bmin, bs, num=1000)
# awc_IC = IC_variation(AWC, nodelist, bmin, bs, num=1000)

# sts = {
#     "WT": {"L-R": ks(wt_IC['AIYL'], wt_IC['AIYR'])},
#     "AFDL-": {"L-L": ks(wt_IC['AIYL'],afdl_IC['AIYL']), "R-R": ks(wt_IC['AIYR'], afdl_IC['AIYR']), "L-R": ks(afdl_IC['AIYL'], afdl_IC['AIYR'])},
#     "AFDR-": {"L-L": ks(wt_IC['AIYL'],afdr_IC['AIYL']), "R-R": ks(wt_IC['AIYR'], afdr_IC['AIYR']), "L-R": ks(afdr_IC['AIYL'], afdr_IC['AIYR'])},
#     "AFD-": {"L-L": ks(wt_IC['AIYL'],afd_IC['AIYL']), "R-R": ks(wt_IC['AIYR'], afd_IC['AIYR']), "L-R": ks(afd_IC['AIYL'], afd_IC['AIYR'])},
#     "AWCL-": {"L-L": ks(wt_IC['AIYL'],awcl_IC['AIYL']), "R-R": ks(wt_IC['AIYR'], awcl_IC['AIYR']), "L-R": ks(awcl_IC['AIYL'], awcl_IC['AIYR'])},
#     "AWCR-": {"L-L": ks(wt_IC['AIYL'],awcr_IC['AIYL']), "R-R": ks(wt_IC['AIYR'], awcr_IC['AIYR']), "L-R": ks(awcr_IC['AIYL'], awcr_IC['AIYR'])},
#     "AWC-": {"L-L": ks(wt_IC['AIYL'],awc_IC['AIYL']), "R-R": ks(wt_IC['AIYR'], awc_IC['AIYR']), "L-R": ks(awc_IC['AIYL'], awc_IC['AIYR'])},
# }

# sts_json = f'./results/newConnectome/AIY_ic/AIY_IC_stats.json'

# with open(sts_json, "w") as outfile: 
#     json.dump(sts, outfile)




##### PHASE TRANSITIONS ####
# random.seed(64746987)
# plot_phase_transitions(G, [('AFDL', 'AFDR'),('ALML', 'ALMR'), ('ASEL', 'ASER'), ('AWCL', 'AWCR'), ('AVAL', 'AVAR'), ('AVBL', 'AVBR'), ('AVDL', 'AVDR'), ('AVEL', 'AVER'), ('PLML', 'PLMR'), ('AVM', 'PVM'), ('PVCL', 'PVCR'),  ('RIML', 'RIMR')], bmin, bs, num=100, font_size=30, figsize=(10,8))
# # plt.legend(bbox_to_anchor=(1.05, 1.0), fontsize='8', loc='upper left')
# legend = plt.legend(fontsize='12')
# # plt.savefig(f'./results/kms/phase-transition.pdf', dpi=300, bbox_inches='tight')

# plt.show()


###### AVERAGE KMS ENTROPY #####
# kin_dist = []
# kout_dist = []
# k_dist = []
# uniform_dist = []
# N = len(neurons)
# k_total = len(G.edges)
# for i, neur in enumerate(neurons):
#     kin = G.in_degree(neur)
#     kout = G.out_degree(neur)
#     k = kin + kout 
#     kin_dist += [ kin / float(k_total), ]
#     kout_dist += [ kout / float(k_total), ]
#     k_dist += [k / float(k_total), ]
#     uniform_dist += [1./ N, ]
# random.seed(64789)
# name = 'DD'
# V = CIRCUITS[name]
# plot_nep_entropy(G, V, bmin, bs, num=40, with_feedback=True,node_colors=NEURONCOLORS, node_labels=node_labels, font_size=30, ylabel=r'$\mathcal{S}(\mathbf{x}^{j|\beta})$', figsize=(10,6))
# name = 'circuit_avg_conprof'
# name = 'all_weighted_conprof'
# # plot_avg_nep_entropy(G, [CIRCUITS['ChemoReceptors'], CIRCUITS['CiliatedMRNs'], CIRCUITS['TouchReceptors'], CIRCUITS['CommandInterneurons'], CIRCUITS['LocomotionCoordination']], [None, None, None, None, None], bmin, bs, num=100, labels=['Chemo', 'Cilia-MRNs', 'Touch', 'Command', 'Coordination'], with_feedback=True, font_size=50, figsize=(10,10))
# G = configuration_model_from_directed_multigraph(G)
# plot_avg_nep_entropy(G, [neurons, neurons, neurons, neurons], [uniform_dist, kin_dist, kout_dist, k_dist], bmin, bs, num=100, labels=[r'$\mathbf{p}$ = Uniform dist',r'$\mathbf{p}$ = In-deg dist', r'$\mathbf{p}$ = Out-deg dist', r'$\mathbf{p}$ = deg dist'], with_feedback=True, font_size=50, ylabel=r'$\mathcal{S}(\mathbf{x}^{\bullet|\beta}, \mathbf{p})$', colors=['tab:orange', 'm', 'dodgerblue', 'tab:green'], figsize=(10,10))
# legend = plt.legend(fontsize='18')
# # # plt.legend(bbox_to_anchor=(1.05, 1.0), fontsize='12', loc='upper left')
# # # # # # legend.set_title(r'$\mathbf{p}$')
# plt.savefig(f'./results/kms/{name}_feedback.pdf', dpi=300, bbox_inches='tight')
# plt.show()
# print(3.5 * bc)


####### WEIGHTED AVERAGE KMS ENTROPY


############ KMS states networks
# bfactors = [3.15, 2.9, 2.5, 1.6, 1.05, 1.001]
# b_factor = 3.15
# name = 'TouchReceptors'
# V = CIRCUITS[name]
# k = len(V)
# P = [1./k] * k
# touch_recep = group_kms_emittance_connectivity(G, b_factor * bc, V, P, tol=1e-3)

# ## Form the emittance network
# s = 'Touch'
# C = nx.DiGraph()
# C.add_node(s)
# C.add_weighted_edges_from([(s, k, float(touch_recep[k])) for k in touch_recep])

# # # ### Generate coordinates ######## 
# edges = G.edges 
# V_targets = [w[1] for w in edges if w[0] in V]

# # connect = [e for _, e in enumerate(kms_weighted_connections) if e[0] == s]
# # connect_unwrap = []
# C_edges = list(C.edges)
# # for e in C_edges:
# #     for v in V:
# #         connect_unwrap += [(v, e[1]),]

# struc = [e[1] for e in C_edges if e[1] in V_targets]
# func = [e[1] for e in C_edges if e[1] not in V_targets]




# k1 = len(struc) + 3
# k2 = len(func) + 1
# r1 = 20.
# r2 = 30.  

# con_list = [(k1, struc, r1), (k2, func, r2)]

# node_coords = [(s, 0.0, 0.0)]

# for (k, con, radius) in con_list:
#     angle = radius * (np.pi / float(k)) 
#     con_coord = []
#     for i, node in enumerate(con, start=1):
#         x = radius * np.cos(i * angle)
#         y = radius  * np.sin(i * angle)
#         con_coord.append((node, x, y))
#     node_coords += con_coord


# pos = {str(co[0]): (float(co[1]), float(co[2])) for co in node_coords}
# # if NOI in struc:
# #     pos.update({NOI: (r1, 0.0)})



# # # # print([e for e in kms_weighted_connections if e[0] == 'AFDR'])
# node_shape = {n: 'round' for n in C.nodes}
# # for n in [v for v in list(K.nodes) if (str(v) != s)]:
# #     if  (s, n) in G.edges:
# #         node_shape[n] = 'polygon'
# #     else:
# #         node_shape[n] = 'polygon'
# #     node_shape.copy()
# node_shape[s] = 'polygon'

# ncolors = {n: node_colors[n] for n in C.nodes if n in neurons} | {s: '#e9beed'}
# nlabels = {s: f'Touch\nReceptors'}
# nsizes = {n: 1.6 for n in C.nodes if n in neurons} | {s: 22.4}

# draw_weighted_digraph(C, pos=pos, node_shape=node_shape, node_size=nsizes, node_colors=ncolors, node_labels=nlabels, arrow_shrinkA=110., arrow_shrinkB=12., edgecolor=func_edge_color, font_size=40, figsize=(10,10), font_kws = dict(ha='center', va='center', fontweight='normal', fontstretch='normal'))
# plt.show()
# # plt.savefig(f'./results/kms/{s}_emittance_{b_factor}xbc.pdf', dpi=300, bbox_inches='tight')

# print(len(struc), len(func), 3.15*bc)
# beta_factor = 1.02

# plot_node_kms_connectivity(G, 'AFDR', beta_factor*bc,node_labels=node_labels, node_colors=node_colors)
# plt.show()