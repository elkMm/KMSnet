from mixed_connectome import *



##### Structural connecvity adj list ########
# name = 'RMDVR'
# # sources = CIRCUITS[name]
# sources = ['RMDVR']
# adj_list = structural_connectivity_adj_list(G, sources)

# struc_conn_f = open(f'./results/data/struc_connectivity/{name}_structural_connectivity.csv', 'w', newline='')
# writer = csv.writer(struc_conn_f, delimiter=',')
# writer.writerow([f'# Structural connectivity of {name}'])
# writer.writerow(['source','target','weight'])

# for we in adj_list:
#     e = [str(we[0]),str(we[1]),float(we[2])]
#     writer.writerow(e)



################### Struct connectivity
# name = 'RMDVR'
# source = 'RMDVR'
# NOI = 'AFDR'
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

# draw_weighted_digraph(K, pos=pos, node_size=4., node_colors={n: node_colors[n] for n in K.nodes}, node_shape=node_shape , node_labels={n: node_labels[n] for n in K.nodes}, arrowstyle='fancy', arrow_shrink=42, edgecolor='silver', font_size=12, figsize=(8,8))
# plt.show()
# plt.savefig(f'./results/newConnectome/nets/struc/{source}_structural_connectivity.svg', dpi=300, bbox_inches='tight')


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

# sources = CIRCUITS[name]

# # # # # # # factors = [.01, .06, .07, .09, 1.1, 1.5, 1.9, 2.]
# b_factor = 2.5
# # # # # # # b_factor = 1.035
# NITER = 5000
# fname = f'./results/data/kms_connectivity/{name}_kms_connectivity_{b_factor}xbeta_c_{NITER}-iters.csv'
# generate_kms_connectivity(G, b_factor, sources, sources, s_name=name, t_name='', n_iter=NITER, fname=fname)


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

######### NODE KMS CONNECTIVITY ######

# # s = 'RMDVR'
# # NOI = 'AFDR'
# sources = CIRCUITS[name]
# # sources = ['RMDVR']
# # targets = list(G.nodes)
# # # # # # # factors = [.01, .06, .07, .09, 1.1, 1.5, 1.9, 2.]
# b_factor = 2.1
# # # # # # # b_factor = 1.035
# NITER = 5000
# fname = f'./results/data/kms_connectivity/{name}_kms_connectivity_{b_factor}xbeta_c_{NITER}-iters.csv'
# # # generate_kms_connectivity(G, b_factor, sources, targets, s_name=name, t_name='', n_iter=NITER, fname=fname)


# alpha = .05
# kms_conn_f = open(fname, 'r', newline='')

# # # print(regular_polygon_coord(10, 4.))

# kms_con_rows = [line.strip().split(',') for line in kms_conn_f.readlines()[2:]]

# kms_weighted_connections = []

# for _, line in enumerate(kms_con_rows):
#     if float(line[3]) <= alpha:
#         kms_weighted_connections += [(str(line[0]), str(line[1]), float(line[2])),] 

# # # # w_sum = nonzero_sum([e[2] for e in kms_weighted_connections])   

# # # # kms_weighted_connections = [(e[0], e[1], e[2] / w_sum) for e in kms_weighted_connections]

# # ### Generate coordinates ######## 
# # edges = G.edges

# # connect = [e for _, e in enumerate(kms_weighted_connections) if e[0] == s]

# struc = [e[1] for _, e in enumerate(connect) if ((e[0], e[1]) in edges)]
# func = [e[1] for e in connect if ((e[0], e[1]) not in edges)]

# k1 = len(struc) + 2
# k2 = len(func) + 3
# r1 = 16.
# r2 = 24.  

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


# draw_weighted_digraph(K, pos=pos, node_shape=node_shape, node_size=4., node_colors={n: node_colors[n] for n in K.nodes}, node_labels={n: node_labels[n] for n in K.nodes}, arrow_shrink=42, edgecolor=func_edge_color, font_size=12, figsize=(8,8))
# plt.show()
# plt.savefig(f'./results/newConnectome/nets/func/{s}_kms_circuit_{b_factor}xbeta_c_{NITER}-iters.svg', dpi=300, bbox_inches='tight')

######## DIVERGENCE/FIDELITY METRIC ########
# name = 'ExtendedThermotaxis'
# V = CIRCUITS[name]
# name = 'connectome'
# V = list(G.nodes)
# beta_factor = 1.035
# b = beta_factor * bc
# # print(1. / (b))
# # plot_structure_kms_state_divergence(G, V, bmin, beta, num=100, colors=COLORS2, node_labels=node_labels)
# # # plt.legend(bbox_to_anchor=(1.05, 1.0), fontsize='12', loc='upper left')
# # plt.show()
# # print(1. / (1.5 * bc))

# #### SAVE divergence data 
# DIV = structure_kms_state_divergence(G, V, b)

# div_f = open(f'./results/data/divergence/{name}_structure_function_divergence_{beta_factor}xbc.csv', 'w', newline='')
# writer = csv.writer(div_f, delimiter=',')
# writer.writerow([f'# Divergence between structural connectivity and KMS state of {name} at inverse temperature {beta_factor}xbeta_c={b}'])
# writer.writerow(['neuron','divergence'])

# for div in DIV:
#     row = [str(div[0]),float(div[1])]
#     writer.writerow(row)

# print(deviations)

####### OTHER MEASURES ################
