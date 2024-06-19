from mixed_connectome import *



##### Structural connecvity adj list ########
# name = 'RID'
# # sources = CIRCUITS[name]
# sources = ['RID']
# adj_list = structural_connectivity_adj_list(G, sources)

# struc_conn_f = open(f'./results/data/struc_connectivity/{name}_structural_connectivity.csv', 'w', newline='')
# writer = csv.writer(struc_conn_f, delimiter=',')
# writer.writerow([f'# Structural connectivity of {name}'])
# writer.writerow(['source','target','weight'])

# for we in adj_list:
#     e = [str(we[0]),str(we[1]),float(we[2])]
#     writer.writerow(e)



################### Struct connectivity
# name = 'AFD'
# source = 'AFDR'
# NOI = 'RMDVR'
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
# name = 'AFD'
# # sources = ['RMDVR']
# targets = list(G.nodes)
# sources = CIRCUITS[name]
# rmv_edges = [('AFDR', 'RMDVR')]

# # # # # # # # factors = [.01, .06, .07, .09, 1.1, 1.5, 1.9, 2.]
# b_factor = 1.05
# # # # # # # # b_factor = 1.035
# NITER = 5000
# # fname = f'./results/data/kms_connectivity/{name}_kms_connectivity_{b_factor}xbeta_c_{NITER}-iters.csv'
# # generate_kms_connectivity(G, b_factor, sources, targets, s_name=name, t_name='', n_iter=NITER, fname=fname)

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

######### NODE KMS CONNECTIVITY ######
# name = 'AFD'
# s = 'AFDR'
# NOI = 'RMDVR'
# # sources = CIRCUITS[name]
# # # sources = ['RID']
# # targets = list(G.nodes)
# # # # # # # # # factors = [.01, .06, .07, .09, 1.1, 1.5, 1.9, 2.]
# b_factor = 1.01
# # # # # # # # # b_factor = 1.035
# NITER = 5000
# fname = f'./results/data/kms_connectivity/{name}_kms_connectivity_{b_factor}xbeta_c_{NITER}-iters.csv'
# # generate_kms_connectivity(G, b_factor, sources, targets, s_name=name, t_name='', n_iter=NITER, fname=fname)


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
# edges = G.edges

# connect = [e for _, e in enumerate(kms_weighted_connections) if e[0] == s]

# struc = [e[1] for _, e in enumerate(connect) if ((e[0], e[1]) in edges)]
# func = [e[1] for e in connect if ((e[0], e[1]) not in edges)]

# k1 = len(struc) + 1
# k2 = len(func) + 1
# r1 = 16.
# r2 = 28.  

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


# draw_weighted_digraph(K, pos=pos, node_shape=node_shape, node_size=4.1, node_colors={n: node_colors[n] for n in K.nodes}, node_labels={n: node_labels[n] for n in K.nodes}, arrow_shrink=56, edgecolor=func_edge_color, font_size=20, figsize=(12,12))
# plt.show()
# plt.savefig(f'./results/newConnectome/nets/func/{s}_kms_circuit_{b_factor}xbeta_c_{NITER}-iters.svg', dpi=300, bbox_inches='tight')

######## DIVERGENCE/FIDELITY METRIC ########
# name = 'ExtendedThermotaxis'
# V = CIRCUITS[name]
# # # # # # name = 'connectome'
# # # # # # V = list(G.nodes)
# # # # # # beta_factor = 1.035
# # # # # # b = beta_factor * bc
# # # # # # print(1. / (b))
# ts = 1./(2.5 * bc)

# plot_sfd(G, V, 1.01 * bc, 2.8*bc, num=100, node_colors=THERMOCOLORS, node_labels=node_labels, font_size=60, figsize=(12,10))

# # plt.axvline(x=1./(2.5 * bc), linestyle='--', color='gray', label='')
# plt.axvline(ts, linestyle='--', color='gray', label='')
# plt.text(ts, 60., r'$1/\beta_s$', fontsize=60, zorder=10)
# plt.legend(bbox_to_anchor=(1.05, 1.0), fontsize='20', loc='upper left')
# plt.show()
# plt.savefig(f'./results/newConnectome/divergence/{name}_sfd.pdf', dpi=300, bbox_inches='tight')
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

# plot_kms_simplex_volume(G, bmin, beta, num=50, font_size=24)
# # plt.legend(bbox_to_anchor=(1.05, 1.0), fontsize='12', loc='upper left')
# # plt.show()
# plt.savefig(f'./results/newConnectome/simplex/kms_simplex_volume.pdf', dpi=300, bbox_inches='tight')



#### ENTROPY 

# plot_kms_states_js_divergence(G, [('AFDL', 'AFDR')], bmin, beta, num=50)
# plt.show()
######## STREAMS ####
# name = 'ExtendedThermotaxis'
# V = CIRCUITS[name]
# # plot_kms_simplex_volume(G, bmin, beta, num=50)
# # plot_feedback_coef_variation(G, V, bmin, beta, colors=COLORS2, node_labels=node_labels, num=50, font_size=20)
# plot_kms_receptance(G, V, bmin, beta, colors=COLORS2, node_labels=node_labels, num=50, font_size=20)
# plt.axvline(x=1./(1.07 * bc), linestyle='--', color='gray', label='')
# plt.legend(bbox_to_anchor=(1.05, 1.0), fontsize='12', loc='upper left')
# # plt.savefig(f'./results/newConnectome/coefficients/{name}_integration.pdf', dpi=300, bbox_inches='tight')

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



######## ENTROPY ############
# name = 'ExtendedThermotaxis'
# V = CIRCUITS[name]
# b_factor = 2.5

# rank = NEP_entropy_ranking(G, b_factor * bc, V, with_feedback=False)

# ### Save data 
# fname = f'./results/data/entropy/{name}_nep_entropy_NoFeedback_{b_factor}xbc.csv'

# nep_f = open(fname, 'w', newline='')
# writer = csv.writer(nep_f, delimiter=',')
# writer.writerow([f'# Neural Emittance Profiles (with no feedback) of neurons in the {name} circuit '])
# writer.writerow(['neuron','entropy'])
# for nep in rank:
#     writer.writerow([str(nep),float(rank[nep])])
# nep_rows = [line.strip().split(',') for line in nep_f.readlines()[2:]]

# vals = sorted([float(r[1]) for r in nep_rows], reverse=True)
# ## Associate color to each neuron based on its NEP entropy
# import matplotlib.cm as cm
# from matplotlib.colors import Normalize

# cmap = cm.get_cmap('hot').reversed()


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
# plot_neural_emittance_profile_entropy(G, V, bmin, 2.5*bc, num=10, node_labels=node_labels, node_colors=THERMOCOLORS, with_feedback=False, font_size=60, figsize=(12,10))    
# # # # # # plot_feedback_coef_variation(G, V, bmin, beta, colors=COLORS2, node_labels=node_labels, num=50, font_size=20)
# # # # # plot_kms_receptance(G, V, bmin, beta, colors=COLORS2, node_labels=node_labels, num=50, font_size=20)
# # # # plt.axvline(x=1./(1.07 * bc), linestyle='--', color='gray', label='')
# plt.legend(bbox_to_anchor=(1.05, 1.0), fontsize='20', loc='upper left')
# # plt.show()
# plt.savefig(f'./results/newConnectome/entropy/{name}_entropy.pdf', dpi=300, bbox_inches='tight')

# colorlist = get_colors(len(list(G.nodes)), pastel_factor=.5)

# cols = {u: colorlist[i] for i, u in enumerate(list(G.nodes))}

# print(cols)

##### RECEPTANCE And FEEDBACK #######
name = 'DD'
# V = CIRCUITS[name]
V = ['RMGL', 'RMGR', 'FLPL', 'FLPR', 'OLQDL', 'OLQDR', 'OLQVL', 'OLQVR', 'ASHL', 'ASHR']
# V = ['SMBDL', 'SMBDR', 'SMBVL', 'SMBVR']
tf = 1./bf
tfmax = 1. / (1.02 * bc)
plot_NIC(G, V, bmin, bs, num=60, node_colors=NEURONCOLORS, node_labels=node_labels, font_size=30, figsize=(12,10))
plt.axvline(x=tf, linestyle='--', color='gray', label='')
plt.axvline(x=tfmax, linestyle='--', color='gray', label='')
plt.text(tf, -.004, r'$1/\beta_o$', fontsize=20, zorder=10)
plt.legend(bbox_to_anchor=(1.05, 1.0), fontsize='10', loc='upper left')
plt.show()
# plt.savefig(f'./results/newConnectome/receptance/{name}_receptance.svg', dpi=300, bbox_inches='tight')
# plt.savefig(f'./results/newConnectome/coefficients/{name}_nic.pdf', dpi=300, bbox_inches='tight')

##### Receptance ranking

# name = 'connectome'
# # V = CIRCUITS[name]
# V = list(G.nodes)
# b_factor = 1.07

# rank = kms_receptance_ranking(G, b_factor * bc, averaging=False, with_feedback=False)

# # ### Save data 
# fname = f'./results/data/receptance/{name}_total_kms_receptance_rank(with_feedback)_{b_factor}xbc.csv'

# rank_f = open(fname, 'w', newline='')
# writer = csv.writer(rank_f, delimiter=',')
# writer.writerow([f'# Ranking neurons by mean KMS receptance at beta = {b_factor}xbc '])
# writer.writerow(['neuron','mean_kms_receptance'])
# for n in rank:
#     writer.writerow([str(n),float(rank[n])])

##### Demo #####
# from scipy.stats import skewnorm
# xs = temperature_range(bmin, 10.*bc, num=100)
# ys = []
# for _, T in enumerate(xs):
#     ys += [skewnorm.pdf(1. / T, 4.5),]

# plt.plot(xs, ys)
# plt.show()

# print(sorted(list(G.nodes), key=lambda x: G.out_degree(x), reverse=True))