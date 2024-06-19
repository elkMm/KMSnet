import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import linalg
from scipy.stats import entropy
from scipy.special import rel_entr
import seaborn as sns 
from collections import defaultdict
from distinctipy import get_colors
from .utils import is_qual, out_deg_ratio_matrix, fidelity, nonzero_sum, remove_ith, js_divergence
from .states import (
    emittance_matrix,
    kms_emittance,
    node_emittance_variation,
    node_kms_emittance_profile_variation,
    node_kms_receptance_profile_variation,
    node_kms_emittance_profile_entropy_range,
    node_kms_emittance_profile_diversity_range,
    KMS_emittance_dist_entropy_variation,
    node_structural_connectivity,
    node_structural_entropy, 
)
from .measures import *
from .kms_graphs import node_kms_emittance_connectivity
from .node_classes import node_kms_emittance_profile_by_classes


def plot_node_kms_stream_variation(graph, sources, targets, beta_min, beta_max, linestyle=None, num=100, colors=None, node_labels=None, font_size=12):

    plt.rcParams.update({
        "figure.autolayout": True,
        "text.usetex": True,
        "font.family": "Helvetica",
        "font.size": font_size,
        "text.latex.preamble": r"\usepackage{amsmath}"
    })
    nodes = list(graph.nodes)

    if linestyle == None:
        linestyle = ['-'] * len(sources)

    for i, source in enumerate(sources):
        streams = node_to_node_kms_flow_stream(graph, source, targets, beta_min, beta_max)

        if colors == None:
            colors = get_colors(len(nodes), pastel_factor=.5)


        labels = defaultdict(lambda: '')
        if node_labels != None:
            for n in node_labels:
                labels[n] = node_labels[n]

        xs = streams['range']
        for u in [n for n in targets if n != source]:
            ys = streams[u]
            l = labels[u]
            color = colors[nodes.index(u)]
            plt.plot(xs, ys, linestyle[i], label=f'{source}' + r'$\rightarrow$' + f'{l}', color=color)
    plt.xlabel(r'Temperature $1/\beta$')
    plt.ylabel('KMS connectivity weight')
    # plt.legend(bbox_to_anchor=(1.05, 1.0), fontsize='8', loc='upper left')


def plot_node_kms_streams_clustering(graph, source, targets, beta_min, beta_max, num=100, font_size=12):

    plt.rcParams.update({
        "figure.autolayout": True,
        "text.usetex": True,
        "font.family": "Helvetica",
        "font.size": font_size,
        "text.latex.preamble": r"\usepackage{amsmath}"
    })
    streams = node_to_node_kms_flow_stream(graph, source, targets, beta_min, beta_max, num=num)
    # targets = [n for n in targets if n != source]
    cols = targets
    drop_cols = ['range']

    if source in targets:
        cols = targets.remove(source)
        drop_cols = ['range', source]
   
    data = pd.DataFrame(streams, columns=cols, index=[round(x, 4) for _, x in enumerate(streams['range'])])
    
    data = data.drop(drop_cols, axis=1)
    cg = sns.clustermap(
        data, 
        metric='correlation', 
        row_cluster=False,
        cmap='hot',
        dendrogram_ratio=(.1, .2),
        cbar_pos=(0, .2, .03, .4),
        figsize=(10, 5),
        xticklabels=[f'{source}' + r'$\rightarrow$' + f'{u}' for _, u in enumerate(targets)],
        yticklabels=20
    )
    plt.setp(cg.ax_heatmap.get_yticklabels(), rotation=0)


def plot_kms_receptance_profile_entropy(graph,  nodelist, beta_min, beta_max, num=100, colors=None, node_labels=None, font_size=12):
    plt.rcParams.update({
        "figure.autolayout": True,
        "text.usetex": True,
        "font.family": "Helvetica",
        "font.size": font_size,
        "text.latex.preamble": r"\usepackage{amsmath}"
    })
    nodes = list(graph.nodes)
    KMSRecep = node_kms_receptance_profile_variation(graph,  nodelist, beta_min, beta_max, num=num)

    xs = KMSRecep['range']

    if colors == None:
        colors = get_colors(len(nodes), pastel_factor=.5)


    labels = defaultdict(lambda: '')
    if node_labels != None:
        for n in node_labels:
            labels[n] = node_labels[n]
    
    for _, u in enumerate(nodelist):
        i = nodes.index(u)
        profiles = KMSRecep[u]
        ys = [entropy(normalize(profile)) for _, profile in enumerate(profiles)]
        label = u
        color = colors[i]
        if labels[u] != '':
            label = labels[u]
        plt.plot(xs, ys, label=label, color=color)
    plt.xlabel(r'Temperature $1/\beta$')
    plt.ylabel('KMS receptance entropy')



def plot_total_receptance(graph, nodelist, beta_min, beta_max, num=50, node_colors=None, node_labels=None, nodes_removed=None, font_size=12, figsize=(12,10)):
    plt.rcParams.update({
        "figure.autolayout": True,
        "text.usetex": True,
        "font.family": "Helvetica",
        "figure.figsize": figsize,
        "font.size": font_size
    })
    nodes = list(graph.nodes)
    weights = node_total_kms_receptance_variation(graph, nodelist, beta_min, beta_max, num=num)

    colorlist = get_colors(len(nodes), pastel_factor=.5)

    color_dict = {u: colorlist[nodes.index(u)] for u in nodelist}
    if node_colors != None:
        color_dict = node_colors

    xs = weights['range']

    labels = defaultdict(lambda: '')
    if node_labels != None:
        for n in node_labels:
            labels[n] = node_labels[n]
    
    for _, u in enumerate(nodelist):
        ys = weights[u]
        color = color_dict[u]
        label = u
        if labels[u] != '':
            label = labels[u]
        plt.plot(xs, ys, color=color, label=label, linewidth=2.0)

    if nodes_removed != None:
        E = list(graph.edges)
        graph.remove_edges_from([e for e in E if (e[0] in nodes_removed) or (e[1] in nodes_removed)])
        weights2 = node_total_kms_receptance_variation(graph, nodelist, beta_min, beta_max, num=num)
        for _, v in enumerate(nodelist):
            if v not in nodes_removed:
                Ys = weights2[v]
                # uind = nodes.index(v)
                color = color_dict[v]
                label = v
                if labels[u] != '':
                    label = labels[u]
                plt.plot(xs, Ys, '-.', color=color, label=label, linewidth=2.0)
    plt.xlabel(r'Temperature $1/\beta$')
    plt.ylabel(r'Total $\beta$-receptance')



def plot_NIC(graph, nodelist, beta_min, beta_max, num=100, node_colors=None, node_labels=None, font_size=12, figsize=(12,10)):
    plt.rcParams.update({
        "figure.autolayout": True,
        "text.usetex": True,
        "font.family": "Helvetica",
        "figure.figsize": figsize,
        "font.size": font_size
    })

    nodes = list(graph.nodes)

    Coef = neural_integration_coefficient(graph, nodelist, beta_min, beta_max, num=num)

    colorlist = get_colors(len(nodes), pastel_factor=.5)

    color_dict = {u: colorlist[nodes.index(u)] for u in nodelist}
    if node_colors != None:
        color_dict = node_colors

    xs = Coef['range']

    labels = defaultdict(lambda: '')
    if node_labels != None:
        for n in node_labels:
            labels[n] = node_labels[n]
    
    for _, u in enumerate(nodelist):
        ys = Coef[u]
        # uind = nodes.index(u)
        color = color_dict[u]
        label = u
        if labels[u] != '':
            label = labels[u]
        plt.plot(xs, ys, color=color, label=label, linewidth=2.0)
    plt.xlabel(r'Temperature $1/\beta$')
    plt.ylabel('NIC')
    


def plot_node_emittance(graph, nodelist, beta_min, beta_max, num=100, node_labels=None, font_size=12):
    plt.rcParams.update({
        "figure.autolayout": True,
        "text.usetex": True,
        "font.family": "Helvetica",
        "font.size": font_size
    })

    emittance = node_emittance_variation(graph, nodelist, beta_min, beta_max, num=num)

    x = emittance['range']

    labels = defaultdict(lambda: '')
    if node_labels != None:
        for n in node_labels:
            labels[n] = node_labels[n]
    
    for u in nodelist:
        y = emittance[u]
        label = u
        if labels[u] != '':
            label = labels[u]
        plt.plot(x, y, label=label)
    plt.xlabel('Inverse temperature')
    plt.ylabel('Node emittance')
    plt.legend()



def plot_sfd(graph, nodelist, beta_min, beta_max, num=50, node_colors=None, node_labels=None, xticks=None, font_size=12, figsize=(10,6)):
    '''Plot the variation of the divergence between 
    structural connectivity and KMS states of nodes.'''

    plt.rcParams.update({
        "figure.autolayout": True,
        "text.usetex": True,
        "font.family": "Helvetica",
        "font.size": font_size,
        "figure.figsize": figsize,
    })

    ax = plt.gca()

    nodes, SS = node_structural_connectivity(graph)
    xs = list(np.linspace(1./beta_max, 1./beta_min, num=num))
    

    colorlist = get_colors(len(nodes), pastel_factor=.5)

    color_dict = {u: colorlist[nodes.index(u)] for u in nodelist}
    if node_colors != None:
        color_dict = node_colors

    labels = defaultdict(lambda: '')
    if node_labels != None:
        for n in node_labels:
            labels[n] = node_labels[n]

    YS = {u: [] for u in nodelist}

    for _, T in enumerate(xs):
        beta = 1./T
        verts, Z = kms_emittance(graph, beta)
        for v in nodelist:
            v_ind = nodes.index(v)
            k = verts.index(v)
            struc = remove_ith(SS[v], v_ind)
            prof = remove_ith(Z[:, v_ind], k)
            fid = fidelity(struc, prof)
            div = (1. - fid) * 100.
            YS[v] += [div,]  
            YS.copy()  

    for _, node in enumerate(nodelist):
        label = node
        # node_ind = nodes.index(node)
        color = color_dict[node]
        ys = YS[node]
        if labels[node] != '':
            label = labels[node]

        plt.plot(xs, ys, label=label, color=color, linewidth=2.0)

    # x_ticks = [round(t, 2) for t in ax.get_xticks(minor=False)]

    # if xticks != None:
    #     x_ticks += [round(t, 2) for t in xticks] 

    # # Set xtick locations to the values of the array `x_ticks`
    # ax.set_xticks(x_ticks)

    xlabel = r'Temperature $1/\beta$'    
    plt.xlabel(xlabel)
    plt.ylabel(r'sfd (\%)')

    

def plot_node_kms_connectivity(graph, node, beta, tol=TOL, node_labels=None, node_colors=None):
    '''Plot the KMS connectivity of the node at inverse temperature beta.'''

    con = node_kms_emittance_connectivity(graph, node, beta, tol=tol)
    ## Node KMS weighted connectity
    # emitter = 'RID'

    ax = plt.gca()
    ys = [k for k in con.keys()]

    labels = defaultdict(lambda: '')
    if node_labels != None:
        for n in node_labels:
            labels[n] = node_labels[n]

    colors = defaultdict(lambda: 'tab:blue')
    if node_colors != None:
        for n in node_colors:
            colors[n] = node_colors[n]

    # Select only the neurons with non-zero receptance
    for nv in ys:
        plt.scatter(nv, con[nv], color=colors[nv])
        ax.annotate(labels[nv], (nv, con[nv]), fontsize=7)

    ax.get_xaxis().set_visible(False)

    plt.xlabel(f'Nodes')
    plt.ylabel(f'{node} neural emittance')
    # plt.show()


def plot_neural_emittance_profile_entropy(graph, nodelist, beta_min, beta_max, num=50, with_feedback=True , node_colors=None, node_labels=None, font_size=12, figsize=(10,6)):
    '''Plot the variation of the entropy of each node KMS emittance in nodelist.'''

    plt.rcParams.update({
        "figure.autolayout": True,
        "text.usetex": True,
        "font.family": "Helvetica",
        "font.size": font_size,
        "figure.figsize": figsize,
    })

    KMS = node_kms_emittance_profile_variation(graph, nodelist, beta_min, beta_max, num=num, with_feedback=with_feedback)
    # vertices, SS = node_structural_connectivity(graph, nodelist=nodelist)

    # H = node_kms_emittance_profile_entropy_range(graph, nodelist, beta_min, beta_max, num=num)

    # struc_entropy = node_structural_entropy(graph, nodelist=nodelist)

    xs = KMS['range']
    nodes = list(graph.nodes)
    
    colorlist = get_colors(len(nodes), pastel_factor=.5)

    color_dict = {u: colorlist[nodes.index(u)] for u in nodelist}
    if node_colors != None:
        color_dict = node_colors
        

    # colors = get_colors(len(nodelist), pastel_factor=.5)

    
    labels = defaultdict(lambda: '')
    if node_labels != None:
        for n in node_labels:
            labels[n] = node_labels[n]
    
    for _, u in enumerate(nodelist):
        profiles = KMS[u]
        ys = [entropy(profile) for _, profile in enumerate(profiles)]
        # u_struc = SS[u]
        label = u
        color = color_dict[u]
        if labels[u] != '':
            label = labels[u]
        plt.plot(xs, ys, 'o-', label=label, color=color, linewidth=2.5)

        # find and plot the structural entropy of node u
        # for step, _ in enumerate(xs):
        #     ind = vertices.index(u)
        #     u_prof = remove_ith(profiles[step], ind)
        #     u_struc = remove_ith(u_struc, ind)
        #     fid = fidelity(u_struc, u_prof)
            
        #     if  is_qual(fid, 1., tol=1e-1):
        #         plt.scatter(xs[step], entropy(u_prof), color=color)

    xlabel = r'Temperature $1/\beta$' 
    plt.xlabel(xlabel)
    plt.ylabel('NEP entropy')
    # plt.legend()

def plot_node_kms_emittance_profile_diversity(graph, nodelist, beta_min, beta_max, num=50, node_labels=None, font_size=12):

    plt.rcParams.update({
        "figure.autolayout": True,
        "text.usetex": True,
        "font.family": "Helvetica",
        "font.size": font_size
    })

    diversity = node_kms_emittance_profile_diversity_range(graph, nodelist, beta_min, beta_max, num=num)


    x = diversity['range']
    
    labels = defaultdict(lambda: '')
    if node_labels != None:
        for n in node_labels:
            labels[n] = node_labels[n]
    
    for u in nodelist:
        y = diversity[u]
        label = u
        if labels[u] != '':
            label = labels[u]
        plt.plot(x, y, label=label)
    xlabel = r'Temperature $1/\beta$'
    plt.xlabel(xlabel)
    plt.ylabel('Node profile diversity')
    plt.legend()


def plot_kms_states_js_divergence(graph, node_pairs, beta_min, beta_max, num=100, linestyle=None, colors=None, node_labels=None, font_size=12):
    '''Plot the variation of Jensen-Shannon divergence betwen the KMS states of each pair of nodes in node_pairs.

    Parameters
    ----------
    node_pairs : list of 2-tuples
    '''

    plt.rcParams.update({
        "figure.autolayout": True,
        "text.usetex": True,
        "font.family": "Helvetica",
        "font.size": font_size
    })
    nodes = list(graph.nodes)
    if linestyle == None:
        linestyle = ['-'] * len(node_pairs)

    nodelist = []
    for pair in node_pairs:
        nodelist += [pair[0], pair[1]]
    
    nodelist = list(set(nodelist))

    KMS = node_kms_emittance_profile_variation(graph, nodelist, beta_min, beta_max, num=num)

    xs = KMS['range']
    if colors == None:
        colors = get_colors(len(node_pairs), pastel_factor=.5)
    
    labels = defaultdict(lambda: '')
    if node_labels != None:
        for n in node_labels:
            labels[n] = node_labels[n]
    
    for i, (left, right) in enumerate(node_pairs):
        profile_pairs = zip(KMS[left], KMS[right])
        li = nodes.index(left)
        ri = nodes.index(right)
        # ys = [js_divergence(p[0], p[1]) for _, p in enumerate(profile_pairs)]
        ys = [js_divergence(remove_ith(p[0], li), remove_ith(p[1], ri)) for _, p in enumerate(profile_pairs)]
        color = colors[i]
        plt.plot(xs, ys, linestyle[i], label=f'({labels[left]}, {labels[right]})', color=color)
        
    plt.xlabel(r'Temperature $1/\beta$')
    plt.ylabel('KMS states divergence')


def plot_kms_states_by_classes_divergence(graph, nodes_by_classes, node_pairs, beta_min, beta_max, num=100, linestyle=None, colors=None, node_labels=None, font_size=12):
    '''Plot the variation of Jensen-Shannon divergence betwen the KMS states of each pair of nodes in node_pairs.

    Parameters
    ----------
    nodes_by_classes : dict
    node_pairs : list of 2-tuples
    '''

    plt.rcParams.update({
        "figure.autolayout": True,
        "text.usetex": True,
        "font.family": "Helvetica",
        "font.size": font_size
    })
    # nodes = list(graph.nodes)
    if linestyle == None:
        linestyle = ['-'] * len(node_pairs)

    nodelist = []
    for pair in node_pairs:
        nodelist += [pair[0], pair[1]]
    
    nodelist = list(set(nodelist))

    KMS = node_kms_emittance_profile_by_classes(graph, nodes_by_classes, nodelist, beta_min, beta_max, num=num)

    xs = KMS['range']
    node_cls = KMS['node_cls']

    if colors == None:
        colors = get_colors(len(node_pairs), pastel_factor=.5)
    
    labels = defaultdict(lambda: '')
    if node_labels != None:
        for n in node_labels:
            labels[n] = node_labels[n]
    
    for i, (left, right) in enumerate(node_pairs):
        profile_pairs = zip(KMS[left], KMS[right])
        # left_cl = [c for c in node_cls if left in nodes_by_classes[c]][0]
        # right_cl = [cl for cl in node_cls if right in nodes_by_classes[cl]][0]
        # li = node_cls.index(left_cl)
        # ri = node_cls.index(right_cl)
        ys = [js_divergence(p[0], p[1]) for _, p in enumerate(profile_pairs)]
        # ys = [js_divergence(remove_ith(p[0], li), remove_ith(p[1], ri)) for _, p in enumerate(profile_pairs)]
        color = colors[i]
        plt.plot(xs, ys, linestyle[i], label=f'({labels[left]}, {labels[right]})', color=color)
        
    plt.xlabel(r'Temperature $1/\beta$')
    plt.ylabel('KMS connectivity symmetry')



def plot_kms_simplex_volume(graph, beta_min, beta_max, num=100, font_size=12):
    '''
    Plot the volume of the simplex of KMS states as a function
    of the temperature 1/beta in the interval [1/beta_max, 1/beta_min].

    Notes:
    -----
    It also plots the horizontal line whose y-coordinate is the volume
    of the simplex defined by the structural states of the nodes.
    '''
    plt.rcParams.update({
        "figure.autolayout": True,
        "text.usetex": True,
        "font.family": "Helvetica",
        "font.size": font_size
    })
    # nodes = list(graph.nodes)

    if beta_min == beta_max:
        interval = [1./beta_min]
    else:
        interval = list(np.linspace(1./beta_max, 1./beta_min, num=num))

    # R = out_deg_ratio_matrix(graph, nodes=nodes)
    # V_R = linalg.det(R)

    xs = interval
    ys = []

    for _, T in enumerate(interval):
        beta = 1./T 
        Z = kms_emittance(graph, beta)[1]
        V_Z = linalg.det(Z)
        ys += [V_Z,]

    plt.plot(xs, ys, '-o', color='tab:red')
    # plt.plot(xs, [V_R] * len(xs), '--', color='tab:gray')
    xlabel = r'Temperature $1/\beta$'
    plt.xlabel(xlabel)
    plt.ylabel('KMS Simplex volume')