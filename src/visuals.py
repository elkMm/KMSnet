import itertools
from collections import defaultdict
from matplotlib import cm, colors
import matplotlib.pyplot as plt 
import numpy as np
from matplotlib.patches import Ellipse, Circle, FancyArrowPatch, PathPatch, RegularPolygon, ArrowStyle
import mpl_toolkits.mplot3d.art3d as art3d
from mpl_toolkits.mplot3d import proj3d

import networkx as nx


def draw_multi_digraph(G, pos=None, layout=nx.spring_layout, node_shape='round', node_size=.4, node_colors=None, node_labels=None, edgecolor='tab:gray', font_size=12, figsize=(10,10), ax=None):
    '''
    Draw a directed multigraph.

    Parameters
    ----------
    G : a networkx MultiDiGraph object
    node_shape: str
        `circle` or `polygon`
    '''

    plt.rcParams.update({
        "figure.autolayout": True,
        "figure.figsize": figsize,
        "text.usetex": True,
        "font.family": "Helvetica",
        "font.size": font_size,
        "font.weight": "bold"
    })

    # pos = nx.kamada_kawai_layout(G)
    if pos is None:
        pos = layout(G)

    # fig = plt.figure(figsize=figsize)
    fig, ax = plt.subplots()

    
    if ax is None:
        ax = plt.gca()
    # ax = plt.gca()
    ax.set_axis_off()
    xs = []
    ys = []

    color_map = defaultdict(lambda: 'white')
    if node_colors != None:
        for k in node_colors:
            color_map[k] = node_colors[k]

    labels = defaultdict(lambda: '')
    if node_labels != None:
        for n in node_labels:
            labels[n] = node_labels[n]
    

    draw_node = round_node
    width = node_size
    if node_shape == 'polygon':
        draw_node = polygonal_node
        width = node_size

    for node in pos:
        x, y = pos[node]
        draw_node((x,y), facecolor=color_map[node], width=width, ax=ax)
        ax.annotate(labels[node], (x + .2, y + .1), fontsize=font_size, zorder=5)
        xs.append(x)
        ys.append(y)
   
    edges = G.edges

    for edge in edges:
        draw_curved_edge(edge, color=edgecolor, pos=pos, ax=ax)

    # if node_labels != None:
    #     # Draw node labels
    #     nx.draw_networkx_labels(
    #         G, 
    #         pos=pos, 
    #         labels=node_labels, 
    #         font_size=font_size, 
    #         font_color='k', 
    #         # font_family='sans-serif', 
    #         font_weight='normal', 
    #         horizontalalignment='left', 
    #         verticalalignment='bottom', 
    #         ax=ax
    #     )


    fig.tight_layout()
    ax.set_aspect('equal')
    plt.xlim([min(xs)-3., max(xs)+3.])
    plt.ylim([min(ys)-3., max(ys)+3.])
    # plt.show()



def draw_weighted_digraph(G, pos=None, layout=nx.spring_layout, node_shape=None, node_size=.4, node_colors=None, node_labels=None, edgecolor='tab:gray', arrowstyle='simple', arrow_shrink=None, font_size=12, figsize=(10,10), ax=None):
    '''
    Draw a directed weighted digraph.

    Parameters
    ----------
    G : a networkx DiGraph object
    node_shape: dict, optional, default: None
        Dictionary of nodes as keys and str as values representing the corresponding node shape: only `round` or `polygon` are accepted FOR NOW!
    '''

    plt.rcParams.update({
        "figure.autolayout": True,
        "figure.figsize": figsize,
        # "text.usetex": True,
        # "font.family": "Helvetica",
        # "font.size": font_size,
        # "font.weight": "bold"
    })

    # pos = nx.kamada_kawai_layout(G)
    if pos is None:
        pos = layout(G)

    # fig = plt.figure(figsize=figsize)
    fig, ax = plt.subplots()

    
    # if ax is None:
    #     ax = plt.gca()
    # ax = plt.gca()
    ax.set_axis_off()
    xs = []
    ys = []

    color_map = defaultdict(lambda: 'white')
    if node_colors != None:
        for k in node_colors:
            color_map[k] = node_colors[k]
    
    labels = defaultdict(lambda: '')
    if node_labels != None:
        for n in node_labels:
            labels[n] = node_labels[n]

    shapes = defaultdict(lambda: round_node)
    if node_shape != None:
        for n in node_shape:
            if node_shape[n] == 'round':
                shapes[n] = round_node
            if node_shape[n] == 'polygon':
                shapes[n] = polygonal_node

            # shapes[n] = node_shape[n]

    # draw_node = round_node
    width = node_size
    # if node_shape == 'polygon':
    #     draw_node = polygonal_node
    #     # width = node_size
    node_kws = dict(fontweight='heavy', fontstretch='normal')
    for node in pos:
        x, y = pos[node]
        shapes[node]((x,y), facecolor=color_map[node], width=width, ax=ax)
        ax.annotate(labels[node], xy=(x, y), fontsize=font_size, zorder=10, ha='center', va='center', **node_kws)
        xs.append(x)
        ys.append(y)
   
    edges = list(G.edges(data=True))
    
    shrink_factor = node_size * 10.
    if arrow_shrink != None:
        shrink_factor = arrow_shrink

    for w_edge in edges:
        edge = (w_edge[0], w_edge[1], 0)
        weight = 24 * w_edge[2]['weight']
        draw_curved_edge(edge, arrowstyle=arrowstyle, width=weight, alpha=1., color=edgecolor, shrink_factor=shrink_factor, pos=pos, ax=ax)

    # if node_labels != None:
    #     # Draw node labels
    #     nx.draw_networkx_labels(
    #         G, 
    #         pos=pos, 
    #         labels=node_labels, 
    #         font_size=font_size, 
    #         font_color='k', 
    #         # font_family='sans-serif', 
    #         font_weight='normal', 
    #         horizontalalignment='left', 
    #         verticalalignment='bottom', 
    #         ax=ax
    #     )


    fig.tight_layout()
    ax.set_aspect('equal')
    plt.xlim([min(xs)-2., max(xs)+2.])
    plt.ylim([min(ys)-2., max(ys)+2.])




    
def draw_flow_mode(G, pos, with_labels=False, node_labels=None, node_size=30, width=.1, alpha=.5, node_color='white', edge_color='tab:blue', edgecolors='black', font_size=20, ax=None):
    '''Visualizes a directed graph with the positions 
    given by the inflow or outflow groupoid.'''
    if ax is None:
        ax = plt.gca()

    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "Times New Roman",
        "font.size": font_size
    })
    
    V = G.nodes
    E = G.edges

    # color_dict = defaultdict(lambda: 'white')
    # if node_colors != None:
    #     for k in node_colors:
    #         color_dict[k] = node_colors[k]

    
    sizes = {}
    coords = [c for c in pos.values()]
    
    for c in coords:
        c_nodes = [n for n in V if (pos[n] == c)]
        for node in c_nodes:
            sizes[node] = len(c_nodes)
        sizes.copy()
    
    # for n in V:
    #     draw_node(pos[n], color=color_dict[n], width=width, alpha=alpha, edgecolor=edgecolor, ax=ax)
    # for node in V:

    edgelist = []

    for n0, n1 in E:
        if tuple(pos[n0]) != tuple(pos[n1]):
            edgelist.append((n0, n1))
    

    nx.draw_networkx(G, pos=pos, edgelist=edgelist, with_labels=with_labels, label=node_labels, node_size=node_size, width=width, alpha=alpha, node_color=node_color, edge_color=edge_color, edgecolors=edgecolors, ax=ax)


def polygonal_node(xy, width=1., edgecolor='dimgray', facecolor='tab:blue', zorder=8, alpha=.4, ax=None):
    '''Node in the form of polygon'''

    kwargs = dict(edgecolor=edgecolor, facecolor=facecolor, zorder=zorder, alpha=alpha)
    patch = RegularPolygon(xy, numVertices=5, radius=width/2, **kwargs)
    ax.add_patch(patch)



def round_node(xy, facecolor='yellow', width=0.06, edgecolor='dimgray', zorder=8, alpha=.4, ax=None):
    '''Draw a node given a dictionary of node positions.'''
    # if ax is None:
    #     ax = plt.gca()

    ell = _node(xy, color=facecolor, width=width, edgecolor=edgecolor, zorder=zorder, alpha=alpha)
    ax.add_patch(ell)


def draw_curved_edge(edge, color='tab:gray',arrowstyle='simple', width=.2, alpha=0.4, zorder=1, shrink_factor=2., pos=None, ax=None):
    '''Draws curved edge given the source and range positions of the edge'''

    arc = _curved_edge(edge=edge, color=color, arrowstyle=arrowstyle, width=width, alpha=alpha, zorder=zorder, shrink_factor=shrink_factor, pos=pos)
    ax.add_patch(arc)


def _node(xy, color='tab:red', width=0.02, edgecolor='dimgray', zorder=5, alpha=0.6):
    '''Plots an ellipse to represent a node given a dictionary of node positions.'''

    ell = Circle(xy, radius=width/2., zorder=zorder, lw=0.5, edgecolor=edgecolor, facecolor=color, alpha=alpha)
    return ell


def _curved_edge(edge, color='blue', arrowstyle='simple', width=0.3, shrink_factor=2., alpha=.6, zorder=1, pos=None):
    '''Returns an object to represent an arc, used mainly for interlayer connections.'''

    n0, n1, route = edge
    x0,y0 = pos[n0]
    x1,y1 = pos[n1]
    head_width = 3 * (1. + width)
    head_length = 2 * (1. + width)
    # shrink = shrink_factor * 10.
    simple = f'Simple, head_width={head_width}, head_length={head_length}'
    wedge = f'Wedge, tail_width={1. + 1.2*width}, shrink_factor=0.1'
    anat = f'->, head_width={.5*width}, head_length={.1*width}'
    bracket = ArrowStyle(']-[', widthA=.0, lengthA=0., angleA=None, widthB=width, lengthB=2.*width, angleB=None)
    fancy = f'fancy, head_length={head_length}, head_width={width}, tail_width={.5*width}'

    style = simple 
    
    linestyle = '-'
    if arrowstyle == 'wedge':
        style = wedge
    elif arrowstyle == 'anat':
        style = anat
    elif arrowstyle == 'bracket':
        style = bracket
    elif arrowstyle == 'fancy':
        style = fancy
        # linestyle = '-'

    angle = (.1 + int(route)/100.)
    arc = FancyArrowPatch((x0,y0),(x1,y1), linestyle=linestyle, arrowstyle=style, color=color, alpha=alpha, connectionstyle='arc3,rad=' + f'{angle}', zorder=zorder, lw=width, shrinkA=shrink_factor, shrinkB=shrink_factor, joinstyle='miter', mutation_scale=1.5)
    return arc


# def _position_communities(g, partition, **kwargs):

#     # create a weighted graph, in which each node corresponds to a community,
#     # and each edge weight to the number of edges between communities
#     between_community_edges = _find_between_community_edges(g, partition)

#     communities = set(partition.values())
#     hypergraph = nx.DiGraph()
#     hypergraph.add_nodes_from(communities)
#     for (ci, cj), edges in between_community_edges.items():
#         hypergraph.add_edge(ci, cj, weight=len(edges))

#     # find layout for communities
#     pos_communities = nx.spring_layout(hypergraph, **kwargs)

#     # set node positions to position of community
#     pos = dict()
#     for node, community in partition.items():
#         pos[node] = pos_communities[community]

#     return pos