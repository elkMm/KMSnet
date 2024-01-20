import itertools
from collections import defaultdict
from matplotlib import cm, colors
import matplotlib.pyplot as plt 
import numpy as np
from matplotlib.patches import Ellipse, Circle, FancyArrowPatch, PathPatch, RegularPolygon
import mpl_toolkits.mplot3d.art3d as art3d
from mpl_toolkits.mplot3d import proj3d

import networkx as nx


def draw_multi_digraph(G, pos=None, layout=nx.spring_layout, node_shape='round', node_colors=None, node_labels=None, font_size=12, figsize=(10,10), ax=None):
    '''
    Draw a directed graph.

    Parameters
    ----------
    G : a networkx DiGraph object
    node_shape: str
        `circle` or `polygon`
    '''

    plt.rcParams.update({
        "figure.autolayout": True,
        "text.usetex": True,
        "font.family": "Times New Roman",
        "font.size": font_size
    })

    # pos = nx.kamada_kawai_layout(G)
    if pos is None:
        pos = layout(G)

    fig = plt.figure(figsize=figsize)
    # fig, ax = plt.subplots()

    
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

    
    draw_node = round_node
    width = .02
    if node_shape == 'polygon':
        draw_node = polygonal_node
        width = .03

    for node in pos:
        x, y = pos[node]
        draw_node((x,y), facecolor=color_map[node], width=width, ax=ax)
        xs.append(x)
        ys.append(y)
   
    edges = G.edges

    for edge in edges:
        draw_curved_edge(edge, pos=pos, ax=ax)

    if node_labels != None:
        # Draw node labels
        nx.draw_networkx_labels(
            G, 
            pos=pos, 
            labels=node_labels, 
            font_size=font_size, 
            font_color='k', 
            # font_family='sans-serif', 
            font_weight='normal', 
            horizontalalignment='left', 
            verticalalignment='bottom', 
            ax=ax
        )


    fig.tight_layout()
    ax.set_aspect('equal')
    plt.xlim([min(xs)-.4, max(xs)+.4])
    plt.ylim([min(ys)-.4, max(ys)+.4])
    # plt.show()


    
def draw_flow_mode(G, pos, with_labels=False, node_labels=None, node_size=30, max_node_size=80, width=.1, alpha=.5, node_color='white', edge_color='tab:blue', edgecolors='black', font_size=20, ax=None):
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

    


def polygonal_node(xy, width=1., edgecolor='Black', facecolor='tab:blue', zorder=4, alpha=.8, ax=None):
    '''Node in the form of polygon'''
    kwargs = dict(edgecolor=edgecolor, facecolor=facecolor, zorder=zorder, alpha=alpha)
    patch = RegularPolygon(xy, numVertices=5, radius=width, **kwargs)
    ax.add_patch(patch)



def round_node(xy, facecolor='yellow', width=0.06, edgecolor='Black', zorder=4, alpha=1, ax=None):
    '''Draw a node given a dictionary of node positions.'''
    # if ax is None:
    #     ax = plt.gca()

    ell = _node(xy, color=facecolor, width=width, edgecolor=edgecolor, zorder=zorder, alpha=alpha)
    ax.add_patch(ell)


def draw_curved_edge(edge, color='tab:gray', width=.2, alpha=0.5, zorder=3, pos=None, ax=None):
    '''Draws curved edge given the source and range positions of the edge'''

    # if ax is None:
    #     ax = plt.gca()

    arc = _curved_edge(edge=edge, color=color, width=width, alpha=alpha, zorder=zorder, pos=pos)
    ax.add_patch(arc)


def _node(xy, color='tab:red', width=0.02, edgecolor='Black', zorder=5, alpha=0.6):
    '''Plots an ellipse to represent a node given a dictionary of node positions.'''

    ell = Circle(xy, radius=width/2., zorder=zorder, lw=0.5,edgecolor=edgecolor, facecolor=color, alpha=alpha)
    return ell




def _curved_edge(edge, color='blue', width=0.3, alpha=.6, zorder=1, pos=None):
    '''Returns an object to represent an arc, used mainly for interlayer connections.'''

    n0, n1, route = edge
    x0,y0 = pos[n0]
    x1,y1 = pos[n1]
    style = "Simple, head_width=6, head_length=8"
    angle = (.5 + int(route)/45.)
    arc = FancyArrowPatch((x0,y0),(x1,y1), linestyle='-', arrowstyle=style, color=color, alpha=alpha, connectionstyle='arc3,rad=' + f'{angle}', zorder=zorder, lw=width)
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