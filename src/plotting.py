import matplotlib.pyplot as plt
import seaborn as sns 
from collections import defaultdict
from distinctipy import get_colors, color_swatch
from .utils import is_qual
from .states import (
    emittance_matrix,
    kms_emittance,
    node_emittance_variation,
    node_kms_emittance_profile_entropy_range,
    node_kms_emittance_profile_diversity_range,
    KMS_emittance_dist_entropy_variation,
    node_structural_entropy
)


def plot_node_emittance(graph, nodelist, beta_min, beta_max, num=100, node_labels=None, font_size=12):
    plt.rcParams.update({
        "figure.autolayout": True,
        "text.usetex": True,
        "font.family": "Times New Roman",
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




def plot_node_kms_emittance_profile_entropy(graph, nodelist, beta_min, beta_max, num=50, node_labels=None, font_size=12):

    plt.rcParams.update({
        "figure.autolayout": True,
        "text.usetex": True,
        "font.family": "Times New Roman",
        "font.size": font_size
    })

    H = node_kms_emittance_profile_entropy_range(graph, nodelist, beta_min, beta_max, num=num)

    struc_entropy = node_structural_entropy(graph, nodelist=nodelist)

    xs = H['range']

    colors = get_colors(len(nodelist), pastel_factor=.5)

    
    labels = defaultdict(lambda: '')
    if node_labels != None:
        for n in node_labels:
            labels[n] = node_labels[n]
    
    for i, u in enumerate(nodelist):
        ys = H[u]
        s = struc_entropy[u]
        label = u
        color = colors[i]
        if labels[u] != '':
            label = labels[u]
        plt.plot(xs, ys, label=label, color=color)

        # find and plot the structural entropy of node u
        for ind, x in enumerate(xs):
            if is_qual(s, ys[ind]):
                plt.scatter(xs[ind], s, color=color)

    xlabel = f'Temperature 1/ß'    
    plt.xlabel(xlabel)
    plt.ylabel('Node profile entropy')
    # plt.legend()

def plot_node_kms_emittance_profile_diversity(graph, nodelist, beta_min, beta_max, num=50, node_labels=None, font_size=12):

    plt.rcParams.update({
        "figure.autolayout": True,
        "text.usetex": True,
        "font.family": "Times New Roman",
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
    xlabel = f'Temperature 1/ß'
    plt.xlabel(xlabel)
    plt.ylabel('Node profile diversity')
    plt.legend()

