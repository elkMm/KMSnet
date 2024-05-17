'''Functions for measures by node classes.'''
from src.utils import nonzero_sum
from src.states import *

def to_node_class_dist(nodes, distr, nodes_by_classes, node_cls=None):
    '''Transform a distributions on the nodes to a distribution on the node classes.

    Parameters
    ----------
    nodes : list of nodes
    distr : array of same dimension as the length of nodes
    nodes_by_classes : dict
        Dictionary with keys the node classes and values list of nodes.
    node_cls : array, optional, default: list of keys of nodes_by_classes
        List of all node classes.
    '''
    if node_cls == None:
        node_cls = list(nodes_by_classes.keys())

    cls_dist = np.zeros(len(node_cls))

    for i, cl in enumerate(node_cls):
        cls_dist[i] = sum([distr[nodes.index(node)] for node in nodes_by_classes[cl]])

    cls_dist = cls_dist / nonzero_sum(cls_dist)
    
    return node_cls, cls_dist



def node_kms_emittance_profile_by_classes(graph, nodes_by_classes, nodelist, beta_min, beta_max, num=100):
    '''Returns KMS emittances of nodes in nodelist as a vector indexed over nodes_by_classes. 

    Basically these vectors are obtained as by the usual KMS states from the function node_kms_emittance_profile_variation, but here all components correspondint to nodes in the same class are added, which lead to a dimension reduction of the vectors.
    '''

    interval = temperature_range(beta_min, beta_max, num=num)
    
    KMSProfiles = {'range': interval} | {node: [] for node in nodelist}
    node_cls = list(nodes_by_classes.keys())

    for _, T in enumerate(interval):
        beta = 1./T
        nodes, Z = kms_emittance(graph, beta)
        for u in nodelist:
            ind = nodes.index(u)
            profile = Z[:, ind]
            profile_by_cl = to_node_class_dist(nodes, profile, nodes_by_classes, node_cls=node_cls)[1]

            u_cl = [c for c in node_cls if str(u) in nodes_by_classes[c]][0]
            i = node_cls.index(u_cl)
            profile_by_cl =  remove_ith(profile_by_cl, i)

            KMSProfiles[u] += [profile_by_cl, ]

            KMSProfiles.copy()

    return KMSProfiles | {'node_cls': node_cls}




