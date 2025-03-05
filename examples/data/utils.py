from itertools import combinations
from collections import Counter
import seaborn as sns
import matplotlib.pyplot as plt
import networkx as nx


def in_degree(edges, node):
    in_nghbs = [e for e in edges if (e[1] == node)]
    return len(in_nghbs)

def out_degree(edges, node):
    out_nghbs = [e for e in edges if (e[0] == node)]
    return len(out_nghbs)

def in_neighbours(V, E, node):
    '''Returns all the in-neighbours of the node, given a set of nodes 
    and a set of edges.'''
    return [v for v in V if ((v, node) in E)]

def out_neighbours(V, E, node):
    '''Returns all the out-neighbours of the node, given a set of nodes 
    and a set of edges.'''
    return [v for v in V if ((node, v) in E)]

def all_neighbours(V, E, node):
    '''Returns all the in and out neighbours.'''
    return in_neighbours(V, E, node) + out_neighbours(V, E, node)

def in_neighbour_subraph(G, node):
    '''Return the subgraph formed by the in-neighbours of the node'''
    V = G.nodes
    E = G.edges 

    Nj = in_neighbours(V, E, node)
    Ej = [e for e in E if ((e[0] in Nj) and (e[1] in Nj))]

    Gj = nx.DiGraph()
    Gj.add_nodes_from(Nj)
    Gj.add_edges_from(Ej + [e for e in E if (e[1] == node)])
    return Gj

def conjugate_graph(G):
    '''
    Returns the conjugate of a directed graph.

    Notes
    -----
    The conjugate of a directed graph *G=(V,E)* has the same
    vertex set *V* and the edges are the elements of *E* with the directions reversed. 
    '''
    V = G.nodes
    E = G.edges

    E_bar = []
    for n1, n2 in E:
        E_bar += [(n2,n1),]

    G_bar = nx.DiGraph()
    G_bar.add_nodes_from(V)
    G_bar.add_edges_from(E_bar)

    return G_bar

def density(G):
    '''Returns the density of a directed graph.

    Parameters
    ----------
    G : a nx.DiGraph object
    
    '''
    V = G.nodes
    E = G.edges
    N = len(V)
    M = len(E)

    return float(M/(N*N))

def avg_in_degree(G):
    '''Returns the average in-degree of G.'''
    N = len(G.nodes)
    indegs = [d[1] for d in G.in_degree()]
    return float(sum(indegs) / N)

def avg_out_degree(G):
    '''Returns the average in-degree of G.'''
    N = len(G.nodes)
    outdegs = [d[1] for d in G.out_degree()]
    return float(sum(outdegs) / N)

def reciprocity(G):
    '''Returns the proportion of reciprocal connections
    among connected nodes in G.
    
    Notes
    -----

    There is a reciprocal connection between node *i* and node *j*
    if there is a directed edge from the former to the later and 
    another directed edge from the later to the former. So an edge *e: i --> j* is reciprocal
    if there is another edge *j --> i*.
    '''
    E = G.edges
    # M = len(E)
    V = G.nodes
    
    V_comb = combinations(V, 2)

    con_pair_count = 0
    reciprocal_con_count = 0
    for n0, n1 in V_comb:
        if ((n0, n1) in E) or ((n1, n0) in E):
            con_pair_count += 1
        if ((n0, n1) in E) and ((n1, n0) in E):
            reciprocal_con_count += 1

    return float(reciprocal_con_count / con_pair_count)
   
    

def is_reciprocal(G, e):
    '''Returns True if the connection given by the directed edge *e* is reciprocal.

    Notes
    -----
    There is a reciprocal connection between node *i* and node *j*
    if there is a directed edge from the former to the later and 
    another directed edge from the later to the former. So an edge *e: i --> j* is reciprocal
    if there is another edge *j --> i*.
    
    '''
    E = G.edges
    if (e[1], e[0]) in E:
        return True
    else:
        return False




def edge_stretches(G, edge, length=2):
    '''
    Given an edge in the directed graph G, returns 
    all the n-stretches of the specified length.

    Parameters
    ----------
    G : a networkx DiGraph object
    edge : 2-tuple of str
    length : int, default = 2
       Length of the streches

    Returns
    -------
    stretches : list of 3-tuples.

    Notes
    -----
    An `n-stretch` for an edge is a path of length n
    with the same source and range as the edge.
    
    '''
    s, r = edge
    E = [e for e in G.edges if ((e[0] == s) or (e[1]== r))]
    nodes = [n for n in G.nodes if n not in [s, r]]

    stretches = []
    for node in nodes:
        if ((s, node) in E) and ((node, r) in E):
            stretch = (s,node,r)
            stretches += [stretch,]

    return stretches

def stretches(G, length=2):
    '''
    Returns the stretches of all edges of G.
    '''

    E = G.edges 

    S = {}

    for e in E:
        stretches = edge_stretches(G, e, length=length)
        count = len(stretches)
        S[e] = {
            'stretches': stretches,
            'count': count
        }
       
    return S 

def edge_inflow_triangles(E, edge):
    '''
    Returns the inflow triangles at an edge.
    '''
    s, t = edge
    edges = [e for e in E if (e[1] == t)]
    # nodes = [n for n in G.nodes if n not in [s, r]]
    # stretches = edge_stretches(G, edge)

    inflow_tri = []

    for e in edges:
        if ((e[0], s) in E) or ((s, e[0]) in E):
            inflow_tri += [(e[0], s, t),]
        # if ((s, e[0]) in E):
        #     inflow_tri += [(s, e[0], t),]
    
    return inflow_tri



# def are_influent(G, e1, e2):
#     '''
#     Returns True if the two edges are influent.

#     Notes
#     -----
#     Two edges are *influent* if they have the same target.
#     '''
    
#     if (e1[1] == e2[1]):
#         return True
#     else:
#         return False
    







    
    
    






