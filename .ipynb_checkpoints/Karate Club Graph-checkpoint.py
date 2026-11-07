import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

G = nx.karate_club_graph()
H = nx.karate_club_graph()
fish = []
e = 1
colours = ["Red","RoyalBlue","Green","Yellow","Violet"]

def Master_B (A,m):
    global Master_B
    
    k = np.sum(A, axis = 1)
    B = A - ((np.outer(k,k))/(2*m))
    Master_B = B

def Calc_B (g):
    global Master_B
    
    C = list(g.nodes)
    Idx = np.ix_(*(C,C))
    return Master_B[Idx]
    
def Mod (B , m):
    
    eigvalues , eigvectors = np.linalg.eigh(B)
    
    try:
        ev_max = np.max(eigvalues)
    except ValueError:
        ev_max = 0
    
    if ev_max > 0:
        ev_maxid = np.argmax(eigvalues)
        u = eigvectors[:,ev_maxid]
        s = ((u > 0).astype(int) * 2) - 1
        return s
    else:
        return np.zeros(1)

def parti (A,g,m):
    
    B = Calc_B(g)
    s = Mod(B, m)
    
    if s.any() == bool(np.False_):
        C1, C2 = [], []
        
    else:
        community_dict = {node:int(val) for node, val in zip(g.nodes(), s.flatten())}
        C1, C2 = [], []
        for i in community_dict.keys():
            if community_dict[i] == 1:
                C1.append(i)
            elif community_dict[i] == -1:
                C2.append(i)
    return C1, C2

def re(A,C1,C2):
    H = G.copy()
    I = G.copy()
    J = G.copy()
    K = G.copy()
    
    H.remove_nodes_from(C1)
    I.remove_nodes_from(C2)

    J.remove_edges_from(H.edges(data=True))
    J.remove_edges_from(I.edges(data=True))
    K.remove_edges_from(J.edges)
        
    return K
    
def plot(g):
    global e
    global pos
    global H
    global colours
    global Col
    global compo
    
    H1 = H.copy()
    g1 = g.copy()
    J = H.copy()
    K = H.copy()
       
    H1.remove_nodes_from(g1)

    J.remove_edges_from(g1.edges(data=True))
    J.remove_edges_from(H1.edges(data=True))
    K.remove_edges_from(J.edges)
    
    if nx.utils.graphs_equal(K,H):
        return
    H = K
    
    Col = ["Grey"]*34
    z=0
    compo = list(nx.connected_components(K))
    for i in compo:
        for j in i:
            Col[j] = colours[z]
        z += 1    
        
    plt.figure(e, figsize=(10,8))
    nx.draw_networkx_nodes(K, pos ,node_color = Col, node_size=300)
    nx.draw_networkx_edges(K, pos, alpha=0.5)
    nx.draw_networkx_labels(K, pos, font_size=10)
    e += 1
    
def graph(G,m):
    global fish
    global e
    global pos

    A = nx.to_numpy_array(G, weight="none") #into numpy matrix
    C1,C2 = [],[]
    C1, C2 = parti(A,G,m)
    if C1 == [] or C2 == []:
        return G
    else:
        K = re(G, C1, C2)
        plot(K)

        Subg1 = K.subgraph(C1).copy()
        Subg2 = K.subgraph(C2).copy()
        
        subg1 = graph(Subg1,m)
        subg2 = graph(Subg2,m)
        K = nx.compose(subg1, subg2)
        
        return K
        
def Gra(G):
    A = nx.to_numpy_array(G, weight="none") #into numpy matrix 
    m = (np.sum(A))/2
    Master_B(A,m)
    return graph(G,m)

plt.figure(figsize=(10, 8))
plt.title("Spectral Modularity Partitioning on Zachary's Karate Club")

pos = nx.spring_layout(G, seed=42)
plt.figure(1)
nx.draw(G,pos, with_labels=True)
e += 1

g = Gra(G)

plt.figure(e, figsize=(10,8))
plt.title("Final Communities")
nx.draw_networkx_nodes(G, pos,node_color = Col, node_size=300)
nx.draw_networkx_edges(g, pos, alpha=0.5)
nx.draw_networkx_labels(g, pos, font_size=10)
plt.plot()

print(nx.algorithms.community.modularity(G,compo))


