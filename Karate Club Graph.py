import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
#%%
G = nx.karate_club_graph()
H = nx.karate_club_graph()
fish = []
e = 2
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

g = Gra(G)

plt.figure(e, figsize=(10,8))
plt.title("Final Communities")
nx.draw_networkx_nodes(G, pos,node_color = Col, node_size=300)
nx.draw_networkx_edges(g, pos, alpha=0.5)
nx.draw_networkx_labels(g, pos, font_size=10)
plt.plot()

print(nx.algorithms.community.modularity(G,compo))
#%%

def spectral_modularity_bisection(G):
    """Performs community bisection on a graph using the leading eigenvector

    of the modularity matrix.
    """
    # 1. Get the adjacency matrix (A) and basic properties
    A = nx.to_numpy_array(G)
    nodes = list(G.nodes())
    n = len(nodes)
    degrees = np.sum(A, axis=1)
    m = np.sum(degrees) / 2

    # 2. Construct the Modularity Matrix (B)
    # B_ij = A_ij - (k_i * k_j) / 2m
    expected_edges = np.outer(degrees, degrees) / (2 * m)
    B = A - expected_edges

    # 3. Compute eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eigh(B)

    # 4. Find the leading eigenvector (corresponding to the largest eigenvalue)
    # np.linalg.eigh returns eigenvalues in ascending order, so the last one is the largest
    idx = np.argmax(eigenvalues)
    leading_eigenvector = eigenvectors[:, idx]

    # 5. Assign nodes to communities based on the sign of the eigenvector elements
    community_1 = []
    community_2 = []

    for i, node in enumerate(nodes):
        if leading_eigenvector[i] >= 0:
            community_1.append(node)
        else:
            community_2.append(node)

    return community_1, community_2, leading_eigenvector


# Initialize the classic Zachary's Karate Club graph
G = nx.karate_club_graph()

# Run the spectral modularity algorithm
comm1, comm2, leading_vec = spectral_modularity_bisection(G)

# Prepare node colors for visualization based on community assignment
node_colors = []
for node in G.nodes():
    if node in comm1:
        node_colors.append("skyblue")
    else:
        node_colors.append("salmon")

# Print details to console
print(f"Community 1 (Size {len(comm1)}): {comm1}")
print(f"Community 2 (Size {len(comm2)}): {comm2}")

# Plot the partitioned graph
plt.figure(figsize=(10, 8))
plt.title("Spectral Modularity Partitioning on Zachary's Karate Club")

# Use a spring layout for reproducible node positioning
pos = nx.spring_layout(G, seed=42)

# Draw the network elements
nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=500)
nx.draw_networkx_edges(G, pos, alpha=0.5)
nx.draw_networkx_labels(G, pos, font_size=10, font_weight="bold")

plt.axis("off")
plt.tight_layout()
plt.show()




#%%

G = nx.karate_club_graph()
G.remove_edge(32, 33)
g = G.edges
print(g)
#%%
'''
# 2. Inspect a node's alignment ('Mr. Hi' or 'Officer')
print(f"Node 5 belongs to: {G.nodes[5]['club']}")      # Output: 'Mr. Hi'
print(f"Node 9 belongs to: {G.nodes[9]['club']}")      # Output: 'Officer'

# 3. Color nodes based on their real faction split
color_map = []
for node in G.nodes:
    if G.nodes[node]['club'] == 'Mr. Hi':
        color_map.append('royalblue')
    else:
        color_map.append('red')

# 4. Plot the graph using a spring layout
pos = nx.spring_layout(G, seed = 69) # Consistent positioning
nx.draw(G, pos, node_color=color_map, with_labels=True, node_size=500, edge_color="gray")
plt.title("Zachary's Karate Club Split Factions")
plt.show()
'''