import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
#%%
G = nx.karate_club_graph()
fish = []
e = 3

def Calc_B (A,m):
    global fish
    k = np.sum(A, axis = 1)
    kT = np.transpose(k)
    B = A - ((np.dot(k,kT))/(2*m))
    fish.append(k)
    fish.append(B)
    return B

def Mod (B , m):
    eigvalues , eigvectors = np.linalg.eig(B)
    global fish
    
    try:
        ev_max = np.max(eigvalues)
    except ValueError:
        ev_max = -1
    fish.append(ev_max)
    
    if ev_max > 0:
        ev_maxid = np.argmax(eigvalues)
        u = eigvectors[:,ev_maxid]
        s = ((u >= 0).astype(int) * 2) - 1
        fish.append(s)
        return s
    else:
        return np.zeros(1)

def parti (A,g,m):
    global fish
    B = Calc_B(A, m)
    s = Mod(B, m)
    
    if s.any() == bool(np.False_) :
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
    
def plot(g,pos1):
    global e
    global pos
    plt.figure(e, figsize=(10,8))
    nx.draw_networkx_nodes(g, pos1, node_size=300)
    nx.draw_networkx_edges(g, pos1, alpha=0.5)
    nx.draw_networkx_labels(g, pos1, font_size=10)
    
    
def graph(G):
    global fish
    global e
    global pos

    A = nx.to_numpy_array(G, weight="none") #into numpy matrix 
    m = 78 #total edges
    fish.append(A)
    C1,C2 = [],[]
    C1, C2 = parti(A,G,m)
    fish.append((C1,C2))
    print(C1)
    if C1 == [] or C2 == []:
        fish.append("YESSS")
        return G
    else:
        K = re(G, C1, C2)
        
        Subg1 = K.subgraph(C1).copy()
        Subg2 = K.subgraph(C2).copy()
        
        subg1 = graph(Subg1)
        subg2 = graph(Subg2)
        K = nx.compose(subg1, subg2)
        return K
        
    
plt.figure(figsize=(10, 8))
plt.title("Spectral Modularity Partitioning on Zachary's Karate Club")

D = nx.Graph()
D.add_nodes_from(["0","1","2"])
D.add_edges_from([("0","1"),("1","2"),("0","2")])

g = graph(G)

compo = list(nx.connected_components(g))
pos = nx.spring_layout(G, seed=42)
plt.figure(1)
nx.draw(G,pos, with_labels=True)
'''
nodec = []
for i in G.nodes:
    if i in C1:
        nodec.append("RoyalBlue")
    if i in C2:
        nodec.append("Red")
'''
plt.figure(2, figsize=(10,8))
nx.draw_networkx_nodes(g, pos, node_size=300)
nx.draw_networkx_edges(g, pos, alpha=0.5)
nx.draw_networkx_labels(g, pos, font_size=10)
plt.plot()

print(compo)

A = nx.to_numpy_array(g, weight="none") #into numpy matrix 
m = (np.sum(A))/2
B = Calc_B(A, m)
eigvalues , eigvectors = np.linalg.eig(B)
ev_max = np.max(eigvalues)

print(nx.algorithms.community.modularity(g,compo))

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