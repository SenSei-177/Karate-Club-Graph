import networkx as nx
import matplotlib.pyplot as plt

# 1. Load the built-in dataset
G = nx.karate_club_graph()

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
