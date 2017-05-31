import networkx as nx
import matplotlib.pyplot as plt


def draw_graph(graph):
    connections = []
    for i in range(graph.shape[0]):
        for j in range(i+1, graph.shape[0]):
            if graph[i, j] == 1:
                connections.append((i, j))

    # extract nodes from graph
    nodes = set([n1 for n1, n2 in connections] + [n2 for n1, n2 in connections])

    # create networkx graph
    G = nx.Graph()

    # add nodes
    for node in nodes:
        G.add_node(node)

    # add edges
    for edge in connections:
        G.add_edge(edge[0], edge[1])

    # draw graph
    pos = nx.shell_layout(G)
    nx.draw(G, pos)

    # show graph
    plt.show()
