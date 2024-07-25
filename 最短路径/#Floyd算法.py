#Folyd算法，最短路径
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

def read_graph_from_file(file_path):
    """
    Read a graph adjacency matrix from a file.

    Parameters:
    file_path (str): Path to the file containing the adjacency matrix.

    Returns:
    graph (2D list): A 2D list representing the adjacency matrix of the graph.
    """
    graph = []
    with open(file_path, 'r') as file:
        for line in file:
            row = list(map(float, line.split()))
            graph.append(row)
    return graph

def floyd_warshall(graph):
    """
    Floyd-Warshall algorithm to find shortest paths between all pairs of nodes.

    Parameters:
    graph (2D list or numpy array): A 2D list or numpy array representing the adjacency matrix of the graph.
                                    graph[i][j] represents the weight of the edge from node i to node j.
                                    If there is no edge, use a large number (e.g., np.inf) to represent it.

    Returns:
    dist (2D numpy array): A 2D numpy array where dist[i][j] is the shortest distance from node i to node j.
    """
    V = len(graph)
    dist = np.array(graph, dtype=float)

    # Apply Floyd-Warshall algorithm
    for k in range(V):
        for i in range(V):
            for j in range(V):
                if dist[i, j] > dist[i, k] + dist[k, j]:
                    dist[i, j] = dist[i, k] + dist[k, j]

    return dist

def visualize_graph(graph, shortest_paths):
    """
    Visualize the graph and the shortest path matrix.

    Parameters:
    graph (2D list or numpy array): The adjacency matrix of the graph.
    shortest_paths (2D numpy array): The shortest path matrix computed by the Floyd-Warshall algorithm.
    """
    G = nx.DiGraph()
    V = len(graph)

    for i in range(V):
        for j in range(V):
            if graph[i][j] != np.inf and i != j:
                G.add_edge(i, j, weight=graph[i][j])

    pos = nx.spring_layout(G)
    edge_labels = {(i, j): f'{graph[i][j]:.2f}' for i, j in G.edges()}

    plt.figure(figsize=(10, 6))
    nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=500, font_size=10, font_weight='bold', arrows=True)
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='red')
    plt.title("Graph with Edge Weights")
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.imshow(shortest_paths, cmap='viridis', interpolation='none')
    plt.colorbar()
    plt.title("Shortest Path Matrix")
    plt.xlabel("Destination Node")
    plt.ylabel("Source Node")
    plt.show()

def main():
    # User can choose to input data or read from file
    choice = input("Do you want to (1) input graph manually or (2) read from file? Enter 1 or 2: ").strip()

    if choice == '1':
        V = int(input("Enter the number of vertices: ").strip())
        graph = []
        print("Enter the adjacency matrix row by row (use 'inf' for no direct edge):")
        for i in range(V):
            row = input().strip().split()
            graph.append([float(x) if x != 'inf' else np.inf for x in row])
    elif choice == '2':
        file_path = input("Enter the path to the adjacency matrix file: ").strip()
        try:
            graph = read_graph_from_file(file_path)
        except Exception as e:
            print(f"Error reading file: {e}")
            return
    else:
        print("Invalid choice. Exiting.")
        return

    try:
        shortest_paths = floyd_warshall(graph)
        print("Shortest path matrix:")
        print(shortest_paths)

        visualize_graph(graph, shortest_paths)
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()


"""
#可以尝试运行一下普通版的，等理解额透彻后再操作上面的升级版本
import numpy as np

def floyd_warshall(graph):
   """ """
    Floyd-Warshall algorithm to find shortest paths between all pairs of nodes.
    
    Parameters:
    graph (2D list or numpy array): A 2D list or numpy array representing the adjacency matrix of the graph.
                                    graph[i][j] represents the weight of the edge from node i to node j.
                                    If there is no edge, use a large number (e.g., np.inf) to represent it.
                                    
    Returns:
    dist (2D numpy array): A 2D numpy array where dist[i][j] is the shortest distance from node i to node j.
    """"""
    V = len(graph)
    dist = np.array(graph, dtype=float)

    # Apply Floyd-Warshall algorithm
    for k in range(V):
        for i in range(V):
            for j in range(V):
                if dist[i, j] > dist[i, k] + dist[k, j]:
                    dist[i, j] = dist[i, k] + dist[k, j]

    return dist

# Example usage:
# Create a graph represented as an adjacency matrix
# Use np.inf to represent no direct edge between nodes
graph = [
    [0, 3, np.inf, 7],
    [8, 0, 2, np.inf],
    [5, np.inf, 0, 1],
    [2, np.inf, np.inf, 0]
]

shortest_paths = floyd_warshall(graph)

print("Shortest path matrix:")
print(shortest_paths)
"""