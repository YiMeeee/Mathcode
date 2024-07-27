"""
选择手动输入图的数据或从文件读取数据。
如果选择手动输入，输入顶点数量和邻接矩阵。
如果选择从文件读取，输入文件路径。
输入起始顶点。
程序将计算最短路径，并显示图和结果。
"""
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import heapq

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

def dijkstra(graph, start_vertex):
    """
    Dijkstra's algorithm to find the shortest paths from a start vertex to all other vertices.

    Parameters:
    graph (2D list or numpy array): A 2D list or numpy array representing the adjacency matrix of the graph.
                                    graph[i][j] represents the weight of the edge from node i to node j.
                                    If there is no edge, use a large number (e.g., np.inf) to represent it.
    start_vertex (int): The starting vertex.

    Returns:
    distances (list): A list where distances[i] is the shortest distance from the start vertex to vertex i.
    predecessors (list): A list where predecessors[i] is the predecessor of vertex i in the shortest path.
    """
    V = len(graph)
    distances = [np.inf] * V
    predecessors = [None] * V
    distances[start_vertex] = 0
    priority_queue = [(0, start_vertex)]
    
    while priority_queue:
        current_distance, current_vertex = heapq.heappop(priority_queue)
        
        if current_distance > distances[current_vertex]:
            continue
        
        for neighbor, weight in enumerate(graph[current_vertex]):
            if weight != np.inf:
                distance = current_distance + weight
                
                if distance < distances[neighbor]:
                    distances[neighbor] = distance
                    predecessors[neighbor] = current_vertex
                    heapq.heappush(priority_queue, (distance, neighbor))
    
    return distances, predecessors

def visualize_graph(graph, start_vertex, distances, predecessors):
    """
    Visualize the graph and the shortest path from the start vertex.

    Parameters:
    graph (2D list or numpy array): The adjacency matrix of the graph.
    start_vertex (int): The starting vertex.
    distances (list): A list where distances[i] is the shortest distance from the start vertex to vertex i.
    predecessors (list): A list where predecessors[i] is the predecessor of vertex i in the shortest path.
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
    shortest_paths_tree = nx.DiGraph()
    for i in range(V):
        if predecessors[i] is not None:
            shortest_paths_tree.add_edge(predecessors[i], i, weight=graph[predecessors[i]][i])

    nx.draw(shortest_paths_tree, pos, with_labels=True, node_color='lightgreen', node_size=500, font_size=10, font_weight='bold', arrows=True)
    shortest_path_labels = {(i, j): f'{graph[i][j]:.2f}' for i, j in shortest_paths_tree.edges()}
    nx.draw_networkx_edge_labels(shortest_paths_tree, pos, edge_labels=shortest_path_labels, font_color='red')
    plt.title(f"Shortest Paths from Vertex {start_vertex}")
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

    start_vertex = int(input("Enter the start vertex: ").strip())
    
    try:
        distances, predecessors = dijkstra(graph, start_vertex)
        print("Shortest distances from vertex", start_vertex, "to all other vertices:")
        print(distances)
        print("Predecessors in the shortest path tree:")
        print(predecessors)

        visualize_graph(graph, start_vertex, distances, predecessors)
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
