# -*- coding: utf-8 -*-
"""
Created on Sat Apr 20 18:20:36 2024

@author: ELITEBOOK
"""

import os
import networkx as nx
import matplotlib.pyplot as plt
from collections import defaultdict
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import numpy as np
import re

# Function to construct graphs from text data in multiple files
def construct_graphs_from_txt_files(directory, topics, num_train_docs, num_test_docs):
    all_graphs = defaultdict(list)
    for topic in topics:
        for i in range(1, num_train_docs + 1):
            filename = os.path.join(directory, f"{topic}{i}.txt")  # Construct file path
            if os.path.exists(filename):
                with open(filename, 'r') as file:
                    words = file.read().strip().split()  # Read words from the file
                    graph = nx.DiGraph()
                    graph.add_nodes_from(words)
                    for j in range(len(words) - 1):
                        graph.add_edge(words[j], words[j+1])
                    all_graphs[topic].append(graph)

    return all_graphs

# Function to count frequent subgraphs from a list of graphs
def count_frequent_subgraphs(graphs, min_support):
    frequent_subgraphs = defaultdict(dict)
    for topic, topic_graphs in graphs.items():
        subgraph_counts = defaultdict(int)
        for graph in topic_graphs:
            for node in graph.nodes():
                subgraph = nx.ego_graph(graph, node)  # Extract ego graph
                subgraph_str = str(sorted(subgraph.edges()))  # Convert subgraph to a string representation
                subgraph_counts[subgraph_str] += 1
        frequent_subgraphs[topic] = {subgraph: count for subgraph, count in subgraph_counts.items() if count >= min_support}

    return frequent_subgraphs

# Function to preprocess training graphs (assuming they are already in NetworkX format)
def preprocess_training_graphs(training_graphs):
    # Optionally perform any preprocessing steps such as node/edge attribute removal, etc.
    return training_graphs

# Function to select common subgraphs among frequent subgraphs
def select_common_subgraphs(frequent_subgraphs, min_frequency):
    common_subgraphs = defaultdict(list)
    for topic, topic_subgraphs in frequent_subgraphs.items():
        common_subgraphs[topic] = [nx.DiGraph(eval(subgraph)) for subgraph, frequency in topic_subgraphs.items() if frequency >= min_frequency]

    return common_subgraphs

def merge_common_subgraphs(common_subgraphs):
    merged_graphs = {}
    for topic, subgraphs in common_subgraphs.items():
        merged_graph = nx.DiGraph()
        for subgraph in subgraphs:
            merged_graph = nx.compose(merged_graph, subgraph)
        merged_graphs[topic] = merged_graph
    return merged_graphs

# Function to visualize a subgraph
def visualize_graph(graph, title="Graph"):
    plt.figure(figsize=(8, 6))
    pos = nx.spring_layout(graph)
    nx.draw(graph, pos, with_labels=True, node_color='skyblue', node_size=300, edge_color='black', linewidths=1, arrows=True)
    plt.title(title)
    plt.show()
def maximal_common_subgraph(graph1, graph2):
    def backtrack(mapping):
        nonlocal max_common_subgraph, max_common_nodes_set
        
        if len(mapping) > max_length:
            max_common_subgraph = nx.subgraph(graph1, mapping.keys())
            max_common_nodes_set = set(mapping.keys())
            return
        
        for node1 in graph1.nodes():
            if node1 not in mapping:
                for node2 in graph2.nodes():
                    if node2 not in mapping.values():
                        new_mapping = dict(mapping)
                        new_mapping[node1] = node2
                        common_neighbors = {n for n in graph1.neighbors(node1)} & {n for n in graph2.neighbors(node2)}
                        try:
                            if all(new_mapping[n1] in common_neighbors for n1 in graph1.neighbors(node1)):
                                backtrack(new_mapping)
                                break
                        except KeyError:
                            pass
    
    # Convert graph2 to NetworkX DiGraph object if it's provided as a string
    if isinstance(graph2, str):
        graph2 = nx.DiGraph(eval(graph2))
    
    # Convert graph1 to NetworkX DiGraph object if it's not already
    if not isinstance(graph1, nx.DiGraph):
        graph1 = nx.DiGraph(graph1)
    
    # Convert graph2 to NetworkX DiGraph object if it's not already
    if not isinstance(graph2, nx.DiGraph):
        graph2 = nx.DiGraph(graph2)
    
    max_common_subgraph = None
    max_common_nodes_set = set()
    max_length = 0
    
    backtrack({})
    
    return max_common_subgraph




# Function to compute the distance between two graphs based on their MCS
def graph_distance(graph1, graph2):
    # Calculate the MCS between the two graphs
    mcs = maximal_common_subgraph(graph1, graph2)
    # print("Graph 1 Nodes:", len(graph1.nodes()))
    # print("Graph 2 Nodes:", len(graph2.nodes()))
    if mcs is not None:
        print("MCS Nodes:", len(mcs.nodes()))
        # Compute the Euclidean distance as the difference in the number of nodes
        distance = len(graph1.nodes()) - len(mcs.nodes())
    else:
        print("No maximal common subgraph found.")
        # Set the distance to a large value if no maximal common subgraph is found
        distance = float('inf')
    
    return distance
def classify_document_knn(test_graph, merged_graphs, k):
    class_votes_sum = defaultdict(float)  # Initialize class votes sum with float to handle division later
    class_votes_count = defaultdict(int)   # Track the count of votes for each class
    for train_topic, train_graph_list in merged_graphs.items():
        for train_graph in train_graph_list:
            distance = graph_distance(test_graph, merged_graphs)
            print(distance)
            if distance != 0:  # Avoid division by zero
                class_votes_sum[train_topic] += 1 / distance
                class_votes_count[train_topic] += 1
            print(f"Distance between test graph and '{train_topic}' graph: {distance}")
            print(f"Class votes for '{train_topic}': {class_votes_sum[train_topic]}")
    max_average_vote = float('-inf')  # Initialize the maximum average vote
    predicted_class = None  # Initialize the predicted class
    for topic, votes_sum in class_votes_sum.items():
        if class_votes_count[topic] > 0:  # Avoid division by zero
            average_vote = votes_sum / class_votes_count[topic]  # Compute the average vote
            if average_vote > max_average_vote:
                max_average_vote = average_vote
                predicted_class = topic
  
    return predicted_class
def construct_graph_from_txt_file(file_path):
    with open(file_path, 'r') as file:
        words = file.read().strip().split('\n')
    
    # Create a directed graph
    graph = nx.DiGraph()
    
    # Add edges between consecutive words
    for i in range(len(words) - 1):
        source_word = words[i]
        target_word = words[i + 1]
        graph.add_edge(source_word, target_word)
    
    return graph
# Example usage:
directory = 'D:/sem6/GT/preprocessed data'  # Replace with the directory path containing your TXT files
topics = ["fashion", "amazonsports", "disease"]
num_train_docs = 12
num_test_docs = 3

# Construct graphs for each topic
graphs = construct_graphs_from_txt_files(directory, topics, num_train_docs, num_test_docs)
# Preprocess training graphs
training_graphs = preprocess_training_graphs(graphs)

# Count frequent subgraphs from training graphs
min_support = 2  # Minimum support threshold for frequent subgraph mining
frequent_subgraphs = count_frequent_subgraphs(training_graphs, min_support)
# Select common subgraphs
min_frequency = 3  # Minimum frequency threshold for common subgraphs
common_subgraphs = select_common_subgraphs(frequent_subgraphs, min_frequency)
merged_graphs = merge_common_subgraphs(common_subgraphs)


# Visualize merged graph for each topic
print("Number of merged graphs:", len(merged_graphs))
for topic, merged_graph in merged_graphs.items():
    print(f"Visualizing merged graph for topic: {topic}")
    visualize_graph(merged_graph, title=f"Merged Common Subgraphs for {topic}")
k = 5  # Number of neighbors to consider

# Test file path (Replace with the path to your test file)
test_file_path = 'D:/sem6/GT/preprocessed data/fashion13.txt'

# Construct test graph from the test file
test_graph = construct_graph_from_txt_file(test_file_path)

# Debug print to check if the test graph is constructed correctly
print("Test Graph Nodes:", len(test_graph.nodes()))
print("Test Graph Edges:", test_graph.edges())

# Check if the test graph is not empty
if test_graph:
    # Classify the test document using KNN algorithm
    test_prediction = classify_document_knn(test_graph, merged_graphs, k)
    print("Test Prediction:", test_prediction)
else:
    print("Test graph is empty. Unable to classify.")

