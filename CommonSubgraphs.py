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

# Example usage:
directory = 'C:/sem 6/text files/preprocessed data'  # Replace with the directory path containing your TXT files
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
for topic, merged_graph in merged_graphs.items():
    visualize_graph(merged_graph, title=f"Merged Common Subgraphs for {topic}")