import os
import networkx as nx
import matplotlib.pyplot as plt
from collections import defaultdict
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns
import re

# Function to construct graphs from text data in multiple files
def construct_graphs_from_txt_files(directory, topics, num_train_docs, num_test_docs):
    all_graphs = defaultdict(list)
    for topic in topics:
        for i in range(1, num_train_docs + 1):
            filename = os.path.join(directory, f"{topic}{i}.txt")
            if os.path.exists(filename):
                with open(filename, 'r') as file:
                    words = file.read().strip().split()
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
                subgraph = nx.ego_graph(graph, node)
                subgraph_str = str(sorted(subgraph.edges()))
                subgraph_counts[subgraph_str] += 1
        frequent_subgraphs[topic] = {subgraph: count for subgraph, count in subgraph_counts.items() if count >= min_support}
    return frequent_subgraphs

# Function to preprocess training graphs (assuming they are already in NetworkX format)
def preprocess_training_graphs(training_graphs):
    return training_graphs

# Function to select common subgraphs among frequent subgraphs
def select_common_subgraphs(frequent_subgraphs, min_frequency):
    common_subgraphs = defaultdict(list)
    for topic, topic_subgraphs in frequent_subgraphs.items():
        common_subgraphs[topic] = [nx.DiGraph(eval(subgraph)) for subgraph, frequency in topic_subgraphs.items() if frequency >= min_frequency]
    return common_subgraphs

# Create a directed graph
def construct_graph_from_txt_file(file_path):
    with open(file_path, 'r') as file:
        words = file.read().strip().split('\n')
    graph = nx.DiGraph()
    for i in range(len(words) - 1):
        source_word = words[i]
        target_word = words[i + 1]
        graph.add_edge(source_word, target_word)
    return graph

# Merged the subgraphs of single topic to make visulize in 1 graph
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
    
# Function to compute MCS between two graphs
def compute_mcs(graph1, graph2):
    mcs_size = 0
    # Iterate over all nodes in graph1 and check if they are present in graph2
    for node1 in graph1.nodes():
        if node1 in graph2.nodes():
            mcs_size += 1
    return mcs_size

# Function to compute MCS between test graph and each merged common subgraph
def compute_mcs_with_test(test_graph, merged_graphs):
    mcs_scores = {}
    for topic, merged_graph in merged_graphs.items():
        mcs_scores[topic] = compute_mcs(test_graph, merged_graph)
    return mcs_scores

# Function to compute distance metric based on MCS between two graphs
def distance_metric(graph1, graph2):
    # Compute distance metric based on MCS between two graphs
    return compute_mcs(graph1, graph2)

def knn(train_graphs, test_graph, train_labels, k):
    # Compute distances between test graph and all training graphs
    distances = [(train_labels[i], distance_metric(train_graph, test_graph)) for i, train_graph in enumerate(train_graphs)]
    distances.sort(key=lambda x: x[1], reverse=True)
    nearest_labels = [label for label, _ in distances[:k]]
    label_counts = {}
    for label in nearest_labels:
        label_counts[label] = label_counts.get(label, 0) + 1
    predicted_label = max(label_counts, key=label_counts.get)
    return predicted_label



directory = 'C:/sem 6/text files/preprocessed data'
topics = ["fashion", "amazonsports", "disease"]
num_train_docs = 12
num_test_docs = 3

# Construct graphs for each topic
graphs = construct_graphs_from_txt_files(directory, topics, num_train_docs, num_test_docs)
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
    
k = 7 # Number of neighbors to consider

# Test file path (Replace with the path to your test file)
test_file_path = 'C:/sem 6/text files/preprocessed data/disease13.txt'


# Construct test graph from the test file
test_graph = construct_graph_from_txt_file(test_file_path)

mcs_scores = compute_mcs_with_test(test_graph, merged_graphs)

# Print MCS scores for each topic
print("MCS Scores:")
for topic, score in mcs_scores.items():
    print(f"Topic: {topic}, MCS Score: {score}")
predicted_topic = knn(list(merged_graphs.values()), test_graph, list(merged_graphs.keys()), k)
print("Predicted Topic:", predicted_topic)

true_label = "disease"

# True labels
true_labels = [true_label]

# Predicted labels
predicted_labels = [predicted_topic]

# Compute accuracy
accuracy = accuracy_score(true_labels, predicted_labels)

# Compute precision, recall, and F1-score
precision = precision_score(true_labels, predicted_labels, average='binary', pos_label=true_label)
recall = recall_score(true_labels, predicted_labels, average='binary', pos_label=true_label)
f1 = f1_score(true_labels, predicted_labels, average='binary', pos_label=true_label)

# Convert scores to percentage
accuracy_percent = accuracy * 100
precision_percent = precision * 100
recall_percent = recall * 100
f1_percent = f1 * 100

# Print scores
print("Accuracy:", accuracy_percent, "%")
print("Precision:", precision_percent, "%")
print("Recall:", recall_percent, "%")
print("F1-score:", f1_percent, "%")

# Define all possible labels
all_labels = ["amazonsports", "disease", "fashion"]

# Check if true label and predicted label are in all_labels
if true_label not in all_labels or predicted_topic not in all_labels:
    print("Error: True label or predicted label is not in the list of all possible labels.")
else:
    # Compute the confusion matrix for the single test document
    conf_matrix = confusion_matrix(true_labels, predicted_labels, labels=all_labels)

    # Plot confusion matrix
    plt.figure(figsize=(6, 4))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=all_labels, yticklabels=all_labels)

    # Annotate only the correct predictions with numerical values
    for i in range(len(all_labels)):
        for j in range(len(all_labels)):
            if all_labels[i] == true_label and all_labels[j] == predicted_topic:
                plt.text(j + 0.5, i + 0.5, str(conf_matrix[i, j]), ha='center', va='center', color='black', fontsize=10)
            else:
                plt.text(j + 0.5, i + 0.5, '', ha='center', va='center')

    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix")
    plt.show()
all_true_labels = []
all_predicted_labels = []

# Loop through each topic
for topic in topics:
    # Initialize lists to store true and predicted labels for the current topic
    true_labels = []
    predicted_labels = []

    # Loop through each test file of the current topic
    for i in range(num_train_docs + 1, num_train_docs + num_test_docs + 1):
        # Test file path
        test_file_path = os.path.join(directory, f"{topic}{i}.txt")

        # Construct test graph from the test file
        test_graph = construct_graph_from_txt_file(test_file_path)

        # Call the function to compute MCS between test graph and each merged common subgraph
        mcs_scores = compute_mcs_with_test(test_graph, merged_graphs)

        # Predict the label for the test file
        predicted_label = knn(list(merged_graphs.values()), test_graph, list(merged_graphs.keys()), k)

        # Append true label
        true_labels.append(topic)

        # Append predicted label
        predicted_labels.append(predicted_label)

    # Append true and predicted labels for the current topic to the overall lists
    all_true_labels.extend(true_labels)
    all_predicted_labels.extend(predicted_labels)

    # Compute metrics for the current topic
    accuracy = accuracy_score(true_labels, predicted_labels)*100
    precision = precision_score(true_labels, predicted_labels, average='micro', labels=np.unique(predicted_labels)) *100
    recall = recall_score(true_labels, predicted_labels, average='micro', labels=np.unique(predicted_labels))*100
    f1 = f1_score(true_labels, predicted_labels, average='micro', labels=np.unique(predicted_labels))*100

    print(f"Metrics for topic {topic}:")
    print("Accuracy:", accuracy, '%')
    print("Precision:", precision, '%')
    print("Recall:", recall, '%')
    print("F1-score:", f1, '%')
    plt.show()

# Compute metrics for all topics combined
combined_accuracy = accuracy_score(all_true_labels, all_predicted_labels)
combined_precision = precision_score(all_true_labels, all_predicted_labels, average='micro', labels=np.unique(all_predicted_labels))
combined_recall = recall_score(all_true_labels, all_predicted_labels, average='micro', labels=np.unique(all_predicted_labels))
combined_f1 = f1_score(all_true_labels, all_predicted_labels, average='micro', labels=np.unique(all_predicted_labels))

combined_accuracy_percentage = combined_accuracy * 100
combined_precision_percentage = combined_precision * 100
combined_recall_percentage = combined_recall * 100
combined_f1_percentage = combined_f1 * 100

print("Combined Accuracy:", combined_accuracy_percentage,'%')
print("Combined Precision:", combined_precision_percentage,'%')
print("Combined Recall:", combined_recall_percentage, '%')
print("Combined F1-score:", combined_f1_percentage, '%')
# Plot confusion matrix for all topics combined
conf_matrix = confusion_matrix(all_true_labels, all_predicted_labels)
plt.figure(figsize=(6, 4))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=topics, yticklabels=topics)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Combined Confusion Matrix for all Topics")
plt.show()
