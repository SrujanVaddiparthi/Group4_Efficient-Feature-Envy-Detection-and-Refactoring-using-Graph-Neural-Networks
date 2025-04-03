import random
from scipy.spatial.distance import pdist, squareform
import torch
import torch.nn.functional as F
from sklearn.metrics.pairwise import euclidean_distances
import numpy as np
import time
random_state = 1


# Note that k_nearest_neighbors may not be connected to input node
def generate_synthetic_nodes(X:torch.Tensor, generate_num:int, k_neighbors=5):
    """
    Generate synthetic nodes based on k-nearest neighbors.

    Args:
        X: Tensor representing the original nodes.
        generate_num: Number of synthetic nodes to generate.
        k_neighbors: Number of nearest neighbors to consider.

    Returns:
        selected_nodes: Tensor representing the selected synthetic nodes.

    """

    node_num = X.shape[0]

    distances = torch.cdist(X, X)

    if k_neighbors+1 > X.shape[0] - 1:
        k_neighbors = X.shape[0] - 1
    _, idx = torch.topk(distances, k=k_neighbors+1, largest=False, dim=1, sorted=True)
    
    k_neighbors_idx = idx[:, 1:] # remove itself

    # Get the neighbors based on the idx
    neighbors = X[k_neighbors_idx, :]

    # Compute distances between neighbors and original nodes
    distances = neighbors - X.unsqueeze(1)
    
    # Generate random weights following a uniform distribution
    weights = torch.rand(node_num, k_neighbors, device=X.device)
    
    # Normalize the weights
    weights = F.normalize(weights, p=1, dim=1)
    
    # Generate synthetic nodes based on weights and distances
    synthetic_nodes = (weights.unsqueeze(2) * distances) + X.unsqueeze(1)
    
    # Flatten
    synthetic_nodes = synthetic_nodes.transpose(0, 1).reshape(-1, synthetic_nodes.shape[2])
    k_neighbors_idx = torch.flatten(k_neighbors_idx.T, start_dim=0)
    
    # Select node
    if generate_num <= node_num * k_neighbors:
        synthetic_nodes = synthetic_nodes[:generate_num]
        k_neighbors_idx = k_neighbors_idx[:generate_num]
    else:
        print(f'Failed to generate {generate_num} synthetic nodes, please increase k_neighbors or reduce generate_num.')
    
    return synthetic_nodes, k_neighbors_idx


def share_edges_with_synthetic_nodes(k_neighbors_idx, adj):
    # share nodes' edge with synthetic nodes
    synthetic_nodes_adj = adj[k_neighbors_idx,:]
    
    neighbors_num = k_neighbors_idx.shape[0]
    nodes_num = adj.shape[0]
    repeat_adj = adj.repeat(neighbors_num // nodes_num, 1)  # 先进行整数倍的重复拼接
    repeat_adj = torch.cat([repeat_adj, repeat_adj[:neighbors_num % nodes_num]], dim=0)
    
    synthetic_nodes_adj += repeat_adj

    return synthetic_nodes_adj

# default binary
def GraphSMOTE(X:torch.Tensor, y:torch.Tensor, train_idx:np.array, adj:torch.Tensor, stragery=0):
    # stragery = 0 -> build balance dataset
    # stragery = k -> generate k times for imbalance samples
    train_X, train_y, train_adj = X[train_idx], y[train_idx], adj[train_idx]

    class_ = 1  # imbalance class
    generate_adjs = []

    class_node_num = (train_y == class_).sum().item()
    
    if stragery == 0:
        _, count = most_frequent_integer(train_y)
        generate_num = count - class_node_num
    else:
        generate_num = int(class_node_num * stragery)

    class_train_idx = train_y == class_
    class_train_X = train_X[class_train_idx]
    class_train_adj = train_adj[class_train_idx, :]

    generate_X, neighbors_idx = generate_synthetic_nodes(class_train_X, generate_num)
    generate_y = torch.full((generate_num,), class_, dtype=torch.int, device=y.device)

    generate_idx = np.arange(y.shape[0], y.shape[0]+generate_num)
    generate_adj = share_edges_with_synthetic_nodes(neighbors_idx, class_train_adj)

    X = torch.cat((X, generate_X), dim=0)
    y = torch.cat((y, generate_y), dim=0)
    train_idx = np.concatenate([train_idx, generate_idx])
    generate_adjs.append(generate_adj)
    
    if generate_adjs:
        add_adj = torch.cat(generate_adjs, dim=0)
        add_num = add_adj.shape[0]
        adj_expended = adj.new(torch.Size((adj.shape[0] + add_num, adj.shape[0] + add_num))).fill_(0.0)
        adj_expended[:adj.shape[0], :adj.shape[0]] = adj
        adj_expended[adj.shape[0]:, :adj.shape[0]] = add_adj
        adj_expended[:adj.shape[0], adj.shape[0]:] = add_adj.T

    return X, y, train_idx, adj_expended


def most_frequent_integer(tensor: torch.Tensor) -> tuple:
    """
    Find the most frequent integer in a 1D torch tensor and count its occurrences.

    Args:
        tensor: 1D torch tensor of integers.

    Returns:
        most_frequent: Most frequent integer.
        count: Number of occurrences of the most frequent integer.
    """
    values, counts = tensor.unique(return_counts=True)
    max_count_index = counts.argmax()
    most_frequent = values[max_count_index].item()
    count = counts[max_count_index].item()
    
    return most_frequent, count