import argparse
import numpy as np
import torch
from sklearn import metrics
import pandas as pd

def get_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--project', type=str, default='kafka', choices=['binnavi', 'activemq', 'kafka', 'alluxio', 'realm-java']) 
    parser.add_argument('--class_num', type=int, default=2) 
    parser.add_argument('--random_seed', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--hidden_dim', type=int, default=64)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--up_scale', type=float, default=1)
    parser.add_argument('--weight', type=float, default=1E-6)
    parser.add_argument('--repeat_time', type=int, default=1)
    parser.add_argument('--device', type=str, default='cuda', choices=['cpu', 'cuda', 'cuda:0', 'cuda:1', 'cuda:2', 'cuda:3'])

    # for pretrain and fine-tune
    parser.add_argument('--pretrained_project', type=str, default='binnavi', choices=['binnavi', 'activemq', 'kafka', 'alluxio', 'realm-java'])
    parser.add_argument('--fine_tuned_project', type=str, default='activemq', choices=['binnavi', 'activemq', 'kafka', 'alluxio', 'realm-java'])
    parser.add_argument('--fine_tune_epochs', type=int, default=400)
    parser.add_argument('--fine_tune_data', type=float, default=0.1, choices=[0, 0.01, 0.05, 0.1])
    

    return parser

def load_data(project):
    nodes_data = pd.read_csv(f'data/{project}/metrics.csv')
    edges_data = pd.read_csv(f'data/{project}/method-invocate-method.csv')
    label_data = pd.read_csv(f'data/{project}/ground_truth.csv')

    features = torch.FloatTensor(nodes_data[['CC', 'PC', 'LOC', 'NCMIC', 'NCMEC', 'NECA', 'NAMFAEC']].values)
    labels = torch.LongTensor(label_data['label'].values)
    edges = torch.LongTensor(edges_data[['caller_id', 'callee_id']].values.T)

    # Create the adjacency matrix in COO format using torch.sparse_coo_tensor
    indices = torch.LongTensor(edges)
    values = torch.FloatTensor([1] * edges.shape[1])  # All values are 1, indicating the existence of an edge
    shape = torch.Size((labels.shape[0], labels.shape[0]))

    # Swap indices and append the transposed indices to create symmetric adjacency matrix
    indices_symmetric = torch.cat((indices, indices.flip(0)), dim=1)
    values_symmetric = torch.cat((values, values), dim=0)

    # Create the symmetric adjacency matrix as a sparse tensor
    adj_symmetric = torch.sparse_coo_tensor(indices_symmetric, values_symmetric, shape, dtype=torch.float32)

    adj_symmetric = adj_symmetric.coalesce()
    # Normalize features using torch.nn.functional.normalize
    features = torch.nn.functional.normalize(features, p=1, dim=1)

    return adj_symmetric, features, labels

def split_labels(labels: np.array, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2, random_seed=42):
    # Ensure the sum of ratios equals 1
    assert train_ratio + val_ratio + test_ratio == 1

    # Set the random seed
    np.random.seed(random_seed)

    def split_indices(indices, train_size, val_size):
        train_indices = indices[:train_size]
        val_indices = indices[train_size:train_size + val_size]
        test_indices = indices[train_size + val_size:]
        return train_indices, val_indices, test_indices

    # Get the indices for pos and neg samples
    pos_indices = np.where(labels == 1)[0]
    neg_indices = np.where(labels == 0)[0]

    # Shuffle the indices
    np.random.shuffle(pos_indices)
    np.random.shuffle(neg_indices)

    # Calculate the number of samples for each set
    pos_sample_num = len(pos_indices)
    neg_sample_num = len(neg_indices)

    # Calculate the number of samples for each class in train and val sets
    pos_train_size = int(train_ratio * pos_sample_num)
    pos_val_size = int(val_ratio * pos_sample_num)

    neg_train_size = int(train_ratio * neg_sample_num)
    neg_val_size = int(val_ratio * neg_sample_num)

    # Split the indices for each set
    pos_train_indices, pos_val_indices, pos_test_indices = split_indices(pos_indices, pos_train_size, pos_val_size)
    neg_train_indices, neg_val_indices, neg_test_indices = split_indices(neg_indices, neg_train_size, neg_val_size)

    # Concatenate the indices for each set
    train_indices = np.concatenate((pos_train_indices, neg_train_indices))
    val_indices = np.concatenate((pos_val_indices, neg_val_indices))
    test_indices = np.concatenate((pos_test_indices, neg_test_indices))

    # Shuffle the indices for each set
    np.random.shuffle(train_indices)
    np.random.shuffle(val_indices)
    np.random.shuffle(test_indices)

    # Print the number of positive and negative samples in each set
    train_pos_num = np.count_nonzero(labels[train_indices] == 1)
    train_neg_num = np.count_nonzero(labels[train_indices] == 0)

    val_pos_num = np.count_nonzero(labels[val_indices] == 1)
    val_neg_num = np.count_nonzero(labels[val_indices] == 0)

    test_pos_num = np.count_nonzero(labels[test_indices] == 1)
    test_neg_num = np.count_nonzero(labels[test_indices] == 0)

    print("=== The Results of Dataset Splitting ===")
    print("Train set - positive samples:", train_pos_num)
    print("Train set - negative samples:", train_neg_num)
    print(train_indices)
    print()
    print("Validation set - positive samples:", val_pos_num)
    print("Validation set - negative samples:", val_neg_num)
    print(val_indices)
    print()
    print("Test set - pos samples:", test_pos_num)
    print("Test set - neg samples:", test_neg_num)
    print(test_indices)
    print()

    return train_indices, val_indices, test_indices


def print_metrics(output, labels, print_flag=True):
    y_pred = output.max(1)[1].type_as(labels).cpu().detach()
    y_true = labels.cpu().detach()
    
    acc = metrics.accuracy_score(y_true, y_pred)
    precision = metrics.precision_score(y_true, y_pred)
    recall = metrics.recall_score(y_true, y_pred)
    f1 = metrics.f1_score(y_true, y_pred)
   
    if print_flag:
        print("Accuracy1: {:.2%}".format(acc))
        print("Precision1: {:.2%}".format(precision))
        print("Recall1: {:.2%}".format(recall))
        print("F1-Score1: {:.2%}\n".format(f1))

    return f1

def elementwise_sparse_dense_multiply(dense_matrix, sparse_matrix):
    indices = sparse_matrix._indices()
    sparse_values = sparse_matrix._values()
    
    dense_values = dense_matrix[indices[0,:], indices[1,:]]
    
    return torch.sparse.FloatTensor(indices, sparse_values * dense_values, sparse_matrix.size())


def adj_mse_loss(adj_rec, adj_tgt):
    edge_num = adj_tgt.nonzero().shape[0]
    total_num = adj_tgt.shape[0] ** 2

    neg_weight = edge_num / (total_num - edge_num)

    weight_matrix = adj_rec.new(adj_tgt.shape).fill_(1.0)
    weight_matrix[adj_tgt == 0] = neg_weight

    loss = torch.sum(weight_matrix * (adj_rec - adj_tgt) ** 2)

    return loss
