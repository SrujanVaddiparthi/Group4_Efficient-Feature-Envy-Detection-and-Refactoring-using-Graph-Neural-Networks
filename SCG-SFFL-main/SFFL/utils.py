import argparse
import numpy as np
import torch
import random
import time
import pandas as pd
from sklearn import metrics
import pickle

def get_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--project', type=str, default='activemq', choices=['binnavi', 'activemq', 'kafka', 'alluxio', 'realm-java'])    
    parser.add_argument('--word_embedding_epochs', type=int, default=300, help='epochs for word2vec training')
    parser.add_argument('--conv', type=str, default='GAT', choices=['GAT', 'GCN', 'Sage'], help='convolution mode')
    parser.add_argument('--head_num', type=int, default=8, help='only works for GAT, should be factor of the hidden layer')
    parser.add_argument('--aggr', type=str, default='mean', choices=['mean', 'add', 'max'], help='aggregation mode, only works for Sage')
    parser.add_argument('--repeat_time', type=int, default=1)
    parser.add_argument('--encoding', type=int, default=1, choices=[1, 2, 3])
    parser.add_argument('--epochs', type=int, default=2400)
    parser.add_argument('--random_seed', type=int, default=0)
    parser.add_argument('--hidden_dim', type=int, default=256)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--device', type=str, default='cuda', choices=['cpu', 'cuda', 'cuda:0', 'cuda:1', 'cuda:2', 'cuda:3'])
    
    # for pretrain and fine-tune
    parser.add_argument('--pretrained_project', type=str, default='activemq', choices=['binnavi', 'activemq', 'kafka', 'alluxio', 'realm-java'])
    parser.add_argument('--fine_tuned_project', type=str, default='alluxio', choices=['binnavi', 'activemq', 'kafka', 'alluxio', 'realm-java'])
    parser.add_argument('--fine_tune_epochs', type=int, default=400)
    parser.add_argument('--fine_tune_data', type=float, default=0.1, choices=[0, 0.01, 0.05, 0.1])

    return parser

def split_dataset(labels: np.array, train_ratio=0.64, val_ratio=0.16, test_ratio=0.20, random_seed=42):
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


def print_metrics(class_pred, class_true, class_real, print_flag=False):
    assert class_pred.shape[0] == class_true.shape[0] and class_true.shape[0] == class_real.shape[0]
    
    # turn it into binary classification task
    y_true = (class_true != class_real).cpu()
    y_pred = (class_pred != class_real).cpu()

    acc_1 = metrics.accuracy_score(y_true, y_pred)
    precision_1 = metrics.precision_score(y_true, y_pred)
    recall_1 = metrics.recall_score(y_true, y_pred)
    f1_1 = metrics.f1_score(y_true, y_pred)

    hit_smell = torch.sum((class_pred == class_true) & (class_true != class_real))
    pred_smell = torch.sum(class_pred != class_real)
    total_smell = torch.sum(class_true != class_real)

    acc_2 = torch.sum(class_pred == class_true) / class_pred.shape[0]
    precision_2 = hit_smell / pred_smell
    recall_2 = hit_smell / total_smell
    f1_2 = 2 * precision_2 * recall_2 / (precision_2 + recall_2)
    if print_flag:
        print("Accuracy1: {:.2%}".format(acc_1))
        print("Precision1: {:.2%}".format(precision_1))
        print("Recall1: {:.2%}".format(recall_1))
        print("F1-Score1: {:.2%}".format(f1_1))
        print("Accuracy2: {:.2%}".format(acc_2.item()))
        print("Precision2: {:.2%}".format(precision_2))
        print("Recall2: {:.2%}".format(recall_2))
        print("F1-Score2: {:.2%}\n".format(f1_2))

    return acc_2, precision_2, recall_2, f1_2
    
def load_data(project):
    df1 = pd.read_csv(f'data/{project}/ground_truth.csv')
    df2 = pd.read_csv(f'data/{project}/method-invocate-method.csv')

    # egdes between methods and classes, representing ownership of method
    mc_own_edges = torch.LongTensor(df1[['method_id','source_class_id']].values.T)

    # egdes between methods, presenting method calling
    mm_call_edges = torch.LongTensor(df2[['caller_id', 'callee_id']].values.T)

    # refactored egdes between methods and classes, representing ownership of method
    fmc_own_edges = torch.LongTensor(df1[['method_id','target_class_id']].values.T) 
    
    with open(f'data/{project}/class_tokens.pkl', 'rb') as f:
        class_tokens = pickle.load(f)

    with open(f'data/{project}/method_tokens.pkl', 'rb') as f:
        method_tokens = pickle.load(f)

    labels = df1['label'].values

    class_num, method_num = len(class_tokens), len(method_tokens)

    # The adjacency matrix representing the directed graph of the calling relationship between methods.
    mm_call_adj = build_sparse_adjacency_matrix(edges=mm_call_edges, shape=(method_num, method_num), symmetric=False)

    # The adjacency matrix representing the directed graph of the ownership relationship between methods and classes.
    mc_own_adj = build_sparse_adjacency_matrix(edges=mc_own_edges, shape=(method_num, class_num), symmetric=False)

    # The adjacency matrix representing the directed graph of the ownership relationship between methods and classes after refactoring.
    fmc_own_adj = build_sparse_adjacency_matrix(edges=fmc_own_edges, shape=(method_num, class_num), symmetric=False)

    # The adjacency matrix representing the directed graph of the calling relationship between methods and classes.
    mc_call_adj = torch.sparse.mm(mm_call_adj, mc_own_adj).coalesce()                              

    return mc_own_adj, mc_call_adj, fmc_own_adj, class_tokens, method_tokens, labels


def build_sparse_adjacency_matrix(edges, shape, symmetric=True):
    # Create the adjacency matrix in COO format using torch.sparse_coo_tensor
    indices = torch.LongTensor(edges)
    values = torch.FloatTensor([1] * indices.shape[1])  # All values are 1, indicating the existence of an edge

    if symmetric:
        assert shape[0] == shape[1]
        # Swap indices and append the transposed indices to create symmetric adjacency matrix
        indices = torch.cat((indices, indices.flip(0)), dim=1)
        values = torch.cat((values, values), dim=0)

    adj = torch.sparse_coo_tensor(indices, values, shape, dtype=torch.float32)

    return adj.coalesce()


def multiclass_to_binary_probabilities(output, adj_mc_own):
    neg_prob = output[adj_mc_own.indices()[0, :], adj_mc_own.indices()[1, :]]
    neg_prob = neg_prob.view(-1, 1)
    
    pos_prob = 1 - neg_prob

    probs = torch.cat((neg_prob, pos_prob), dim=1)

    return probs