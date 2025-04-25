import argparse
import numpy as np
import torch
import random
import time
import pandas as pd
from sklearn import metrics
import pickle
import os  # Add os for file path operations

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
    assert train_ratio + val_ratio + test_ratio == 1
    np.random.seed(random_seed)

    def split_indices(indices, train_size, val_size):
        train_indices = indices[:train_size]
        val_indices = indices[train_size:train_size + val_size]
        test_indices = indices[train_size + val_size:]
        return train_indices, val_indices, test_indices

    pos_indices = np.where(labels == 1)[0]
    neg_indices = np.where(labels == 0)[0]
    np.random.shuffle(pos_indices)
    np.random.shuffle(neg_indices)

    pos_sample_num = len(pos_indices)
    neg_sample_num = len(neg_indices)

    pos_train_size = int(train_ratio * pos_sample_num)
    pos_val_size = int(val_ratio * pos_sample_num)
    neg_train_size = int(train_ratio * neg_sample_num)
    neg_val_size = int(val_ratio * neg_sample_num)

    pos_train_indices, pos_val_indices, pos_test_indices = split_indices(pos_indices, pos_train_size, pos_val_size)
    neg_train_indices, neg_val_indices, neg_test_indices = split_indices(neg_indices, neg_train_size, neg_val_size)

    train_indices = np.concatenate((pos_train_indices, neg_train_indices))
    val_indices = np.concatenate((pos_val_indices, neg_val_indices))
    test_indices = np.concatenate((pos_test_indices, neg_test_indices))

    np.random.shuffle(train_indices)
    np.random.shuffle(val_indices)
    np.random.shuffle(test_indices)

    print("=== The Results of Dataset Splitting ===")
    print("Train set - positive samples:", np.count_nonzero(labels[train_indices] == 1))
    print("Train set - negative samples:", np.count_nonzero(labels[train_indices] == 0))
    print("Validation set - positive samples:", np.count_nonzero(labels[val_indices] == 1))
    print("Validation set - negative samples:", np.count_nonzero(labels[val_indices] == 0))
    print("Test set - pos samples:", np.count_nonzero(labels[test_indices] == 1))
    print("Test set - neg samples:", np.count_nonzero(labels[test_indices] == 0))
    print()

    return train_indices, val_indices, test_indices

def print_metrics(class_pred, class_true, class_real, print_flag=False):
    assert class_pred.shape[0] == class_true.shape[0] and class_true.shape[0] == class_real.shape[0]
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
    base_path = os.path.dirname(__file__)
    data_dir = os.path.join(base_path, "data", project)

    df1 = pd.read_csv(os.path.join(data_dir, "ground_truth.csv"))
    df2 = pd.read_csv(os.path.join(data_dir, "method-invocate-method.csv"))

    mc_own_edges = torch.LongTensor(df1[['method_id','source_class_id']].values.T)
    mm_call_edges = torch.LongTensor(df2[['caller_id', 'callee_id']].values.T)
    fmc_own_edges = torch.LongTensor(df1[['method_id','target_class_id']].values.T) 

    with open(os.path.join(data_dir, "class_tokens.pkl"), 'rb') as f:
        class_tokens = pickle.load(f)

    with open(os.path.join(data_dir, "method_tokens.pkl"), 'rb') as f:
        method_tokens = pickle.load(f)

    labels = df1['label'].values
    class_num, method_num = len(class_tokens), len(method_tokens)

    mm_call_adj = build_sparse_adjacency_matrix(edges=mm_call_edges, shape=(method_num, method_num), symmetric=False)
    mc_own_adj = build_sparse_adjacency_matrix(edges=mc_own_edges, shape=(method_num, class_num), symmetric=False)
    fmc_own_adj = build_sparse_adjacency_matrix(edges=fmc_own_edges, shape=(method_num, class_num), symmetric=False)
    mc_call_adj = torch.sparse.mm(mm_call_adj, mc_own_adj).coalesce()                              

    return mc_own_adj, mc_call_adj, fmc_own_adj, class_tokens, method_tokens, labels

def build_sparse_adjacency_matrix(edges, shape, symmetric=True):
    indices = torch.LongTensor(edges)
    values = torch.FloatTensor([1] * indices.shape[1])

    if symmetric:
        assert shape[0] == shape[1]
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
