import pandas as pd
from sklearn import metrics
import numpy as np


def refactor(project, preds, mm_call, idx_test):
    method_info = pd.read_csv('data/' + project + '/method.csv')
    ground_truth = pd.read_csv('data/'+project+'/ground_truth.csv')

    class_real = ground_truth['source_class_id'].values
    class_true = ground_truth['target_class_id'].values

    aggregated = method_info.groupby('classID')['nodeId'].apply(list).reset_index()
    class_own_list = aggregated['nodeId'].tolist()

    mc_call = mm_call_2_mc_call(mm_call, class_own_list)

    # set the calling strength between method and its class as zero
    mc_call[np.arange(mc_call.shape[0]), class_real] = 0
    
    # find class_id with max calling strength
    class_pred = np.argmax(mc_call, axis=1)          

    # if method is predicted as non-smelly, its target class should be its enclosing class
    class_pred[preds == 0] = class_real[preds == 0]

    # print metrics at test set
    print_metrics(class_pred[idx_test], class_true[idx_test], class_real[idx_test])


# call graph of methods -> call graph of methods and classes
# sum edge weight of same class
def mm_call_2_mc_call(mm_call, class_own_list):
    shape = (mm_call.shape[0], len(class_own_list))
    mc_call = np.zeros(shape)

    for i, indices in enumerate(class_own_list):
        mc_call[:, i] = np.sum(mm_call[:, indices], axis=1)

    return mc_call



def print_metrics(class_pred, class_true, class_real, print_flag=True):
    assert class_pred.shape[0] == class_true.shape[0] and class_true.shape[0] == class_real.shape[0]
    
    # turn it into binary classification task
    y_true = (class_true != class_real)
    y_pred = (class_pred != class_real)

    acc_1 = metrics.accuracy_score(y_true, y_pred)
    precision_1 = metrics.precision_score(y_true, y_pred)
    recall_1 = metrics.recall_score(y_true, y_pred)
    f1_1 = metrics.f1_score(y_true, y_pred)

    hit_smell = np.sum((class_pred == class_true) & (class_true != class_real))
    pred_smell = np.sum(class_pred != class_real)
    total_smell = np.sum(class_true != class_real)

    acc_2 = np.sum(class_pred == class_true) / class_pred.shape[0]
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