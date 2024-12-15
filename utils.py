# util functions for RGAN
# Status: in developing

import json
from sklearn.metrics import auc
import numpy as np


def read_json(filename):
    with open(filename) as buf:
        return json.loads(buf.read())

def compute_closeness(predictions, labels):
    # Example: Mean squared error per prediction
    predictions = np.concatenate(predictions, axis=0)
    labels = np.concatenate(labels, axis=0)
    squared_errors = (predictions - labels) ** 2
    closeness_scores = np.sum(squared_errors, axis=1)
    return closeness_scores
        
def compute_aucs(fpr_list, tpr_list):
    aucs = []
    for fpr, tpr in zip(fpr_list, tpr_list):
        if fpr is not None and tpr is not None:
            aucs.append(auc(fpr, tpr))
    return aucs

def flatten_and_compute_auc(data):
    aucs = []
    for entry in data:
        for fpr, tpr in zip(*entry):
            try:
                auc_value = auc(fpr, tpr)
                aucs.append(auc_value)
            except ValueError as e:
                print(f"Error computing AUC: {e}")
    return aucs