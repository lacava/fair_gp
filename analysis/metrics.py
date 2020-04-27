import numpy as np
import pandas as pd


def MAE(y_true, y_pred):
    return np.mean(np.abs(y_pred - y_true))

def false_positives(y_true, y_pred):
    """returns false positives for each sample"""
    return np.array([yt != 1 if yp==1 else 0 for (yt,yp) in zip(y_true, y_pred)])

def mean_marginal_fairness(scores, X_protected):
    """Calculates the average fairness over all protected groups, 
    relative to a score vector"""
#     pdb.set_trace()
    base_score = np.mean(scores)
    count = 0
    avg_score = 0
    for col in X_protected.columns:
        for val in X_protected[col].unique():
            group_idx = np.flatnonzero( X_protected[col].values == val )
            avg_score += np.abs(base_score - np.mean(scores[group_idx]))
            count += 1
                                
    avg_score /= count
    return avg_score

def marginal_unfairness(scores, X_protected):
    """Calculates the max unfairness over all levels of protected groups, 
    relative to a score vector"""
#     pdb.set_trace()
    base_score = np.mean(scores)
    score = []
    
    for col in X_protected.columns:
        for val in X_protected[col].unique():
            group_idx = np.flatnonzero( X_protected[col].values == val )
            score.append(np.abs(base_score - np.mean(scores[group_idx])))
    return scores

def subgroup_unfairness(scores, X_protected):
    """Calculates the max unfairness over all protected subgroups, 
    defined as all unique samples in X_protected. determined relative to a score vector"""
#     pdb.set_trace()
    base_score = np.mean(scores)
    score = []
    X_unique = X_protected.drop_duplicates() 
    for _, x in X_unique.iterrows():
#         pdb.set_trace()
        group_idx = np.flatnonzero( 
            [np.array(row.values == x.values).all() for _, row in X_protected.iterrows()] )
        tmp = np.array([row.values == x.values for _, row in X_protected.iterrows()]).all()
        score.append(np.abs(base_score - np.mean(scores[group_idx])))
    return score
                                    
# aggregate functions
def mean_marginal_unfairness(scores, X_protected):
    return np.mean(marginal_unfairness(scores, X_protected))

def max_marginal_unfairness(scores, X_protected):
    return np.max(marginal_unfairness(scores, X_protected))
                                
def max_subgroup_unfairness(scores, X_protected):
    return np.max(subgroup_unfairness(scores, X_protected))
                                     
def mean_subgroup_unfairness(scores, X_protected):
    return np.mean(subgroup_unfairness(scores, X_protected))

