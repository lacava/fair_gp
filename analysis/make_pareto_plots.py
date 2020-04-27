from utils import pareto_compare_plots, pareto_multicompare_plots
from glob import glob
import pdb
import json
from tqdm import tqdm 
from collections import namedtuple
perf = namedtuple('perf',['method','dataset','seed'])
# rdir = 'results_r4_mod'
rdir = 'results_r6'
performances = {} 
methods = set()
datasets = set()
seeds = set()

for f in tqdm(glob(rdir+'/*.json')):
    with open(f) as fh:
        a = json.load(fh)
        method = a[0]['method']
        dataset = a[0]['dataset']
        seed = a[0]['seed']
        performances.update({perf(method,dataset,seed): a})
        methods.add(method)
        datasets.add(dataset)
        seeds.add(seed)
        
print('methods:',methods)
print('datasets:',datasets)
print('seeds:',seeds)

# pairwise comparisons
# from itertools import combinations
# import matplotlib.pyplot as plt
# for m1,m2 in combinations(methods,2):
#     for d in datasets:
#         for s in seeds:
#             if perf(m1,d,s) in performances.keys() and perf(m2,d,s) in performances.keys():
#                 h = pareto_compare_plots(performances[perf(m1,d,s)],
#                                      performances[perf(m2,d,s)],
#                                      d)
#                 plt.close(h)

# multiple comparisons

import matplotlib.pyplot as plt
# metrics
fair_metrics = {
    'auditor_fp_violation':'Audit FP Violation $\gamma$',
#     'auditor_fn_violation':'Audit FN Violation $\gamma$',
#     'mean_subgroup_unfairness':'Mean Subgroup Unfairness',
#     'max_subgroup_unfairness':'Max Subgroup Unfairness',
#     'max_marginal_unfairness':'Max Marginal Unfairness',
#     'mean_marginal_unfairness':'Mean Marginal Unfairness',
    }
loss_metrics = {
#     'fpr':'False Positive Rate',
#    'accuracy':'1-Accuracy',
#     'logloss':'Log Loss',
#     'mae':'Mean Absolute Error',
#     'precision':'1-Precision',
#     'recall':'1-Recall',
     'ave_precision_score':'1-Average Precision Score',
#     'auc_prc':'1 - Area Under Precision-Recall Curve'
    }
for d in datasets:
    for s in seeds:
        keepgoing = True
        for m in methods:
            if perf(m,d,s) not in performances.keys():
                keepgoing=False
        if keepgoing:
            h = pareto_multicompare_plots(
                    [performances[perf(m,d,s)] for m in methods], 
                    d,fair_metrics, loss_metrics, rdir)
            plt.close(h)

