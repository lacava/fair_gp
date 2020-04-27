import sys
import gerryfair
from train_models import train_gerryfair_model
import json
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.kernel_ridge import KernelRidge

dataset = sys.argv[1]
dataset_name = dataset.split('/')[-1].split('.')[0]
attributes = sys.argv[2]
seed = int(sys.argv[3])
rdir = sys.argv[4]

# set up Gerry Fair model
fair_model = gerryfair.model.Model(C=15, 
                    printflag=False, 
                    fairness_def='FP', 
                    max_iters=100,
                    predictor=KernelRidge())
# train
perf, hv = train_gerryfair_model(fair_model, 'gerryfair_kernel', dataset, 
                    attributes, seed, rdir)
