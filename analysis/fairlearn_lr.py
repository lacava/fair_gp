import sys
from train_models import train_fairlearn_model
import json
from fairlearn.reductions import GridSearch, GroupLossMoment, ZeroOneLoss
from sklearn.linear_model import LogisticRegression

dataset = sys.argv[1]
dataset_name = dataset.split('/')[-1].split('.')[0]
attributes = sys.argv[2]
seed = int(sys.argv[3])
rdir = sys.argv[4]

# set up gridsearch model
model = LogisticRegression()
sweep = GridSearch(model,
                   constraints=EqualizedOdds(),
                   grid_size=100,
                   grid_limit=2)

sweep.fit(df_train_balanced, Y_train_balanced, sensitive_features=A_train_balanced)

# train
# perf, hv = train_gerryfair_model(fair_model, 'gerryfair', dataset, 
#                     attributes, seed, rdir)
