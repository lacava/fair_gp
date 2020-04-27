import sys
import json
from train_models import train_feat_model
from feat_common_args import common_args
from FeatFair import FeatFair

dataset = sys.argv[1]
dataset_name = dataset.split('/')[-1].split('.')[0]
attributes = sys.argv[2]
seed = int(sys.argv[3])
rdir = sys.argv[4]

est = FeatFair(
            **common_args,
            sel = 'nsga2',
            surv = 'nsga2',
            random_state=seed,
            )

# set up Feat NSGA2 model
perf, hv = train_feat_model(est, 'feat_nsga2', dataset, attributes, seed,
        rdir)

