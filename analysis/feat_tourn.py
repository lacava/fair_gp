import sys
import json
from FeatFair import FeatFair
from train_models import train_feat_model
from feat_common_args import common_args

dataset = sys.argv[1]
dataset_name = dataset.split('/')[-1].split('.')[0]
attributes = sys.argv[2]
seed = int(sys.argv[3])
rdir = sys.argv[4]

common_args['obj'] = 'fitness'
est = FeatFair(
            **common_args,
            sel='tournament',
            surv='offspring',
            random_state=seed,
            )

# set up Feat model
perf, hv = train_feat_model(est, 'feat_tourn', dataset, attributes, seed, rdir)

