import pandas as pd

import numpy as np

from feat import Feat
import sys
seed = sys.argv[1]

df = pd.read_csv('../examples/d_heart.csv', sep=',')
df.describe()
X = df.drop('class',axis=1).values
y = df['class'].values
clf = Feat(max_depth=3,
        max_dim=1,
        gens=100,
        pop_size=200,
        verbosity=2,
        shuffle=True,
        classification=True,
        functions="+,-,*,/,exp,log,and,or,not,=,<,>,ite",
        random_state=seed,
        softmax_norm=True)
clf.fit(X,y)
