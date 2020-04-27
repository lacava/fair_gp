from FeatFair import FeatFair
from clean import clean_dataset
import numpy as np

dataset = "GerryFair/dataset/communities.csv"
attributes = "GerryFair/dataset/communities_protected.csv"

centered = True
X, X_prime, y = clean_dataset(dataset, attributes, centered)
X_train = np.hstack((X, X_prime))
protected_groups = ([False for f in np.arange(X.shape[1])] + 
                   [True for f in np.arange(X_prime.shape[1])])

# print('fair_feat.py protected groups:',protected_groups)
feature_names = list(X.columns) + list(X_prime.columns)
# print('feature names:', feature_names)
est = FeatFair(
        classification = True,
        # scorer='fpr', 
        scorer='zero_one',
        sel = 'fair_lexicase',
        # surv = 'offspring',
        surv = 'nsga2',
        obj='fairness,fitness',
        protected_groups=protected_groups,
        feature_names=','.join(feature_names),
        verbosity=2,
        gens=1000,
        pop_size=1000,
        )
print('est protected_groups: ', est.protected_groups)
est.fit(X_train, y)

