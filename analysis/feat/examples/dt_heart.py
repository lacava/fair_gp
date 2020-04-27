from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from metrics import balanced_accuracy_score
import pandas as pd

dataset = pd.read_csv('d_heart.csv')
X = StandardScaler().fit_transform(dataset.drop('class',axis=1))
y = dataset['class']

X_t,X_v,y_t,y_v = train_test_split(X,y,test_size=0.25,shuffle=False)

clf = DecisionTreeClassifier(max_depth=4,criterion='gini').fit(X_t, y_t)

print('train score:',balanced_accuracy_score(y_t, clf.predict(X_t)))
print('test score:',balanced_accuracy_score(y_v,clf.predict(X_v)))
print('feature importances:', clf.feature_importances_)
import numpy as np
print('argsort: ',np.argsort(clf.feature_importances_))
