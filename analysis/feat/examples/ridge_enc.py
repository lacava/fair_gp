from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error as mse

import pandas as pd

dataset = pd.read_csv('d_enc.csv')
X = StandardScaler().fit_transform(dataset.drop('label',axis=1))
y = dataset['label']

X_t,X_v,y_t,y_v = train_test_split(X,y,test_size=0.25,shuffle=False)

clf = Ridge().fit(X_t,y_t)

print('train score:',mse(y, clf.predict(X)))
print('test score:',mse(y_v, clf.predict(X_v)))
print('coefficients:',clf.coef_)
