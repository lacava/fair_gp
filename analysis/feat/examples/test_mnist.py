import pandas as pd
from pmlb import fetch_data

df = pd.read_csv('mnist.csv',sep='\t')
print(df.columns)
X = df.drop('class',axis=1).values
y = df['class'].values

from feat import Feat

ft = Feat(classification=True,verbosity=2)

ft.fit(X[:60000],y[:60000])

print(ft.score(X[60000:],y[60000:]))
