
from sklearn import datasets
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score



data = datasets.load_breast_cancer()
print (data.feature_names)
print (data.target_names)

df = pd.read_csv('data.csv', sep = ',')


X = data.data[:, [3,11]]
y = data.target
np.unique(y)

X_train, X_test, y_train, y_test  = train_test_split(X, y, test_size=0.3, random_state=0 )
print(len(X_train))
print(len(X_test))
sc = StandardScaler()
sc.fit(X_train)
StandardScaler(copy= True,with_mean= True, with_std= True )
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

ppn = Perceptron(max_iter=40, eta0=0.1, random_state=0)
ppn.fit(X_train_std, y_train)

y_pred = ppn.predict(X_test_std)
print('Misclassified samples: %d'% (y_test!=y_pred).sum())

print('Accuracy : %.2f' %accuracy_score(y_test, y_pred))


