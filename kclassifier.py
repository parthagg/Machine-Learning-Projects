import itertools
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
import pandas as pd
import matplotlib.ticker as ticker
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.model_selection import train_test_split

df=pd.read_csv("iris.data")
print(df.head())
print(df["Species"].value_counts())
X=df[['Sepal.Length','Sepal.Width','Petal.Length','Petal.Width']].values
print(X[0:5])
y=df['Species'].values
print(y[0:5])
X_train,X_test,y_train,y_test= train_test_split(X,y,test_size=0.1, random_state=4)
print('Train set:',X_train.shape,y_train.shape)
print(y_test)
k=10
neigh=KNeighborsClassifier(n_neighbors=k).fit(X_train,y_train)
print(neigh)
y_hat=neigh.predict(X_test)
print(y_hat[0:5])
print(y_test[0:5])
print("Train set Accuracy: ", metrics.accuracy_score(y_train, neigh.predict(X_train)))
print("Test set Accuracy: ", metrics.accuracy_score(y_test, y_hat))
custom_sample=[[5.0,3.5,1.0,0.2]]
result=neigh.predict(custom_sample)
print(result)