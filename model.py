import numpy as np
import pandas as pd
import pickle

df = pd.read_csv('students_placement.csv')

print(df.head())

X = df.drop(columns=['placed'])
y = df['placed']

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)

from sklearn.metrics import accuracy_score

from sklearn.svm import SVC

# accuracy_score(y_test,SVC(kernel='rbf').fit(X_train,y_train).predict(X_test))

svc = SVC(kernel='rbf')
a = svc.fit(X_train, y_train)

pickle.dump(a, open('model.pkl', 'wb'))
