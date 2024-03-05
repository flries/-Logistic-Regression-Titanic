import pandas as pd

import sys
import io
import matplotlib.pyplot as plt
import numpy as np

train = pd.read_csv("input/train.csv")
test = pd.read_csv("input/test.csv")

def data_process(titanic):
    # Embarked_C = 0 not embarked in Cherbourg, 1 = embarked in Cherbourg.
    ports = pd.get_dummies(titanic.Embarked , prefix='Embarked')
    
    titanic = titanic.join(ports)
    # drop the original column
    titanic.drop(['Embarked'], axis=1, inplace=True) 

    titanic.Sex = titanic.Sex.map({'male':0, 'female':1})

    # copy y column values out
    y = titanic.Survived.copy()
    # then, drop y column
    X = titanic.drop(['Survived'], axis=1) 

    X.drop(['PassengerId', 'Name', 'Ticket', 'Cabin', 'Fare'], axis=1, inplace=True)

    # replace NaN with average age
    X.Age.fillna(X.Age.mean(), inplace=True)

    return X,y

X_train, y_train = data_process(train)
X_test, y_test = data_process(test)

from sklearn.linear_model import LogisticRegression

model = LogisticRegression(solver='liblinear', max_iter=300)
model.fit(X_train, y_train)

print("train Accuracy:", model.score(X_train, y_train))
print("test  Accuracy:", model.score(X_test, y_test))

print(model.intercept_) # the fitted intercept
print(model.coef_)  # the fitted coefficients
