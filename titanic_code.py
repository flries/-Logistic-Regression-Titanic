import math
import pandas as pd
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

    # %% normalization
    X = (X -np.min(X))/(np.max(X)-np.min(X)).values

    return X,y

# beta, eta = initialize_parameters(len(X.columns))
def initialize_parameters(column):
    beta = [1] * column
    learning_rate = 0.005
    return beta,learning_rate

# loss_function(X_train,y_train,beta)
def loss_function(X, Y, beta):
    label = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Embarked_C', 'Embarked_Q', 'Embarked_S']
    loss = 0
    for i,data in X.iterrows():# i from 0 to n=len(row) = 891
        dot = 0
        # caculate the function of exponential
        for j in range(len(beta)):# j from 0 to k=len(column) = 8
            dot += data[label[j]]*beta[j]

        #fun = Y[i]*math.log(1/(1+math.exp(-dot))) + (1-Y[i])*(1-math.log(1/(1+math.exp(-dot))))
        fun = (Y[i]-1)*dot - math.log(1+math.exp(-dot))
        
        loss += fun
    return -loss

# list = gradient_descent_algorithm(X_train,y_train,beta,eta)
def gradient_descent_algorithm(X, Y, beta, eta):
    set_num = 0.0001
    gradient = list()
    for i in range(len(beta)):# j from 0 to k=len(column) = 8
        beta_k = beta.copy()
        beta_k[i] = beta[i] + set_num
        
        fun = (loss_function(X,Y,beta_k) - loss_function(X,Y,beta)) / set_num
        
        gradient.append(fun)
        
    return gradient
    
''' ---------- DATA PROCESS ---------- '''
X_train, y_train = data_process(train)
X_test, y_test = data_process(test)

# Step 1. Initialize parameter vector(beta) and learning rate(eta)
beta, eta = initialize_parameters(len(X_train.columns))

def Logistic_Regression(X, Y, beta, eta):
    # Step 2. Calculate the Loss(error) function:
    loss = loss_function(X,Y,beta)

    # Step 3. Calculate the gradient of loss function:
    gradient = gradient_descent_algorithm(X_train,y_train,beta,eta)

    # Step 4. Update parameter vector(beta)
    new_beta = beta.copy()
    for j in range(len(beta)):# j from 0 to k=len(column) = 8
        new_beta[j] = beta[j] - eta*gradient[j]

    return loss, new_beta

# prediction
def predict_function(X, beta):
    label = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Embarked_C', 'Embarked_Q', 'Embarked_S']

    y = [0] * len(X)
    for i,data in X.iterrows():# i from 0 to n=len(row) = 891
        dot = 0
        for j in range(len(beta)):# j from 0 to k=len(column) = 8
            dot += data[label[j]]*beta[j]

        fun = 1/(1+math.exp(-dot))
        if fun >= 0.5:
            y[i] = 1
    return y

''' ----------   LEARNING   ---------- '''
loss_data = list()

num_iterations = 0
interval_error = 0.0001
while True:
    num_iterations += 1

    # repeat the Logistic Regression process
    loss, new_beta = Logistic_Regression(X_train, y_train, beta, eta)
    
    if num_iterations == 1 or num_iterations % 50 == 0:
        print('epoch '+ str(num_iterations) +' loss: '+ str(loss))
        loss_data.append(loss)
    
    flag = [0] * len(beta)
    for i in range(len(beta)):
        interval = abs(new_beta[i]-beta[i])
        if interval > interval_error:
            flag[i] = 0
        else:
            flag[i] = interval
    
    if 0 not in flag:
        print('epoch '+ str(num_iterations) +' loss: '+ str(loss))
        loss_data.append(loss)
        
        beta = new_beta
        break
    beta = new_beta

''' ----------   PREDICT   ---------- '''
y_pre = predict_function(X_train, beta)
print("train accuracy: {} %".format(100-np.mean(np.abs(y_pre-y_train))*100))

y_pre = predict_function(X_test, beta)
print("test  accuracy: {} %".format(100-np.mean(np.abs(y_pre-y_test ))*100))

loss_size = list(range(0,num_iterations,50))
loss_size.append(num_iterations)

plt.plot(loss_size, loss_data)
plt.ylabel('loss')
plt.xlabel('Epochs')
plt.show()
