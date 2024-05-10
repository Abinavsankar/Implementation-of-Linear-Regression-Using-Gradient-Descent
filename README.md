# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
Step 1: Import the standard python libraries for Gradient design.

Step 2: Introduce the variables needed to execute the function.

Step 3: Use function for the representation of the graph.

Step 4: Using for loop apply the concept using the formulae.

Step 5: Execute the program and plot the graph.

Step 6: Predict and execute the values for the given conditions.

## Program:
```
/*
Program to implement the linear regression using gradient descent.
Developed by: Abinav Sankar S
RegisterNumber: 212222040002
*/

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
def linear_regression(X1,y,learning_rate=0.1,num_iters=1000):
X=np.c_[np.ones(len(X1)),X1]
theta=np.zeros(X.shape[1]).reshape(-1,1)
for _ in range(num_iters):
#calculate predictions
predictions=(X).dot(theta).reshape(-1,1)
#calculate errors
errors=(predictions-y).reshape(-1,1)
#Update theta using gradient descent
theta-=learning_rate*(1/len(X1))*X.T.dot(errors)
return theta
data=pd.read_csv("/content/50_Startups.csv")
data.head()

x=(data.iloc[1:,:-2].values)
x1=x.astype(float)
scaler=StandardScaler()
y=(data.iloc[1:,-1].values).reshape(-1,1)
x1_scaled=scaler.fit_transform(x1)
y1_scaled=scaler.fit_transform(y)
print(x)
print(x1_scaled)
```



## Output:
![image](https://github.com/Abinavsankar/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/119103734/cc8cbf36-cba7-4f54-a75c-c13735389598)



## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
