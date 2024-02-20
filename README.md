# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the standard Libraries.
2.Set variables for assigning dataset values.
3.Import linear regression from sklearn.
4.Assign the points for representing in the graph.
5.Predict the regression for marks by using the representation of the graph.
6.Compare the graphs and hence we obtained the linear regression for the given datas.

## Program:
```

Program to implement the simple linear regression model for predicting the marks scored.
Developed by: masina sree karsh
RegisterNumber: 212223100033

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error
df=pd.read_csv('student_scores.csv')
print(df)
df.head(0)
df.tail(0)
print(df.head())
print(df.tail())
x = df.iloc[:,:-1].values
print(x)
y = df.iloc[:,1].values
print(y)
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)
y_pred = regressor.predict(x_test)
print(y_pred)
print(y_test)
#Graph plot for training data
plt.scatter(x_train,y_train,color='black')
plt.plot(x_train,regressor.predict(x_train),color='blue')
plt.title("Hours vs Scores(Training set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
#Graph plot for test data
plt.scatter(x_test,y_test,color='black')
plt.plot(x_train,regressor.predict(x_train),color='red')
plt.title("Hours vs Scores(Testing set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
mse=mean_absolute_error(y_test,y_pred)
print('MSE = ',mse)
mae=mean_absolute_error(y_test,y_pred)
print('MAE = ',mae)
rmse=np.sqrt(mse)
print("RMSE= ",rmse)
```

## Output:

### Dataset:


![image1 0](https://github.com/sreekarsh/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/139841918/ab94eec6-00df-4759-bfbd-04c5549f41c2)


### Head Values:


![head](https://github.com/sreekarsh/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/139841918/dd4ec90a-4a5b-4c44-a013-6e0d51be8606)


### Tail Values:


![tail](https://github.com/sreekarsh/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/139841918/88243129-c997-4729-a61a-85b2370b6a71)


### X and Y values:


![xyvalues](https://github.com/sreekarsh/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/139841918/5222cbe1-9f82-474b-b70d-b9ff1e4f0143)


## Predication values of X and Y


![predict ](https://github.com/sreekarsh/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/139841918/23f11abd-24d5-4824-8354-699e83fe87a5)


### MSE,MAE and RMSE:


![values](https://github.com/sreekarsh/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/139841918/503a7618-d705-454c-abf2-ee1d65b2ebb0)


### Training Set:

![Screenshot 2024-02-20 112557](https://github.com/sreekarsh/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/139841918/06c13bb9-2a1e-483e-b9a3-6b3ad0c0d9aa)


### Testing Set:

![Screenshot 2024-02-20 112731](https://github.com/sreekarsh/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/139841918/67dcd107-9435-4af8-9054-01893afb8706)



## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
