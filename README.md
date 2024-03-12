
## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. 
2. 
3. 
4. 

## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by:piyush kumar 
RegisterNumber:212223220075

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
df=pd.read_csv("C:/Users/SEC/Downloads/student_scores.csv")
df.head()
df.tail()
X=df.iloc[:,:-1].values
print(X)
Y=df.iloc[:,-1].values
print(Y)
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=1/3,random_state=0)

from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,Y_train)
Y_pred=regressor.predict(X_test)

print(Y_pred)
print(Y_test)
plt.scatter(X_train,Y_train,color="orange")
plt.plot(X_train,regressor.predict(X_train),color="red")
plt.title("Hours vs scores(Training Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
plt.scatter(X_test,Y_test,color="orange")
plt.plot(X_test,regressor.predict(X_test),color="red")
plt.title("Hours vs scores(Test Data Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
mse=mean_squared_error(Y_test,Y_pred)
print("MSE = ",mse)
mae=mean_absolute_error(Y_test,Y_pred)
print("MAE = ",mae)
rmse=np.sqrt(mse)
print("RMSE : ",rmse)
*/
```

## Output:


![WhatsApp Image 2024-03-05 at 08 55 10_7b46d23e](https://github.com/H515piyush/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/147472999/cc4d9bb8-1bc0-4238-a934-443c02a29b5b)
![WhatsApp Image 2024-03-05 at 08 55 09_6a094297](https://github.com/H515piyush/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/147472999/c10e760a-4b53-4ea4-bf8e-e9479737ec98)
![WhatsApp Image 2024-03-05 at 08 55 09_6a094297](https://github.com/H515piyush/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/147472999/ff0d19f9-8918-4557-b2eb-84fcd53765d1)



![WhatsApp Image 2024-03-05 at 08 55 09_9ebf8f4b](https://github.com/H515piyush/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/147472999/fed66777-97fb-4b91-adf3-37908d152894)

## Result:

Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
