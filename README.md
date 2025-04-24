# Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee

## AIM:
To write a program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the libraries and read the data frame using pandas.
2. Calculate the null values present in the dataset and apply label encoder.
3. Determine test and training data set and apply decison tree regression in dataset.
4. Calculate Mean square error,data prediction and r2.

## Program:
```
/*
Program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.
Developed by: MUGIL RAJ S A
RegisterNumber: 212223220062 
*/
```
```
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn import metrics
import numpy as np

data = pd.read_csv("Salary.csv")
print(data.head())
print(data.info())
print(data.isnull().sum())

le = LabelEncoder()
data["Position"] = le.fit_transform(data["Position"])
print(data.head())

x = data[["Position", "Level"]]
y = data["Salary"]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=2)

dt = DecisionTreeRegressor()
dt.fit(x_train, y_train)

y_pred = dt.predict(x_test)
print("Predicted:", y_pred)

r2 = metrics.r2_score(y_test, y_pred)
print("R2 Score:", r2)

print("New prediction:", dt.predict(np.array([[5, 6]])))
## Output:
![Decision Tree Regressor Model for Predicting the Salary of the Employee](sam.png)
```
## Output:
![image](https://github.com/user-attachments/assets/bd532e21-06d1-40b1-ae0f-3250cdb75aa6)
![image](https://github.com/user-attachments/assets/89a5b161-2298-4d31-a1eb-d797190b66e5)
![image](https://github.com/user-attachments/assets/66e572af-ffb5-476d-bb62-bfda8a0ac1d5)
## Result
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.
