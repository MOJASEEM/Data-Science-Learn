from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
import pandas as pd
import numpy as np

# Load the Titanic dataset
titanic=pd.read_csv('D:/Data Science/naive_bayers/alive_or_not/titanic.csv')
titanic.drop(['PassengerId','Name','SibSp','Parch','Ticket','Cabin','Embarked',], axis=1, inplace=True)
target=titanic['Survived']
inputs=titanic.drop(['Survived'], axis=1)
print(titanic.head())
dummies=pd.get_dummies(inputs.Sex)
inputs=pd.concat([inputs,dummies], axis=1)
inputs.drop(['Sex'], axis=1, inplace=True)
print(inputs.head())
na_values=inputs.columns[inputs.isna().any()]
print("Columns with NaN values:", na_values.tolist())
inputs['Age'].fillna(inputs['Age'].mean(), inplace=True)
x_train, x_test, y_train, y_test = train_test_split(inputs, target, test_size=0.2, random_state=42)
model = GaussianNB()
model.fit(x_train, y_train)
y_pred = model.predict(x_test)
print("Predicted values:", y_pred)
accuracy = model.score(x_test, y_test)
print("Model accuracy:", accuracy)
