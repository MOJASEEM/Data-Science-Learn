import pandas as pd
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np
# Load dataset
df=pd.read_csv('D:/Data Science/Logistic_reg/HR_comma_sep.csv')
# Select features and target variable
left=df[df.left==1]
retained=df[df.left==0]
pd.crosstab(df.salary,df.left).plot(kind='bar')
plt.title('Employee Retention by Salary Level')
plt.xlabel('Salary Level')
plt.ylabel('Number of Employees')
plt.show()
pd.crosstab(df.Department,df.left).plot(kind='bar')
plt.title('Employee Retention by Department')
plt.xlabel('Department')
plt.ylabel('Number of Employees')
plt.show()
subdf = df[['satisfaction_level','average_montly_hours','promotion_last_5years','salary']]
salary_dummies = pd.get_dummies(subdf.salary, prefix="salary")
df_with_dummies = pd.concat([subdf,salary_dummies],axis='columns')
df_with_dummies.drop('salary',axis='columns',inplace=True)
X = df_with_dummies
y = df.left
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
score = model.score(X_test, y_test)
print("Score",score)

