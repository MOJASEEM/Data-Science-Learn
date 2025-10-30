from sklearn.preprocessing import LabelEncoder
from sklearn import tree
import pandas as pd

data = pd.read_csv('D:/Data Science/Decision_tree/titanic_survive/titanic.csv')
inputs = data.drop(['Survived','PassengerId','Name', 'SibSp','Parch','Ticket','Cabin','Embarked'],axis=1)
target = data['Survived']
le_pclass = LabelEncoder()
le_sex = LabelEncoder()
le_age = LabelEncoder()
le_fare = LabelEncoder()
inputs['Pclass_n'] = le_pclass.fit_transform(inputs['Pclass'])
inputs['Sex_n'] = le_sex.fit_transform(inputs['Sex'])
inputs['Age_n'] = le_age.fit_transform(inputs['Age'])
inputs['Fare_n'] = le_fare.fit_transform(inputs['Fare'])
inputs_n = inputs.drop(['Pclass', 'Sex', 'Age','Fare'], axis=1)
model = tree.DecisionTreeClassifier()
model = model.fit(inputs_n, target)
ip=pd.DataFrame([[3, 1, 22.0, 7.25]],columns=['Pclass_n', 'Sex_n', 'Age_n', 'Fare_n'])
op=model.predict(ip)  # Example input: [Pclass_n, Sex_n, Age_n, Fare_n]
if op[0]:
    print("Survived")
else:
    print("Not Survived")
