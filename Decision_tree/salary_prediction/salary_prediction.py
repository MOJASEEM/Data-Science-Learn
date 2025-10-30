from sklearn.preprocessing import LabelEncoder
from sklearn import tree
import pandas as pd

data = pd.read_csv('D:/Data Science/Decision_tree/salary_prediction/salaries.csv')
inputs = data.drop('salary_more_then_100k', axis='columns')
target = data['salary_more_then_100k']
le_company = LabelEncoder()
le_job = LabelEncoder()
le_degree = LabelEncoder()
inputs['company_n'] = le_company.fit_transform(inputs['company'])
inputs['job_n'] = le_job.fit_transform(inputs['job'])
inputs['degree_n'] = le_degree.fit_transform(inputs['degree'])
inputs_n = inputs.drop(['company', 'job', 'degree'], axis='columns')
print(inputs_n.head())
model = tree.DecisionTreeClassifier()
model.fit(inputs_n, target)
predictions = model.predict([[2,1,0]])  # Example input: [company_n, job_n, degree_n]
if predictions[0]:
    print("Salary more than 100k")
else:
    print("Salary less than 100k")
score=model.score(inputs_n, target)
print("Accuracy:", score)