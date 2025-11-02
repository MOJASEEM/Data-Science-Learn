from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
import pandas as pd
import seaborn as sns
# Load the iris dataset
iris = load_iris()
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)

plt.gray()
for i in range(4):
    plt.matshow(iris.data[i].reshape(2, 2))
    plt.title(f'Sample {i+1}')
plt.show()
df['target'] = iris.target
print(df.head())
x_train, x_test, y_train, y_test = train_test_split(df.drop('target', axis='columns'), df['target'], test_size=0.2)
model = RandomForestClassifier(n_estimators=10) 
model.fit(x_train, y_train)
y_predicted = model.predict(x_test)
print("Predicted values:", y_predicted)
cm=confusion_matrix(y_test, y_predicted)
plt.figure(figsize=(10,7))
sns.heatmap(cm, annot=True)
plt.xlabel('Predicted')
plt.ylabel('Truth')
plt.show()
print(classification_report(y_test, y_predicted))
print("Predicted values:", y_predicted)   