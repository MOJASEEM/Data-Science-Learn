from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import numpy as np

iris=load_iris()
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2)
print(iris.target_names)
model=LogisticRegression()
model.fit(X_train,y_train)
y_pred=model.predict(X_test)
mat=iris.target_names[y_pred]
print(mat)
values, counts = np.unique(mat, return_counts=True)
fl = values[np.argmax(counts)]
print("The most frequent predicted flower is:", fl)