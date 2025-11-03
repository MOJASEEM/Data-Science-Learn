from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.datasets import load_wine
import pandas as pd
import numpy as np

# Load the Wine dataset
wine = load_wine()
df = pd.DataFrame(data=wine.data, columns=wine.feature_names)
data = df.values
target = wine.target
x_train, x_test, y_train, y_test = train_test_split(data, target, test_size=0.2, random_state=42)
clf = Pipeline([
    ('nb', MultinomialNB())
])
clf.fit(x_train, y_train)
y_pred = clf.predict(x_test)
print("Predicted values using multinomialdb:", y_pred)
accuracy = clf.score(x_test, y_test)
print("Model accuracy for multinomialdb:", accuracy)
ylf=Pipeline([
    ('gnb', GaussianNB())
])
ylf.fit(x_train, y_train)
y_pred_gnb = ylf.predict(x_test)
print("Predicted values using GaussianNB:", y_pred_gnb)
accuracy_gnb = ylf.score(x_test, y_test)
print("Model accuracy for GaussianNB:", accuracy_gnb)
