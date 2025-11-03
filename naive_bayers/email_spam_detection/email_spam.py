from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

# Load the email spam dataset
df = pd.read_csv('D:/Data Science/naive_bayers/email_spam_detection/spam.csv')
# Preprocess the dataset
df['spam'] = df['Category'].apply(lambda x: 1 if x == 'spam' else 0)
target = df['spam']
inputs = df['Message']
# Convert text data to numerical data using Bag of Words
v=CountVectorizer()
inputs = v.fit_transform(inputs).toarray()
x_train, x_test, y_train, y_test = train_test_split(inputs, target, test_size=0.2, random_state=42)
model = MultinomialNB()
model.fit(x_train, y_train)
y_pred = model.predict(x_test)
print("Predicted values:", y_pred)
accuracy = model.score(x_test, y_test)
print("Model accuracy:", accuracy)
