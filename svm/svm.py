from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

# Load the digits dataset
digits = load_digits()
df=pd.DataFrame(data=digits.data, columns=digits.feature_names)
df['target'] = digits.target

df0=df[df.target==0]
df1=df[df.target==1]
df2=df[df.target==2]
df3=df[df.target==3]
plt.scatter(df0['pixel_0_0'],df0[ 'pixel_0_1'],color='red',marker='+')
plt.scatter(df1['pixel_0_0'],df1[ 'pixel_0_1'],color='green',marker='.')
plt.scatter(df2['pixel_0_0'],df2[ 'pixel_0_1'],color='blue',marker='o')
x=df.drop('target',axis='columns')
y=df['target']
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)
model=SVC()
model.fit(x_train,y_train)
op=model.predict(x_test)
print("Predicted values:", op)
print(model.score(x_test,y_test))

