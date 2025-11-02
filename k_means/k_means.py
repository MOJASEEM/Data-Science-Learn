from sklearn.cluster import KMeans
from sklearn.datasets import load_iris
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

iris = load_iris()
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
scaler = MinMaxScaler()
scaler.fit(df[['sepal length (cm)']])
df['sepal_length'] = scaler.transform(df[['sepal length (cm)']])
scaler.fit(df[['petal length (cm)']])
df['petal_length'] = scaler.transform(df[['petal length (cm)']])
print(df.head)
rg=range(1, 11)
sse = []
for k in rg:
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(df[['sepal_length', 'petal_length']])
    sse.append(kmeans.inertia_)
plt.plot(rg, sse)
plt.xlabel('K')
plt.ylabel('Sum of squared error')
plt.title('Elbow Method For Optimal K')
plt.show()
kmeans = KMeans(n_clusters=3)
y_predicted=kmeans.fit_predict(df[['sepal_length', 'petal_length']])
df['cluster']=y_predicted
print(df.head)
plt.scatter(df['sepal_length'], df['petal_length'], c=df['cluster'], cmap='rainbow')
centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='black', marker='X', s=200)
plt.show()
