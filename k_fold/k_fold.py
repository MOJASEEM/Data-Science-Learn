from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import pandas as pd

iris=load_iris()
df=pd.DataFrame(data=iris.data,columns=iris.feature_names)
df['target']=iris.target
x=df.drop('target',axis='columns')
y=df['target']
# stratified k fold
skf=StratifiedKFold(n_splits=5)
lr=cross_val_score(LogisticRegression(max_iter=200),x,y,cv=skf)
svm=cross_val_score(SVC(max_iter=200),x,y,cv=skf)
rf=cross_val_score(RandomForestClassifier(),x,y,cv=skf)
print("Logistic Regression scores for each fold:", lr)
print("SVM scores for each fold:", svm)
print("Random Forest scores for each fold:", rf)
print("Average Logistic Regression score:", np.mean(lr))
print("Average SVM score:", np.mean(svm))
print("Average Random Forest score:", np.mean(rf))

