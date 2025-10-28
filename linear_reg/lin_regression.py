import matplotlib.pyplot as plt
import pandas as pd
import numpy as np 
from sklearn.linear_model import LinearRegression

df=pd.read_csv('D:Data Science/homeprices.csv')
plt.xlabel('Area in sq ft')
plt.ylabel('Price in USD')
plt.scatter(df.area,df.price,color='red',marker='+')
reg=LinearRegression()
reg.fit(df[['area']],df.price)
predicted_price = reg.predict(pd.DataFrame([[3300]], columns=['area']))
print(predicted_price )
d=pd.read_csv('D:Data Science/areas.csv')
p=reg.predict(d)
d['price']=p
d.to_csv('D:Data Science/predicted_prices.csv',index=False)
plt.plot(df.area,reg.predict(df[['area']]),color='blue')
plt.show()