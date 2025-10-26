import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

company = ['Apple', 'Google', 'Microsoft', 'Amazon', 'Facebook']
revenue = [260, 161, 125, 280, 70]
profit = [55, 34, 39, 11, 18]
data = pd.DataFrame({'Company': company, 'Revenue': revenue, 'Profit': profit})
xpos = np.arange(len(data['Company']))
# Bar chart for Revenue
plt.bar(xpos-0.2, data['Revenue'],width=0.4, color='skyblue')
plt.bar(xpos+0.2, data['Profit'],width=0.4, color='Red')
plt.title('Company Revenue, Profit')
plt.xlabel('Company')
plt.ylabel('Revenue (in billions)')
plt.legend(['Revenue', 'Profit'])
plt.show()