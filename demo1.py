import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
data=pd.read_csv('./house_price1.csv')
print(data)
x=data[['area']]
y=data.iloc[:,-1:]
fig=plt.figure()
plt.scatter(x,y)
from sklearn import linear_model
model=linear_model.LinearRegression().fit(x,y)
a=model.coef_
b=model.intercept_
x1=np.arange(10,70,1)
y1=a[0]*x1+b
plt.plot(x1,y1)
plt.show()

