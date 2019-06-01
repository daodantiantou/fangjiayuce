import pandas as pd
data=pd.read_csv('./logistic_dat.csv')
data1=data.dropna()
data2=pd.get_dummies(data1.Gender)
data3=data2.drop('Female',axis=1)
data4=pd.concat([data3,data1.drop(['User ID','Gender'],axis=1)],axis=1)
# print(data4)
x=data4.iloc[:,:-1]
y=data4.iloc[:,-1:]
# y=data4[['Purchased']]
print(y)
from sklearn import linear_model
model2=linear_model.LogisticRegression(max_iter=100000,tol=0.00001,solver='liblinear').fit(x,y)
res=model2.predict([[1,18,100]])
print(res)