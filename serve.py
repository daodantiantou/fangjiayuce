from flask import Flask,render_template,request,redirect

app=Flask(__name__)

import numpy as np
import pandas as pd
data=pd.read_csv('./house_price.csv')
data1=data.dropna()
data2=pd.get_dummies(data1[['dist','floor']])
pd.set_option('display.max_columns',None)
data3=data2.drop(['dist_shijingshan','floor_high'],axis=1)
data4=pd.concat([data3,data1[['roomnum','halls','AREA','subway','school','price']]],axis=1)
x=data4.iloc[:,:-1]
y=data4.iloc[:,-1:]
from sklearn import linear_model
from sklearn.model_selection import train_test_split
x_train,x_text,y_train,y_text=train_test_split(x,y,test_size=0.3,random_state=42)
model=linear_model.LinearRegression().fit(x_train,y_train)

data=pd.read_csv('./logistic_dat.csv')
data1=data.dropna()
data2=pd.get_dummies(data1.Gender)
data3=data2.drop('Female',axis=1)
data4=pd.concat([data3,data1.drop(['User ID','Gender'],axis=1)],axis=1)
# print(data4)
x=data4.iloc[:,:-1]
y=data4.iloc[:,-1:]
# y=data4[['Purchased']]
# print(y)
from sklearn import linear_model
model2=linear_model.LogisticRegression(max_iter=100000,tol=0.00001,solver='liblinear').fit(x,y)
@app.route('/')
def login():
    return render_template('login.html')

@app.route('/chacklogin',methods=['POST'])
def chacklogin():
    sex=int(request.form['sex'])
    age=int(request.form['age'])
    salary=int(request.form['salary'])
    res=model2.predict([[sex,age,salary]])
    print(res)
    return redirect('/')

# @app.route('/',methods=['GET'])
# def index():
#     return render_template('index.html')
#
# @app.route('/',methods=['POST'])
# def form():
#     arr=[0,0,0,0,0,0,0,0,0,0,0,0]
#     dist=request.form['dist']
#     if dist==-1:
#         pass
#     else:
#         arr[int(dist)]=1
#     floor = request.form['floor']
#     if floor==-1:
#         pass
#     else:
#         arr[int(floor)]=1
#     roomnum = request.form['roomnum']
#     arr[7]=int(roomnum)
#     halls = request.form['halls']
#     arr[8] = int(halls)
#     area = request.form['area']
#     if area:
#         arr[9] = int(area)
#     else:
#         area[9]=int(0)
#     subway = request.form['subway']
#     arr[10] = int(subway)
#     school = request.form['school']
#     arr[11] = int(school)
#     res=model.predict([arr])[0][0]
#     res=round(res,2)
#     return render_template('index.html',data={'res':res})

app.run()