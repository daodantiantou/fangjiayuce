from flask import Flask,render_template,request

app=Flask(__name__)

@app.route('/',methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/',methods=['POST'])
def form():
    arr=[0,0,0,0,0,0,0,0,0,0,0,0]
    dist=request.form['dist']
    arr[int(dist)]=1
    print(arr)
    return render_template('index.html')

app.run()