from flask import Flask, appcontext_popped, flash, render_template, request
import pickle
import numpy as np
import pandas as pd

model = pickle.load(open('model.pkl','rb'))
app = Flask(__name__)
@app.route('/')
def heart():
    return render_template('heart.html')

@app.route('/predict', methods=['POST'])
def predict():
    data1 = request.form['a']
    data2 = request.form['b']
    data3 = request.form['c']
    data4 = request.form['d']
    data5 = request.form['e']
    data6 = request.form['f']
    data7 = request.form['g']
    data8 = request.form['h']
    data9 = request.form['i']
    data10 = request.form['j']
    data11 = request.form['k']
    data12 = request.form['l']
    data13 = request.form['m']
    prediction = model.predict([[data1,data2,data3,data4,data5,data6,data7,data8,data9,data10,data11,data12,data13]])
    p=0
    if prediction==0:
        p=1
        return render_template('heart.html', prediction_text='The patient has no heart disease')
    else:
        p=1
        return render_template('heart.html',p=p, prediction_text='The patient is having heart disease')
if __name__ == '__main__':
    app.run(debug=True)