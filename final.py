from flask import Flask, appcontext_popped, render_template, request
import pickle
import numpy as np
import pandas as pd

model = pickle.load(open('model.pkl', 'rb'))
app = Flask(__name__)
@app.route('/')
def prediction():
    return render_template('Flower.html')

#preview data set
@app.route('/preview')
def preview():
    datacsv = pd.read_csv('iris.csv')
    #change the name of the columns
    datacsv.columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class']
    return render_template('preview.html',datacsvview=datacsv)



@app.route('/predict', methods=['POST'])
def predicts():
    data1 = request.form['a']
    data2 = request.form['b']
    data3 = request.form['c']
    data4 = request.form['d']
    #data = np.array([[data1, data2, data3, data4]])
    prediction = model.predict([[data1, data2, data3, data4]])
    return render_template('Flowerfinal.html', prediction=prediction)
if __name__ == '__main__':
    app.run(debug=True)