from flask import Flask, render_template, request
import os 
import numpy as np
import pandas as pd
from winequality.pipeline.prediction import PredictionPipeline

app = Flask(__name__)

@app.route('/', methods=['GET']) # home page
def homePage():
    return render_template("index.html")


@app.route('/train/', methods=['GET']) # model training path
def training():
    os.system("python main.py")
    return "Training Successfully completed"

@app.route('/predict/', methods=['POST','GET'])
def index():
    if request.method == 'POST':
        try:
            # defning the inputs for users
            fixed_acidity =float(request.form['fixed_acidity'])
            volatile_acidity =float(request.form['volatile_acidity'])
            citric_acid =float(request.form['citric_acid'])
            residual_sugar =float(request.form['residual_sugar'])
            chlorides =float(request.form['chlorides'])
            free_sulfur_dioxide =float(request.form['free_sulfur_dioxide'])
            total_sulfur_dioxide =float(request.form['total_sulfur_dioxide'])
            density =float(request.form['density'])
            pH =float(request.form['pH'])
            sulphates =float(request.form['sulphates'])
            alcohol =float(request.form['alcohol'])
       
        except Exception as e:
            print("The exception message is: ", e)
            return "somthing is wrong"
        
    else:
        return render_template('index.html')

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080)