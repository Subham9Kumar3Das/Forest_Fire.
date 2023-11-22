import pickle
from flask import Flask,jsonify,render_template,request
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

# Import Ridge regressor model.
ridge_pickle=pickle.load(open('Models./Model.pkl','rb'))
standard_scaler_model=pickle.load(open('Models./Scaler.pkl','rb'))

# Route for home page.
@app.route('/')
def index():
    return render_template('index.html')

#Predict Data.
@app.route('/predict_data',methods=['GET','POST'])
def predict_datapoint():
    if request.method=='POST':
        Temperature=float(request.form.get('Temperature'))
        RH=float(request.form.get('RH'))
        Ws=float(request.form.get('Ws'))
        Rain=float(request.form.get('Rain'))
        FFMC=float(request.form.get('FFMC'))
        DMC=float(request.form.get('DMC'))
        ISI=float(request.form.get('ISI'))
        Classes=float(request.form.get('Classes'))
        Region=float(request.form.get('Region'))


        new_data_scaled=standard_scaler_model.transform([[Temperature,RH,Ws,Rain,FFMC,DMC,ISI,Classes,Region]])
        result=ridge_pickle.predict(new_data_scaled)

        return render_template('home.html',result=result[0])

        
    else:
        return render_template('home.html')
    

if __name__=="__main__":
    app.run(host="0.0.0.0")
