from flask import Flask, jsonify, request
from flask_cors import CORS
from numpy import loadtxt
import tensorflow as tf
from tensorflow.keras.models import load_model
import pandas as pd
import numpy as np
import pickle

app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

lr_model = pickle.load(open('finalized_lr_model.sav','rb'))
nn_model = load_model('model.h5')
scaler = pickle.load(open('scaler.pkl', 'rb'))

@app.route('/predict', methods=["GET"])
def predict():
    if request.method == 'GET':
        a = int(request.args.get('age'))
        b = int(request.args.get('gender'))
        c = int(request.args.get('height'))
        d = float(request.args.get('weight'))
        e = int(request.args.get('systolic_bp'))
        f = int(request.args.get('diastolic_bp'))
        g = int(request.args.get('cholestrol_level'))
        h = int(request.args.get('glucose_level'))
        i = int(request.args.get('smoke'))
        j = int(request.args.get('alcohol'))
        k = int(request.args.get('physical_active'))
        l = d / (c / 100)**2
        
        features = np.array([a,b,c,d,e,f,g,h,i,j,k,l])
        features_df = pd.DataFrame(features)
        finalized_df = features_df.transpose()
        scaled_df = scaler.transform(finalized_df)
        lr_prediction = lr_model.predict(scaled_df)
        nn_prediction = nn_model.predict(scaled_df)
        pred = [1 if y >= 0.5 else 0 for y in nn_prediction]
        return jsonify(
            {"lr_prediction":str(lr_prediction[0]),
             "nn_prediction": str(pred[0]),
             "probability" : str(nn_prediction[0][0])
            })

