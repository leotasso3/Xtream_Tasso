import json
import pickle

from flask import Flask,request,app,jsonify,url_for,render_template
import numpy as np
import pandas as pd

app=Flask(__name__)
## Load the model
model=pickle.load(open('/Users/leo/Xtream_repo/Xtream_Tasso/decision_tree.pkl','rb'))
@app.route('/')
def home():
    return render_template('web_app_rest.html')

@app.route('/predict_api',methods=['POST'])
def predict_api():
    data=request.json['data']
    print(data)
    output=model.predict(data)
    print(output[0])
    return jsonify(output[0])

@app.route('/predict',methods=['POST'])
def predict():
    data=[x for x in request.form.values()]
    data = pd.get_dummies(data)
    output=model.predict(data)[0]
    return render_template("web_app_rest.html",prediction_text="The House price prediction is {}".format(output))



if __name__=="__main__":
    app.run(debug=True)
   
     