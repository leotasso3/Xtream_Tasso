import json
import pickle
from flask import Flask, request, app, jsonify, render_template
import numpy as np
import pandas as pd

app=Flask(__name__)
## Load the model
model=pickle.load(open('/Users/leo/Projects/decision_tree.pkl','rb'))
@app.route('/')
def home():
    return render_template('web_app.html')

@app.route('/predict__rest_api',methods=['POST'])
def predict_api():
    data=request.json['data']
    print(data)
    print(np.array(list(data.values())).reshape(1,-1))
    output=model.predict(data)
    print(output[0])
    return jsonify(output[0])

@app.route('/predict',methods=['POST'])
def predict():
    data=[x for x in request.form.values()]
    final_input=pd.get_dummies(data)
    print(final_input)
    output=model.predict(final_input)[0]
    return render_template("web_app.html",prediction_text="The prediction is {}".format(output))

if __name__=="__main__":
    app.run(debug=True)
   
     
