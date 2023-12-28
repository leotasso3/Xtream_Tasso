import json
import pickle
from flask import Flask,request,app,jsonify,url_for,render_template
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer


app=Flask(__name__)
## Load the model
model=pickle.load(open('/Users/leo/Xtream_repo/Xtream_Tasso/decision_tree.pkl','rb'))
# Homepage
@app.route('/')
def home():
    return render_template('web_app.html')


# Endpoint per le predizioni
@app.route('/predict', methods=['GET', 'POST'])
def predict():

    input_data = [x for x in request.form.values()] 

    input_data_processed = pd.DataFrame([input_data]) 

    input_data_processed = pd.get_dummies(input_data_processed)

    numeric_columns = []
    for col in input_data_processed:
        if input_data_processed[col].dtype in ['int64', 'float64']:
            numeric_columns.append(col)
    
    prediction = model.predict(input_data_processed) 

    return render_template('web_app.html', prediction_text=f"Prediction: {prediction}")


if __name__ == "__main__":
    app.run(debug=True, port=5001)


