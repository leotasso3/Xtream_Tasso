import json
import pickle
from flask import Flask,request,app,jsonify,url_for,render_template
import pandas as pd

app=Flask(__name__)
## Load the model
model=pickle.load(open('/Users/leo/Xtream_repo/Xtream_Tasso/decision_tree.pkl','rb'))
# Homepage
@app.route('/')
def home():
    return render_template('web_app_rest.html')


# Endpoint per le predizioni
@app.route('/predict', methods=['GET', 'POST'])
def predict():
    data=request.json['data']
    print(input_data)

    input_data_processed = pd.DataFrame([input_data]) 
    #input_data_processed = input_data_processed.drop(['city', 'target', 'enrollee_id'], axis=1)

    input_data_processed = pd.get_dummies(input_data_processed)

    numeric_columns = [col for col in input_data_processed.columns if input_data_processed[col].dtype in ['int64', 'float64']]
    imputer = SimpleImputer(strategy='mean')
    input_data_processed[numeric_columns] = imputer.fit_transform(input_data_processed[numeric_columns])

    prediction = model.predict(input_data_processed) 

    return render_template('web_app_rest.html', prediction_text=f"Prediction: {prediction}")


if __name__ == "__main__":
    app.run(debug=True, port=5001)


