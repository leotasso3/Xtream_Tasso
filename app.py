import json
import pickle
from flask import Flask,request,app,jsonify,url_for,render_template
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer


app=Flask(__name__)
## Load the model
model=pickle.load(open('/Users/leo/Xtream_repo/Xtream_Tasso/one_row_DT.pkl','rb'))
# Homepage
@app.route('/')
def home():
    return render_template('web_app.html')


# Endpoint per le predizioni
@app.route('/predict', methods=['GET', 'POST'])

def predict():

    columns = ['city_development_index', 'gender', 'relevent_experience', 'enrolled_university', 'education_level', 'major_discipline', 'experience', 'company_size', 'company_type', 'last_new_job', 'training_hours']

    input_data = [x for x in request.form.values()] 

    dataset = pd.DataFrame(input_data, columns = columns) 

    input_data_processed = pd.get_dummies(dataset)

    prediction = model.predict(input_data_processed) 

    return render_template('web_app.html', prediction_text=f"Prediction: {prediction}")


if __name__ == "__main__":
    app.run(debug=True, port=5001)


