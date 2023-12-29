import json
import pickle
from flask import Flask, request, app, jsonify, url_for, render_template
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

# Initialize Flask app
app = Flask(__name__)

# Load the trained model
model = pickle.load(open('/Users/leo/Xtream_repo/Xtream_Tasso/one_row_DT.pkl', 'rb'))

# Homepage - Renders the html template
@app.route('/')
def home():
    return render_template('web_app.html')

# Endpoint for predictions: the POST method allows the user to send an HTTP request (the prediction) to the server throught the html form
@app.route('/predict', methods=['POST']) #Setting the url of the endpoint '/predict'
def predict():

    # List of all the columns from the dataset on which the DT has been trained (already dummy coded)
    all_cols = ['city_development_index','training_hours','gender_Female', 'gender_Male', 'gender_Other', 'relevent_experience_Has relevent experience', 'relevent_experience_No relevent experience', 'enrolled_university_Full time course', 'enrolled_university_Part time course', 'enrolled_university_no_enrollment', 'education_level_Graduate', 'education_level_High School', 'education_level_Masters', 'education_level_Phd', 'education_level_Primary School', 'major_discipline_Arts', 'major_discipline_Business Degree', 'major_discipline_Humanities', 'major_discipline_No Major', 'major_discipline_Other', 'major_discipline_STEM', 'experience_1', 'experience_10', 'experience_11', 'experience_12', 'experience_13', 'experience_14', 'experience_15', 'experience_16', 'experience_17', 'experience_18', 'experience_19', 'experience_2', 'experience_20', 'experience_3', 'experience_4', 'experience_5', 'experience_6', 'experience_7', 'experience_8', 'experience_9', 'experience_<1', 'experience_>20', 'company_size_10/49', 'company_size_100-500', 'company_size_1000-4999', 'company_size_10000+', 'company_size_50-99', 'company_size_500-999', 'company_size_5000-9999', 'company_size_<10', 'company_type_Early Stage Startup', 'company_type_Funded Startup', 'company_type_NGO', 'company_type_Other', 'company_type_Public Sector', 'company_type_Pvt Ltd', 'last_new_job_1', 'last_new_job_2', 'last_new_job_3', 'last_new_job_4', 'last_new_job_>4', 'last_new_job_never']

    # Store the input data sent by the user in the HTML page
    input_data = [[x for x in request.form.values()]] # This 'double squared parenthesis' is necessary for creating the pandas dataset
    input_data_list = input_data[0] #this further list is created for an easier iteration through the values 

    # Dictionary to hold input categorical columns and their values
    input_categorical_cols = {
        'gender': input_data_list[2],
        'relevent_experience': input_data_list[3],
        'enrolled_university': input_data_list[4],
        'education_level': input_data_list[5],
        'major_discipline': input_data_list[6],
        'experience': input_data_list[7],
        'company_size': input_data_list[8],
        'company_type': input_data_list[9],
        'last_new_job': input_data_list[10]
    }

    # Create an empty DataFrame with all the columns 
    df = pd.DataFrame(columns=all_cols)

    display_pred = '' # Defining the variable which will store the output of the model
                      # I have defined it here for eventually storing an error message if the user inserts non valid values in the form


    # Set values in the DataFrame based on input categorical columns
    # Here I set 1 in the dummy coded columns according to the input given by the user
    # If a non valid value is given, of the course the related dummy column doesn't exist: therefore I store in the output an error message
    for key in input_categorical_cols:
        dummy_col = f"{key}_{input_categorical_cols[key]}"
        if dummy_col in all_cols:
            df[dummy_col] = [1] 
        else:   
            display_pred = f"Error: The value for {key} is not valid"

    # Set other dummy columns in the DataFrame to 0
    for col in df:
        if df[col][0] != 1:
            df.loc[0, col] = 0

    # Assigning the 2 numerical values to the non-categorical columns
    df.loc[0, 'city_development_index'] = input_data_list[0]
    df.loc[0, 'training_hours'] = input_data_list[1]

    # Make predictions using the loaded model
    pred = model.predict(df)

    # Prepare the display prediction text based on the prediction result
    if display_pred == '':
        if pred[0] == 0:
            display_pred = 'Loyal employee'
        else:
            display_pred = 'Non-loyal employee'

    # Render web_app.html template and pass the prediction text to be displayed
    return render_template('web_app.html', prediction_text=display_pred)


# Run the Flask app
if __name__ == "__main__":
    app.run(debug=True, port=5001)
