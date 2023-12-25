## Imports
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.tree import DecisionTreeClassifier
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix



## This is a custom function aimed at displayig the most importan metrics for a binary classification, together with a confusion matrix
## The input for this function must be binary predictions
def evaluate(y_pred, y_validation):
    cm = confusion_matrix(y_validation, y_pred)
    acc = accuracy_score(y_validation,y_pred)
    tn, fp, fn, tp = cm.ravel()
    recall = tp / (tp + fn)
    beta = 0.5

    if tp == 0 and fp == 0:
        precision = 0.0
        F1 = 0.0
    else:
        precision = tp / (tp + fp)
        F1 = (1+beta**2)*((precision * recall)/((beta**2 * precision) + recall))
        
    print("Confusion Matrix:")
    print("{:>10} {:>10} {:>10}".format("", "Predicted 0", "Predicted 1"))
    print("{:>10} {:>10} {:>10}".format("Actual 0", tn, fp))
    print("{:>10} {:>10} {:>10}".format("Actual 1", fn, tp))
    print("Recall:", round(recall, 3))
    print("Precision:", round(precision, 3))
    print("Accuracy:", round(acc, 4))
    print("F1 score:", round(F1, 4))




### I will provide a function that performs and end to end pipeline for a new dataset.
### The motivation for model tuning and data preprocessing are the same present in the 'challenge2_and_3' file.
### The pipeline contains the following steps:
### - dataset split: this is done mainly for showing the user how the model is generalizing on new data
### - dummy coding the catogorical variables
### - replacing any eventual 'na' value for the continuos variables 
### - training the Decision tree classifier (with optimized HP) on the new dataset (training set, which is going to be 80% of the total)
### - evalation of the generalization capabilities of the model on out-of-sample data (test set, 20% of the total)
### The function returns the trained model. The user can decide to save it and load it back for any future uses.
    
### Important note: as the challenge 4 requested, this function returns a trained model. If in the future the user wants
### to use this model for doing prediction on new data, he must be sure that the data will be pre-processed at the same
### way that they have been pre-processed in this pipeline

def train_model(data): #The data needs to be a pandas dataset

    #Division of predictors and target variable, plus the drop of the 'city' column (explanation in challenge 2)
    x = data.drop(['city', 'target', 'enrollee_id'], axis = 1) 
    y = data['target']

    #Train test split
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 42)

    #Storing all the continuos variables columns
    numeric_columns = []
    for col in x.columns:
        if x[col].dtype in ['int64', 'float64']:
            numeric_columns.append(col)


    #Dummy coding the categorical variables 
    X_train = pd.get_dummies(X_train) 
    X_test = pd.get_dummies(X_test) 

    # Defining the pipeline for imputing 'na' values for the numeric columns 
    preprocessor = ColumnTransformer([
        ('imputer', SimpleImputer(strategy='mean'), numeric_columns)
    ])

    # defining the pipeline for the full pre processing and model training 
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', DecisionTreeClassifier(criterion='entropy', max_depth=15, min_samples_split=40))  
    ])

    # Train the model
    pipeline.fit(X_train, y_train)



    ## MODEL EVALUATION
    print("Model evaluation:\n")

    y_pred = pipeline.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy on test set: {accuracy:.2f}")

    print("Classification Report:\n\n")
    evaluate(y_test, y_pred)

    return pipeline



# Testing the code
df = pd.read_csv("https://raw.githubusercontent.com/leotasso3/Xtream_Tasso/main/datasets/employee-churn/churn.csv")  
model = train_model(df)

print(model)