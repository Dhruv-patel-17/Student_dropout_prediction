from src.student_dropout_prediction.logger import logging
from src.student_dropout_prediction.exception import CustomException
from src.student_dropout_prediction.components.data_ingestion import DataIngestion
from src.student_dropout_prediction.components.data_ingestion import DataIngestionConfig
from src.student_dropout_prediction.components.data_transformation import DataTransformation,DataTransformationConfig
from src.student_dropout_prediction.components.model_trainer import ModelTrainer,ModelTrainerConfig
from flask import Flask,render_template,request,jsonify
import pickle
import numpy as np
import pandas as pd
import sys

app=Flask(__name__)

with open(r"C:\Users\Dhruv\student_dropout_prediction\artifacts\model.pkl","rb") as f:
    model=pickle.load(f)

with open(r"C:\Users\Dhruv\student_dropout_prediction\artifacts\preprocessor.pkl","rb") as f:
    preprocessor=pickle.load(f)
@app.route("/")
def home():
    return render_template("index.html")  

@app.route("/predict",methods=["GET","POST"])
def predict():
    
    if request.method=="GET":
        return "Predict endpoint is working. Use POST."
    

    data=request.form

    input_data={
        "Marital status":data["Marital status"],
        "Application mode":data["Application mode"],
        "Application order":data["Application order"],
        "Course":data["Course"],
        "Daytime/evening attendance":data["Daytime/evening attendance"],
        "Previous qualification":data["Previous qualification"],
        "Mother's qualification":data["Mother's qualification"],
        "Father's occupation":data["Father's occupation"],
        "Displaced":data["Displaced"],
        "Educational special needs":data["Educational special needs"],
        "Debtor":data["Debtor"],
        "Tuition fees up to date":data["Tuition fees up to date"],
        "Gender":data["Gender"],
        "Scholarship holder":data["Scholarship holder"],
        "Age at enrollment":data["Age at enrollment"],
        "International":data["International"],
        "Curricular units 1st sem (credited)":data["Curricular units 1st sem (credited)"],
        "Curricular units 1st sem (enrolled)":data["Curricular units 1st sem (enrolled)"],
        "Curricular units 1st sem (evaluations)":data["Curricular units 1st sem (evaluations)"],
        "Curricular units 1st sem (approved)":data["Curricular units 1st sem (approved)"],
        "Curricular units 1st sem (grade)":data["Curricular units 1st sem (grade)"],
        "Curricular units 1st sem (without evaluations)":data["Curricular units 1st sem (without evaluations)"],
        "Curricular units 2nd sem (credited)":data["Curricular units 2nd sem (credited)"],
        "Curricular units 2nd sem (enrolled)":data["Curricular units 2nd sem (enrolled)"],
        "Curricular units 2nd sem (evaluations)":data["Curricular units 2nd sem (evaluations)"],
        "Curricular units 2nd sem (approved)":data["Curricular units 2nd sem (approved)"],
        "Curricular units 2nd sem (grade)":data["Curricular units 2nd sem (grade)"],
        "Curricular units 2nd sem (without evaluations)":data["Curricular units 2nd sem (without evaluations)"]
    }


    features=pd.DataFrame([input_data])

    features_transformed=preprocessor.transform(features)
    probability=model.predict_proba(features_transformed)[0][1]
    prediction=int(probability>0.30)

    return render_template(
        "result.html",
        probability=round(float(probability),3),
        prediction="HIGH RISK" if prediction ==1 else "LOW RISK"
    )
          
if __name__=="__main__":
    logging.info("The execution has started")

    try:
        app.run(debug=True)



    except Exception as e:
        logging.info("Custom Exception")
        raise CustomException(e,sys)    
