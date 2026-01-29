import os
import sys
from src.student_dropout_prediction.logger import logging
from src.student_dropout_prediction.exception import CustomException
import pandas as pd
import pickle
from sklearn.metrics import recall_score
import numpy as np

def read_data():
    logging.info("Reading Data")
    try:
        df=pd.read_csv(os.path.join('notebook/data','raw.csv'))
        return df
    except Exception as ex:
        raise CustomException(ex)
    
def save_object(file_path,obj):
    try:
        dir_path=os.path.dirname(file_path)

        os.makedirs(dir_path,exist_ok=True)

        with open(file_path,"wb") as file_obj:
            pickle.dump(obj,file_obj)

    except Exception as e:
        raise CustomException(e,sys)     

def evaluate_models(X_train,y_train,X_test,y_test,models,param):
    try:
        report={}

        for i in range(len(list(models))):
            model=list(models.values())[i]
            para=param[list(models.keys())[i]]
            model.set_params(**para)

            model.fit(X_train,y_train)
            y_pred=model.predict(X_test)

            test_score=recall_score(y_test,y_pred)

            report[list(models.keys())[i]]=test_score

        return report    
    except Exception as e:
        raise CustomException(e,sys)       