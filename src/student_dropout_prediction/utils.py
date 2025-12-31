import os
import sys
from src.student_dropout_prediction.logger import logging
from src.student_dropout_prediction.exception import CustomException
import pandas as pd
import pickle
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