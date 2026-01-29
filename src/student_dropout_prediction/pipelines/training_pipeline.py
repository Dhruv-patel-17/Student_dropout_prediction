from src.student_dropout_prediction.logger import logging
from src.student_dropout_prediction.exception import CustomException
from src.student_dropout_prediction.components.data_ingestion import DataIngestion
from src.student_dropout_prediction.components.data_ingestion import DataIngestionConfig
from src.student_dropout_prediction.components.data_transformation import DataTransformation,DataTransformationConfig
from src.student_dropout_prediction.components.model_trainer import ModelTrainer,ModelTrainerConfig
import pickle
import numpy as np
import pandas as pd
import sys
import os

if __name__=="__main__":
    logging.info("The execution has started")
    try:
        data_ingestion=DataIngestion()
        train_data_path,test_data_path=data_ingestion.initiate_data_ingestion()


        data_transformation=DataTransformation()
        train_df,test_df,_=data_transformation.initiate_data_transformation(train_data_path,test_data_path)

        model_trainer=ModelTrainer()
        print(model_trainer.initiate_model_trainer(train_df,test_df))

        
    except Exception as e:
        logging.info("Custom_Exception")
        raise CustomException(e,sys)
    