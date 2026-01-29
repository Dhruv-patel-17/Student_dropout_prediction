import os
import sys
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier,ExtraTreesClassifier
from xgboost import XGBClassifier
import lightgbm as lgb
import pickle

from sklearn.metrics import recall_score
from src.student_dropout_prediction.utils import save_object,evaluate_models
from src.student_dropout_prediction.logger import logging
from src.student_dropout_prediction.exception import CustomException

@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join("artifacts","model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()

    def initiate_model_trainer(self,train_df,test_df):
        try:
            logging.info("Split training and test input data")
            X_train=train_df.iloc[:,:-1]
            y_train=train_df.iloc[:,-1]

            X_test=test_df.iloc[:,:-1]
            y_test=test_df.iloc[:,-1]

            models={
                "Decision Tree":DecisionTreeClassifier(random_state=42),
                "Random Forest":RandomForestClassifier(random_state=42),
                "Extra Tree":ExtraTreesClassifier(random_state=42),
                "XGBClassifier":XGBClassifier(random_state=42),
                "LGBMClassifier":lgb.LGBMClassifier(verbose=-1,random_state=42)
            }


            params={
                "Decision Tree":{
                    'max_depth':3
                },
                "Random Forest":{
                    'max_depth':3,
                    'max_features':'sqrt',
                    'max_leaf_nodes':9,
                    'n_estimators':100
                },
                "Extra Tree":{
                    'max_depth':9,
                    'criterion':'entropy',
                    'max_features':None,
                    'n_estimators':100
                },
                "XGBClassifier":{
                    'max_depth':3,
                    'gamma':0.2,
                    'learning_rate':0.1,
                    'n_estimators':100
                },
                "LGBMClassifier":{
                    'max_depth':9,
                    'learning_rate':0.1,
                    'objective':'binary',
                    'metric':'binary_logloss',
                    'n_estimators':75
                },
            }

            model_report=evaluate_models(X_train,y_train,X_test,y_test,models,params)

            best_model_score=max(sorted(model_report.values()))

            best_model_name=list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model=models[best_model_name]

            print("This is the best model")
            print(best_model_name)

            model_names=list(params.keys())
            actual_model=""

            for model in model_names:
                if best_model_name==model:
                    actual_model=actual_model+model

            best_params=params[actual_model]        



            if best_model_score<0.4:
                raise CustomException("NO BEST MODEL FOUND")
            logging.info("Best found model on both training and test dataset")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            predicted=best_model.predict(X_test)

            recall=recall_score(y_test,predicted)

            return recall
            

            
        except Exception as e:
            raise CustomException(e,sys)
            