import os
import sys
from dataclasses import dataclass

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier,ExtraTreesClassifier
from xgboost import XGBClassifier
import lightgbm as lgb


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

    def initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info("Split training and test input data")
            X_train,y_train,X_test,y_test=(
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )

            models={
                "Decision Tree":DecisionTreeClassifier(max_depth=3,random_state=42),
                "Random Forest":RandomForestClassifier(max_depth=3,max_features='sqrt',max_leaf_nodes=9,n_estimators=100,random_state=42),
                "Extra Tree":ExtraTreesClassifier(max_depth=9,criterion='entropy',max_features=None,n_estimators=100,random_state=42),
                "XGBClassifier":XGBClassifier(max_depth=3,gamma=0.2,learning_rate=0.1,n_estimators=100,random_state=42),
                "LGBMClassifier":lgb.LGBMClassifier(max_depth=3,gamma=0.2,learning_rate=0.2,n_estimators=100,objective='binary',metric='auc',verbose=-1)
            }

            model_report=evaluate_models(X_train,y_train,X_test,y_test,models)

            best_model_score=max(sorted(model_report.values()))

            best_model_name=list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model=models[best_model_name]


            if best_model_score<0.5:
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
            