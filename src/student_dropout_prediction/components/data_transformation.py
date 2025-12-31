import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from src.utils.static_encoder import StaticCategoryEncoder
from src.student_dropout_prediction.utils import save_object

from src.student_dropout_prediction.logger import logging
from src.student_dropout_prediction.exception import CustomException
import os

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join('artifacts','preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()

    def get_data_transformer_object(self):
        try:
            other_feat=[
                "Marital status","Application mode","Application order","Course",
                "Daytime/evening attendance","Previous qualification",
                "Mother's qualification","Father's occupation",
                "Displaced","Educational special needs",
                "Debtor","Tuition fees up to date","Gender",
                "Scholarship holder","Age at enrollment","International"   
            ]
            curriculum_feat=[
                "Curricular units 1st sem (credited)",
                "Curricular units 1st sem (enrolled)",
                "Curricular units 1st sem (evaluations)",
                "Curricular units 1st sem (approved)",
                "Curricular units 1st sem (grade)",
                "Curricular units 1st sem (without evaluations)",
                "Curricular units 2nd sem (credited)",
                "Curricular units 2nd sem (enrolled)",
                "Curricular units 2nd sem (evaluations)",
                "Curricular units 2nd sem (approved)",
                "Curricular units 2nd sem (grade)",
                "Curricular units 2nd sem (without evaluations)"
            ]    
            pca_transformer=ColumnTransformer(
                transformers=[
                    ("other_feat","passthrough",other_feat),
                    ("pca_transform",PCA(n_components=1),curriculum_feat)
                ],remainder='drop'
            )
            logging.info("PCA Transformation")

            preprocessor=Pipeline(
                steps=[
                    ("static_encoder",StaticCategoryEncoder(encoding_path="artifacts/encoding.json",mode="train")),
                    ("feature_eng",pca_transformer)
                ]
            )

            return preprocessor
        except Exception as e:
            raise CustomException(e,sys)       
        
    def initiate_data_transformation(self,train_path,test_path):
        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)

            logging.info("Reading the train and test file")

            preprocessing_obj=self.get_data_transformer_object()

            target_column_name="Target"

            #Divide the train dataset to independent and dependent feature
            input_feature_train_df=train_df.drop(columns=[target_column_name],axis=1)
            target_feature_train_df=train_df[target_column_name]

            dummies_train =pd.get_dummies(target_feature_train_df,dtype=int)
            dummies_train.drop(['Enrolled','Graduate'],axis=1,inplace=True)
            target_feature_train_df=dummies_train.squeeze()

            #Divide the test dataset to independent and dependent feature
            input_feature_test_df=test_df.drop(columns=[target_column_name],axis=1)
            target_feature_test_df=test_df[target_column_name]

            dummies_test =pd.get_dummies(target_feature_test_df,dtype=int)
            dummies_test.drop(['Enrolled','Graduate'],axis=1,inplace=True)
            target_feature_test_df=dummies_test.squeeze()

            logging.info("Applying preprocessing on training and test dataframe")

            input_feature_train_arr=preprocessing_obj.fit_transform(input_feature_train_df)
            
            input_feature_test_arr=preprocessing_obj.transform(input_feature_test_df)

            train_arr=np.c_[
                input_feature_train_arr,np.array(target_feature_train_df)
            ]
            test_arr=np.c_[
                input_feature_test_arr,np.array(target_feature_test_df)
            ]

            logging.info("Saved preprocessing object")

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            return(
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )




        except Exception as e:
            raise CustomException(sys,e)