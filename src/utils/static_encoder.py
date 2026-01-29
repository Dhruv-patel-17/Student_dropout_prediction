import json
import pandas as pd
from sklearn.base import BaseEstimator,TransformerMixin

class StaticCategoryEncoder(BaseEstimator,TransformerMixin):
    def __init__(self,encoding_path,mode='train'):
        self.encoding_path=encoding_path
        self.mode=mode
        self.feature_names_=None

    def fit(self,X,y=None):
        with open(self.encoding_path) as f:
            self.encoding=json.load(f)
            
        self.feature_names_=self._build_feature_names(X)

        #Pre-compute valid encoded values
        self.valid_encoded_values={
            col: set(mapping.values())
            for col,mapping in self.encoding.items()
        }    
        return self

    def transform(self,X):
        X=X.copy()

        for col,mapping in self.encoding.items():
            if col not in X.columns:
                continue

            if X[col].dtype=="object":
                unknown=set(X[col].unique())-set(mapping.keys())
                if unknown:
                    raise ValueError(f"Unknown categories {unknown} in column'{col}")
                X[col]=X[col].map(mapping)

            else:
                if self.mode=="interence":
                    invalid=set(X[col].unique())-self.valid_encoded_values[col]
                    if invalid:
                        raise ValueError(f"Invalid encoded values {invalid} in column '{col}")
        return X 
    def _build_feature_names(self,X):
        feature_names=[]

        feature_names.extend(["Marital status","Application mode","Application order","Course","Daytime/evening attendance","Previous qualification","Mother's qualification","Father's occupation","Displaced","Educational special needs","Debtor","Tuition fees up to date","Gender","Scholarship holder","Age at enrollment","International","Curricular 1st and 2nd sem PCA"])
        return feature_names          

