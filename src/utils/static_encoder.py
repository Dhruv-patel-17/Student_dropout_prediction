import json
import pandas as pd
from sklearn.base import BaseEstimator,TransformerMixin

class StaticCategoryEncoder(BaseEstimator,TransformerMixin):
    def __init__(self,encoding_path,mode='train'):
        self.encoding_path=encoding_path
        self.mode=mode

    def fit(self,X,y=None):
        with open(self.encoding_path) as f:
            self.encoding=json.load(f)

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

