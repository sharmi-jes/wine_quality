import sys
import os
from src.exception import CustomException
from src.logger import logging
import pandas as pd
from sklearn.preprocessing import StandardScaler
from src.utils import load_object

class PredictPipeline:
    def __init__(self):
        pass

    def predict(self,features):
        try:
            model_path="artifacts/model.pkl"
            preprocessor_path='artifacts/preprocessor.pkl'
            model_file=load_object(file_path=model_path)
            preprocessor_file=load_object(file_path=preprocessor_path)
            data_scaled=preprocessor_file.transform(features)
            prediction=model_file.predict(data_scaled)
            return prediction
        except Exception as e:
            raise CustomException(e,sys)


class CustomData:
    def __init__(self, fixed_acidity,volatile_acidity,citric_acid,residual_sugar,chlorides,free_sulfur_dioxide,total_sulfur_dioxide,density,pH,sulphates,alcohol):
        self.fixed_acidity=fixed_acidity
        self.volatile_acidity=volatile_acidity
        self.citric_acid=citric_acid
        self.residual_sugar=residual_sugar
        self.chlorides=chlorides
        self.free_sulfur_dioxide=free_sulfur_dioxide
        self.total_sulfur_dioxide=total_sulfur_dioxide
        self.density=density
        self.pH=pH
        self.sulphates=sulphates
        self.alcohol=alcohol
        
    def get_data_as_data_frame(self):
        try:
            input_data={
                "fixed_acidity":[self.fixed_acidity],
                "volatile_acidity":[self.volatile_acidity],
                "citric_acid":[self.citric_acid],
                "residual_sugar":[self.residual_sugar],
                "chlorides":[self.chlorides],
                "free_sulfur_dioxide":[self.free_sulfur_dioxide],
                "total_sulfur_dioxide":[self.total_sulfur_dioxide],
                "density":[self.density],
                "pH":[self.pH],
                "sulphates":[self.sulphates],
                "alcohol":[self.alcohol]
               

            }

            return pd.DataFrame(input_data)
        
        except Exception as e:
            raise CustomException(e,sys)