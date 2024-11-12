import sys
import os
import pandas as pd
import numpy as np
from ..exception import CustomException
from src.exception import CustomException
from src.logger import logging

from src.components.data_transformation import DataTransformation
from src.components.data_transformation import DataTransformationConfig

from src.components.model_training import ModelTrainer
from src.components.model_training import ModelTrainerConfig

from sklearn.model_selection import train_test_split
from dataclasses import dataclass


@dataclass
class DataIngestionConfig:
    train_data_path:str=os.path.join("artifacts/train.csv")
    test_data_path:str=os.path.join("artifacts/test.csv")
    raw_data_path:str=os.path.join("artifacts/raw.csv")

class DataIngestion:
    def __init__(self):
        self.ingestion_config=DataIngestionConfig()


    def initiate_data_ingestion(self):
        try:
            logging.info("read the data")
            df=pd.read_csv(r"D:\RESUME ML PROJECTS\WINE_QUALITY\notebooks\cleandata.csv")

            logging.info("create a directory")
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True)

            df.to_csv(self.ingestion_config.raw_data_path,index=False,header=True)

            logging.info("split the data")
            train_set,test_set=train_test_split(df,test_size=0.2)


            logging.info("pass the train data its path")
            train_set.to_csv(self.ingestion_config.train_data_path)
            logging.info("pass the test data into its path")
            test_set.to_csv(self.ingestion_config.test_data_path)
            return(
                train_set,
                test_set
            )
        
        except Exception as e:
            raise CustomException(e,sys)
if __name__=="__main__":
    obj = DataIngestion()
    train_data, test_data = obj.initiate_data_ingestion()

    transformer = DataTransformation()
    # Save DataFrames to temporary files before passing their paths
    train_data.to_csv("temp_train.csv", index=False)
    test_data.to_csv("temp_test.csv", index=False)

    train_arr,test_arr,_=transformer.intiate_data_transformation("temp_train.csv", "temp_test.csv")

    model=ModelTrainer()
    print(model.initiate_model_trainer(train_arr,test_arr))
