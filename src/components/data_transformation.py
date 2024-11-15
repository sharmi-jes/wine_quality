# import sys
# import os
# from src.exception import CustomException
# import pandas as pd
# import numpy as np
# from src.logger import logging
# from sklearn.preprocessing import StandardScaler
# from sklearn.pipeline import Pipeline
# from sklearn.impute import SimpleImputer
# from sklearn.compose import ColumnTransformer
# from src.utils import save_object
# from dataclasses import dataclass

# @dataclass
# class DataTransformationConfig:
#     preprocessor_obj_file_path:str=os.path.join("artifacts/preprocessor.pkl")

# class DataTransformation:
#     def __init__(self):
#         self.transformation_config=DataTransformationConfig()

#     def get_data_transformation(self):
#         logging.info("take numerical and categorical cols")
#         numerical_col=['fixed acidity','volatile acidity','citric acid','residual sugar','chlorides','free sulfur dioxide','total sulfur dioxide','density','pH','sulphates','alcohol']
#         categorical_cols=[]

#         numerical_pipeline = Pipeline(steps=[
#     ('imputer', SimpleImputer(strategy='mean')),  # Handle missing values
#     ('scaler', StandardScaler())  # Standardize the data
# ])

# # Create a ColumnTransformer that applies the numerical pipeline to the numerical columns
#         preprocessor = ColumnTransformer(
#         transformers=[
#         ('num', numerical_pipeline, numerical_col)  # Apply pipeline to numerical columns
#     ],
#         remainder='passthrough'  # Keep non-numerical columns as they are
# )
        
#         return preprocessor
    

#     def intiate_data_transformation(self,train_path,test_path):
#         try:
#             logging.info("read the train and test data")
#             print(f"Train file path: {train_path}")
#             print(type(train_path))
#             train_df = pd.read_csv(train_path)

#             test_df=pd.read_csv(test_path)


#             logging.info("take the target col")
#             target_col="quality"

#             preprocessor_obj=self.get_data_transformation()

#             logging.info("remove the target from the train abd test data")
#             input_train_data=train_df.drop(columns=target_col)
#             input_target_train=train_df[target_col]

#             input_test_data=test_df.drop(columns=target_col)
#             input_target_test=test_df[target_col]

#             logging.info("apply prprocessor obj for the train and test")
#             preprocessor_train=preprocessor_obj.fit_transform(input_train_data)
#             preprocessor_test=preprocessor_obj.transform(input_test_data)

#             train_array=np.c_[
#                 preprocessor_train,np.array(input_target_train)
#             ]
#             test_array=np.c_[
#                 preprocessor_test,np.array(input_target_test)
#             ]

#             save_object(
#             file_path=self.transformation_config.preprocessor_obj_file_path,
#             obj=preprocessor_obj
#             )

#             return (
#                 train_array,
#                 test_array,
#                 self.transformation_config.preprocessor_obj_file_path
#             )
        
#         except Exception as e:
#             raise CustomException(e,sys)

import sys
import os
import pandas as pd
import numpy as np

from src.exception import CustomException
from src.logger import logging

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from dataclasses import dataclass
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path: str = os.path.join("artifacts/preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.transformation_config = DataTransformationConfig()

    def get_data_transformation(self):
        logging.info("Getting the data transformation based on the data")
        numerical_col = ['fixed_acidity','volatile_acidity','citric_acid','residual_sugar','chlorides','free_sulfur_dioxide','total_sulfur_dioxide','density','pH','sulphates','alcohol']
        categorical_cols = []  # Assuming no categorical columns for now

        num_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler())
            ]
        )

        cat_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehotencoder", OneHotEncoder()),
                ("scaler", StandardScaler())
            ]
        )

        preprocessor = ColumnTransformer(
            [
                ("num_pipeline", num_pipeline, numerical_col),
                ("cat_pipeline", cat_pipeline, categorical_cols)
            ]
        )

        return preprocessor
    
    def transform_quality_column(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform the 'quality' column into binary values.
        - 0: Quality < 6
        - 1: Quality >= 6
        """
        df['quality'] = df['quality'].apply(lambda x: 1 if x >= 6 else 0)
        return df
    
    def intiate_data_transformation(self, train_path, test_path):
        try:
            logging.info("Reading the train and test data")
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            # Transform the 'quality' column into binary values (0 or 1)
            train_df = self.transform_quality_column(train_df)
            test_df = self.transform_quality_column(test_df)

            target_column_name = "quality"
            logging.info("Getting the preprocessor for transforming train and test data to the same format")
            preprocessor_obj = self.get_data_transformation()
             
            logging.info("Dropping the target column from the train and test datasets") 
            input_train_data = train_df.drop(columns=[target_column_name], axis=1)
            target_feature_train = train_df[target_column_name]
            input_test_data = test_df.drop(columns=[target_column_name], axis=1)
            target_feature_test = test_df[target_column_name]

            logging.info("Applying the preprocessor object to transform train and test data")
            preprocessor_obj_train = preprocessor_obj.fit_transform(input_train_data)
            preprocessor_obj_test = preprocessor_obj.transform(input_test_data)
            
            logging.info("Combining the target and independent features")
            train_arr = np.c_[
                preprocessor_obj_train, np.array(target_feature_train),
            ]
            test_arr = np.c_[
                preprocessor_obj_test, np.array(target_feature_test)
            ]

            # Saving the preprocessor object
            save_object(
                file_path=self.transformation_config.preprocessor_obj_file_path,
                obj=preprocessor_obj
            )

            return train_arr, test_arr, self.transformation_config.preprocessor_obj_file_path

        except Exception as e:
            raise CustomException(e, sys)



        