## First thing in data transformation is to import all libraries needed for Feature Engineering
import os # for accessing the Operating System
import sys
from dataclasses import dataclass 
from src.exception import CustomException # for raising exceptions and errors
from src.logger import logging # for logging information
import pandas as pd # for reading dataset
import numpy as np # for python numerical and arrays handling and manipulation
from sklearn.preprocessing import StandardScaler, OrdinalEncoder # for scaling features and categorical features
from sklearn.impute import SimpleImputer # for handling missing values
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from src.utils import save_obj


@dataclass

class DataTransformationConfig:
    preprocessor_obj_file_path:str = os.path.join("artifacts", "preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.preprocessor = DataTransformationConfig()
    def get_data_transformation_obj(self):
        try:
            # Define which column should be ordinal-encoded and which should be scaled
            categorical_cols = ["cut", "color", "clarity"]
            numerical_cols = ["carat", "depth", "table", "x", "y", "z"]

            # Define the custom ranking for each ordinal variable
            cut_categories = ["Fair", "Good", "Very Good", "Premium", "Ideal"]
            color_categories = ["D", "E", "F", "G", "H", "I", "J"]
            clarity_categories = ["I1", "SI2", "SI1", "VS2", "VS1", "VVS1", "VVS2", "IF"]

            ## Numerical Pipeline

            logging.info("Pipeline Initiated")

            numerical_pipeline = Pipeline(

                steps = [("imputer",SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler())]
            )


            ## Categorical Pipeline

            categorical_pipeline = Pipeline(

                steps= [
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("ordinal_encoder", (OrdinalEncoder(categories=[cut_categories, color_categories, clarity_categories]))),
                    ("scaler", StandardScaler())
                ]
            )
            # preprocessor

            preprocessor = ColumnTransformer([

                ("numerical_pipeline", numerical_pipeline, numerical_cols),
                ("categorical_pipeline", categorical_pipeline, categorical_cols)
            ])

            logging.info("Pipeline Completed")
            return preprocessor
        except Exception as e:
            logging.info("An Error Has Occured In Data Transformation")
            raise CustomException(e, sys)
        
    def initiate_data_transformation(self, train_path, test_path):
        try:
            logging.info("Reading the csv files")
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Read train and test data completed")
            logging.info(f"Train DataFrame Head : \n {train_df.head().to_string()}")
            logging.info(f"Test DataFrame Head : \n {test_df.head().to_string()}")

            logging.info("Obtaining preprocessor info")

            preprocessor_obj = self.get_data_transformation_obj()

            target_column_name = "price"
            drop_columns = [target_column_name, "Unnamed: 0"]

            input_feature_train_df = train_df.drop(columns=drop_columns, axis=1)
            target_feature_train_df = train_df[target_column_name]


            input_feature_test_df = test_df.drop(columns=drop_columns, axis=1)
            target_feature_test_df = test_df[target_column_name]

            ### Transforming using the preprocessing object
            input_feature_train_arr = preprocessor_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessor_obj.transform(input_feature_test_df)

            ### Concatenating the input features and the target features using numpy for fast machine learning

            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            save_obj(
                file_path=self.preprocessor.preprocessor_obj_file_path, 
                     obj=preprocessor_obj
                     )
            
            logging.info("Pickle file saved")
            
            return (
                train_arr,
                test_arr,
                self.preprocessor.preprocessor_obj_file_path
            )

            
        except Exception as e:
            logging.info("Error has occured in intiate data transformation")
            raise CustomException(e,sys)