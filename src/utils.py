import os
import sys
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.exception import CustomException
from src.logger import logging
import pickle

# for saving file
def save_obj(file_path, obj):
    try:
        dir_name = os.path.dirname(file_path)
        os.makedirs(dir_name, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)
        logging.info("A pickled file has been created")
    except Exception as e:
        logging.info("An error has happened at the save_obj")
        raise CustomException(e, sys)
    
def evaluate_model(X_train, y_train, X_test, y_test, models:dict):
    try: 
        report = {}
        for i in range(len(models)):
            model = list(models.values())[i]
            # Train model
            model.fit(X_train,y_train)

            

            # Predict Testing data
            y_test_pred =model.predict(X_test)

            # Get R2 scores for train and test data
            #train_model_score = r2_score(ytrain,y_train_pred)
            test_model_score = r2_score(y_test,y_test_pred)

            report[list(models.keys())[i]] =  test_model_score

        return report
    except Exception as e:
        logging.info("Error happened at the evaluate_model function")
        raise CustomException(e, sys)
# for loading file

def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)
    except Exception as e:
        logging.info("Exception occured in load_objec function")
        raise CustomException(e, sys)