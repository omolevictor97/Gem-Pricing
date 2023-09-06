import os
import sys
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.exception import CustomException
from src.logger import logging
import pickle


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
        for i in models:
            value = models[i]
            
            # Train model'ÄÜ_.
            value.fit(X_train, y_train)
            # predict model
            y_test_pred = value.predict(X_test)

            # Check score
            test_score = r2_score(y_test, y_test_pred)
            report[i] = test_score
            
        return report
    except Exception as e:
        logging.info("Error happened at the evaluate_model function")
        raise CustomException(e, sys)