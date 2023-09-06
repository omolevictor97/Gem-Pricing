import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet

from src.exception import CustomException
from src.logger import logging


from src.utils import save_obj, evaluate_model

from dataclasses import dataclass
import sys
import os

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")

class Model_Trainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
        
    def initiate_model_training(self, train_array, test_array):
        try:

            X_train, y_train, X_test, y_test = (
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )

            models = {
                "Lasso_Regression" : Lasso(),
                "Linear_Regression" : LinearRegression(),
                "Ridge_Regression" : Ridge(),
                "Elastic_Net" : ElasticNet()
            }
            model_report:dict = evaluate_model(X_train, y_train, X_test, y_test, models)
            print(model_report)
            print("="* 50, end=" ")
            logging.info(f"Model Report: {model_report} ")
            best_model_score = max(sorted(model_report.values()))
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model = models[best_model_name]

            logging.info(f"The best model score is {best_model_score} and the best model name is {best_model_name}")

            save_obj(
                self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )
        except Exception as e:
            logging.info(f"Error is {e}")
            raise CustomException(e, sys)