import os
import sys
from sklearn.linear_model import LinearRegression,Ridge,Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor,AdaBoostRegressor,GradientBoostingRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from dataclasses import dataclass

from src.eception import CustomException
from src.logger import logging
from src.utils import save_object,evaluate_models
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor

@dataclass
class ModelTrainerConfig:
    model_trainer_file_path=os.path.join('artifacts','model.pkl')
    
class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()
    def initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info('splitting the train and test array')
            X_train,y_train,X_test,y_test=(
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )
            models={
                "random forest":RandomForestRegressor(),
                "Decision tree": DecisionTreeRegressor(),
                "linera regression":LinearRegression(),
                "catboost regression":CatBoostRegressor(verbose=False),
                "addaboost regressor":AdaBoostRegressor(),
                "Xg boost regressor":XGBRegressor(),
                "K-neighbour regressor":KNeighborsRegressor(),
                "gradient boost regressor":GradientBoostingRegressor()
                
                
            }
            params={
                    "Decision tree": 
                    {
                        "criterion": ["squared_error", "friedman_mse"],
                        "splitter": ["best", "random"],
                        "max_depth": [None, 5, 10, 20]
                    },

                    "random forest": {
                        "n_estimators": [8, 16, 32, 64, 128, 256],
                        "max_depth": [None, 5, 10, 20]
        },

                "linera regression": {
                        "fit_intercept": [True, False]
        },

                "catboost regression": {
                        "iterations": [100, 200],
                        "learning_rate": [0.01, 0.05, 0.1],
                        "depth": [4, 6, 8]
        },

                "addaboost regressor": {
                        "n_estimators": [50, 100, 200],
                        "learning_rate": [0.01, 0.1, 1.0]
        },

                "Xg boost regressor": {
            "n_estimators": [50, 100, 200],
            "learning_rate": [0.01, 0.05, 0.1],
            "max_depth": [3, 5, 7]
        },

        "K-neighbour regressor": {
            "n_neighbors": [3, 5, 7, 9],
            "weights": ["uniform", "distance"],
            "algorithm": ["auto", "ball_tree", "kd_tree", "brute"]
        },

        "gradient boost regressor": {
            "n_estimators": [50, 100, 200],
            "learning_rate": [0.01, 0.05, 0.1],
            "max_depth": [3, 5, 7]
        }
    }
            model_report:dict=evaluate_models(x_train=X_train,y_train=y_train,x_test=X_test,y_test=y_test,models=models,param=params)
            # to get best model score from dict
            best_model_score=max(sorted(model_report.values()))
            ## to get best model from dict
            best_model_name=list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model=models[best_model_name]
            if best_model_score<0.6:
                raise CustomException("no best model found")
            logging.info(f'best found model on both training and testing dataset')
            save_object (
                file_path=self.model_trainer_config.model_trainer_file_path,
                obj=best_model
            )
            predicted=best_model.predict(X_test)
            r2_square=r2_score(y_test,predicted)
            return r2_square
        except Exception as e:
            raise CustomException(e,sys)
    

