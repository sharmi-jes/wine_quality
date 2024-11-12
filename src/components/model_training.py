import os
import sys
from src.exception import CustomException
from src.logger import logging
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.metrics import accuracy_score
from dataclasses import dataclass
from sklearn.model_selection import train_test_split
from src.utils import evaluate_models, save_object


@dataclass
class ModelTrainerConfig:
    model_file_path: str = os.path.join("artifacts/model.pkl")


class ModelTrainer:
    def __init__(self):
        self.trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Splitting train and test arrays into features and labels.")
            
            # Extract X and y from the train and test arrays
            x_train, y_train = train_array[:, :-1], train_array[:, -1]
            x_test, y_test = test_array[:, :-1], test_array[:, -1]

            # Define models and their corresponding hyperparameter grids
            models = {
                
                "RandomForest": RandomForestClassifier(),
                "GradientBoosting": GradientBoostingClassifier(),
                "AdaBoost": AdaBoostClassifier()
            }

            params = {
               
                "RandomForest": {
                    'n_estimators': [100, 200, 300],
                    'max_depth': [10, 20, 30, None],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4],
                    'max_features': ['auto', 'sqrt', 'log2'],
                    'bootstrap': [True, False],
                    'class_weight': [None, 'balanced'],
                    'random_state': [42]
                },
                "GradientBoosting": {
                    'n_estimators': [100, 200, 300],
                    'learning_rate': [0.01, 0.1, 0.5],
                    'max_depth': [3, 4, 5],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4],
                    'subsample': [0.8, 1.0],
                    'random_state': [42]
                },
                "AdaBoost": {
                    'n_estimators': [50, 100, 200],
                    'learning_rate': [0.01, 0.1, 1.0],
                    'random_state': [42]
                }
            }

            logging.info("Evaluating models.")
            # Here, you should ensure that evaluate_models handles model selection, tuning, etc.
            model_report = evaluate_models(x_train=x_train, x_test=x_test, y_train=y_train, y_test=y_test, models=models, params=params)
            
            best_score = max(model_report.values())
            logging.info(f"Best model score: {best_score}")

            best_name = max(model_report, key=model_report.get)
            best_model = models[best_name]

            # Optionally, raise an exception if the best model is not performing well enough
            if best_score < 0.6:
                raise CustomException("No suitable model found with satisfactory performance.")

            logging.info(f"Best model found: {best_name} with score {best_score}.")

            # Save the best model to the specified file path
            save_object(
                file_path=self.trainer_config.model_file_path,
                obj=best_model
            )

            logging.info("Making predictions with the best model.")
            predictions = best_model.predict(x_test)
            score = accuracy_score(y_test, predictions)
            return score

        except Exception as e:
            raise CustomException(e, sys)
