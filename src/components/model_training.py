import os
import sys
from src.exception import CustomException
from src.logger import logging
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier,AdaBoostClassifier
from sklearn.metrics import accuracy_score
from dataclasses import dataclass
from sklearn.model_selection import train_test_split
from src.utils import evaluate_models,save_object


@dataclass
class ModelTrainerConfig:
    model_file_path:str=os.path.join("artifacts/model.pkl")


class ModelTrainer:
    def __init__(self):
        self.trainer_config=ModelTrainerConfig()

    

    def initiate_model_trainer(self, train_array, test_array):
     try:
        logging.info("Splitting train and test arrays into features and labels.")
        
        # Assume train_array and test_array are already split, so just extract X and y
        x_train, y_train = train_array[:, :-1], train_array[:, -1]
        x_test, y_test = test_array[:, :-1], test_array[:, -1]

        models = {
            "Logistic": LogisticRegression(),
            "RandomForest": RandomForestClassifier(),
            "GradientBoosting": GradientBoostingClassifier(),
            "AdaBoost": AdaBoostClassifier()
        }

        logging.info("Evaluating models.")
        model_report = evaluate_models(x_train=x_train, x_test=x_test, y_train=y_train, y_test=y_test, models=models)
        
        best_score = max(model_report.values())
        logging.info(f"Best model score: {best_score}")

        best_name = max(model_report, key=model_report.get)
        best_model = models[best_name]

        if best_score < 0.5:
            raise CustomException("No suitable model found with satisfactory performance.")

        logging.info(f"Best model found: {best_name} with score {best_score}.")

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
