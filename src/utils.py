import sys
import os
from src.exception import CustomException
import dill
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
import pickle


def save_object(file_path,obj):
    try:
        dir_path=os.path.dirname(file_path)
        os.makedirs(dir_path,exist_ok=True)

        with open(file_path,"wb") as file_obj:
            dill.dump(obj,file_obj)
    except Exception as e:
        raise CustomException(e,sys)
    

def evaluate_models(x_train,x_test,y_train,y_test,models):
    try:
        report={}
        for model_name,model in models.items():
         model.fit(x_train,y_train)

         y_train_pred=model.predict(x_train)
         y_test_pred=model.predict(x_test)

         y_train_accuracy=accuracy_score(y_train,y_train_pred)
         y_test_accuracy=accuracy_score(y_test,y_test_pred)

         report[model_name]=y_test_accuracy

         return report
    except Exception as e:
        raise CustomException(e,sys)
