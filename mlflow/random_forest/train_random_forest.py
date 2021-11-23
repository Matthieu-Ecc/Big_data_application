import os
import warnings
import sys
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.ensemble import RandomForestClassifier
import mlflow
import mlflow.sklearn

import logging
logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)

CURRENT_FOLDER = os.path.dirname(os.path.abspath(__file__))

def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    return rmse, mae

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    np.random.seed(40)
    
    # The predicted column is "quality" which is a scalar from [3, 9]
    X_train = pd.read_csv("../../data/X_train_ready.csv")
    X_test  = pd.read_csv("../../data/X_test_ready.csv")
    y_train = pd.read_csv("../../data/y_train_ready.csv")
    y_test  = pd.read_csv("../../data/y_test_ready.csv")
    X_train = X_train.drop(columns='Unnamed: 0')
    X_test = X_test.drop(columns='Unnamed: 0')
    y_train = y_train.drop(columns='Unnamed: 0')
    y_test  = y_test.drop(columns='Unnamed: 0')

    # Set default values if no alpha is provided
    
    max_depth = int(sys.argv[1]) if len(sys.argv) > 3 else 10
    
    # Useful for multiple runs (only doing one run in this sample notebook)    
    with mlflow.start_run():
        # Execute XGBoost
        model=RandomForestClassifier(max_depth=max_depth)
        model.fit(X_train, y_train)

        # Evaluate Metrics
        predicted_qualities = model.predict(X_test)
        (rmse, mae) = eval_metrics(y_test.to_numpy(), predicted_qualities.transpose())

        # Print out metrics
        print("Random Forest model (max_depth=%f):" % (max_depth))
        print("  RMSE: %s" % rmse)
        print("  MAE: %s" % mae)
        print("  SCORE: %s" % model.score(X_test,y_test))

        # Log parameter, metrics, and model to 
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("model_score", model.score(X_test,y_test))
        mlflow.log_metric("mae", mae)

        mlflow.sklearn.log_model(model, "model_random_forest")
    
