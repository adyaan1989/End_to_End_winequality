import os
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from urllib.parse import urlparse
import mlflow
import mlflow.sklearn
import numpy as np
import joblib
from winequality.entity import ModelEvaluationConfig
from winequality.utils.common import save_json
from pathlib import Path


class ModelEvaluation:
    def __init__(self, config: ModelEvaluationConfig):
        self.config = config
    
    def eval_metrics(self, actual, pred):
        r2 = r2_score(actual, pred)
        rmse = np.sqrt(mean_squared_error(actual, pred))
        mae = mean_absolute_error(actual, pred)
        return r2, rmse, mae
    
    def log_into_mlflow(self):
        test_data = pd.read_csv(self.config.test_data_path)
        model = joblib.load(self.config.model_path)

        x_test = test_data.drop([self.config.target_column], axis=1)
        y_test = test_data[self.config.target_column]

        mlflow.set_registry_uri(self.config.mlflow_uri)
        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

        with mlflow.start_run():
            
            predict_quality = model.predict(x_test)
            (r2, rmse, mae) = self.eval_metrics(y_test, predict_quality)

            # Saving metrics as local
            scores = {"r2":r2, "rmse":rmse, "mae":mae}
            save_json(path=Path(self.config.metric_file_name), data=scores)
            
            mlflow.log_params(self.config.all_params)
            
            mlflow.log_metric("r2", r2)
            mlflow.log_metric("rmse", rmse)
            mlflow.log_metric("mae", mae)
            
            # Model registry does not work with file store
            if tracking_url_type_store != "file":
                
                # Register the model
                # There are other ways to use the Model Registry, which depends on the use case,
                # please refer to the doc for more information:
                # https://mlflow.org/docs/latest/model-registry.html#api-workflow
                mlflow.sklearn.log_model(model,"model", registered_model_name="ElasticnetModel")
                #mlflow.sklearn.log_model(model,"model", registered_model_name="sk-learn-random-forest-reg-model")
            else:
                mlflow.sklearn.log_model(model, "model")
                