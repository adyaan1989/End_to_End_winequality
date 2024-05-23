import joblib
import numpy as np
import pandas as pd
from pathlib import Path


class PredictionPipeline(object):
    def __init__(self):
        self.model = joblib.load(Path('dataStore/model_trainer/model.joblib'))

    def predict(self, data):
        prediction = self.model.predict(data)

        return prediction
    