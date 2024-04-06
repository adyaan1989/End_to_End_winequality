import os
import urllib.request as request
import zipfile
from winequality.logging import logger
from winequality.utils.common import get_size
from winequality.entity import ModelTrainerConfig
from pathlib import Path
import pandas as pd
from sklearn.linear_model import ElasticNet
import joblib


class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        self.config = config

    def train(self):
        train_data = pd.read_csv(self.config.train_data_path)
        test_data = pd.read_csv(self.config.test_data_path)

        x_train = train_data.drop([self.config.target_column], axis=1)
        y_train = train_data[self.config.target_column]
        
        x_test = train_data.drop([self.config.target_column], axis=1)
        y_test = test_data[self.config.target_column]


        lr = ElasticNet(alpha=self.config.alpha, l1_ratio=self.config.l1_ratio, random_state=42)
        lr.fit(x_train, y_train)

        joblib.dump(lr, os.path.join(self.config.root_dir, self.config.model_name))
                    
