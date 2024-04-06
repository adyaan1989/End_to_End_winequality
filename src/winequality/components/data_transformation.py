import os
import urllib.request as request
import zipfile
from winequality.logging import logger
from winequality.utils.common import get_size
from winequality.entity import DataTransformationConfig
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split


class DataTransformation:
    def __init__(self, config: DataTransformationConfig):
        self.config = config

    def train_test_split_ratio(self):
        df = pd.read_csv(self.config.data_path)

        train, test = train_test_split(df)

        train.to_csv(os.path.join(self.config.root_dir, "train.csv"), index = False)
        test.to_csv(os.path.join(self.config.root_dir, "test.csv"), index = False)

        logger.info("Split data into train and test")
        logger.info(train.shape)
        logger.info(test.shape)
        logger.info(train.head)