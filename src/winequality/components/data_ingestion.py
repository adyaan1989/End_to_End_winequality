import os
import urllib.request as request
import zipfile
from winequality.logging import logger
from winequality.utils.common import get_size
from winequality.entity import DataIngestionConfig
from pathlib import Path


class DataInegetation:
    def __init__(self, config: DataIngestionConfig):
        self.config = config

    def download_file(self):
        if not os.path.exists(self.config.local_data_file):
            filename, headers = request.urlretrieve(
                url = self.config.source_URL,
                filename = self.config.local_data_file
            )

            logger.info(f"{filename} downloaded with following information: \n{headers}")
        else:
            logger.info(f"file already exists with size: {get_size(Path(self.config.local_data_file))}")

    def extract_zip_file(self):
        unzip_path = self.config.unzip_dir
        os.makedirs(unzip_path, exist_ok=True)
        with zipfile.ZipFile(self.config.local_data_file, "r") as zip_ref:
            zip_ref.extractall(unzip_path)