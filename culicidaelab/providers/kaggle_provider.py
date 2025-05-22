# src/culicidaelab/providers/kaggle_provider.py
from kaggle.api.kaggle_api_extended import KaggleApi
from ..core.base_provider import BaseProvider
from ..core.env_manager import get_api_key


class KaggleProvider(BaseProvider):
    def __init__(self):
        """Initialize Kaggle provider."""
        self.provider_name = "kaggle"
        self.api = KaggleApi()
        self.api_key = get_api_key("kaggle")
        if not self.api_key:
            raise ValueError("Kaggle API key not found in environment variables")
        self.authenticate()

    def download_dataset(self, dataset: str, path: str = "./data"):
        """Download dataset from Kaggle."""
        self.authenticate()
        self.api.dataset_download_files(dataset, path=path, unzip=True)
