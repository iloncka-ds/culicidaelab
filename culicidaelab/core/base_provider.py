# src/culicidaelab/core/base_provider.py
from abc import ABC, abstractmethod


class BaseProvider(ABC):
    @abstractmethod
    def download_dataset(self, *args, **kwargs):
        """Abstract method for downloading resources"""
        pass

    @abstractmethod
    def download_model_weights(self, *args, **kwargs):
        """Abstract method for downloading model weights"""
        pass

    @abstractmethod
    def get_provider_name(self) -> str:
        """Abstract method for getting provider name"""
        pass

    def __call__(self, *args, **kwargs):
        return self.download(*args, **kwargs)
