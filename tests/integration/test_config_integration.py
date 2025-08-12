import yaml
from pathlib import Path
from culicidaelab.core.provider_service import ProviderService


def test_user_config_overrides_defaults(user_config_dir: Path, settings_factory):
    """
    Tests that a user configuration correctly overrides the default configuration.
    """
    user_processing_value = {
        "confidence_threshold": 0.99,
    }
    with open(user_config_dir / "processing.yaml", "w") as f:
        yaml.dump(user_processing_value, f)

    settings = settings_factory(config_dir=user_config_dir)

    assert settings.get_config("processing.confidence_threshold") == 0.99
    assert settings.get_config("processing.batch_size") == 32


def test_instantiate_provider_via_service(user_config_dir: Path, settings_factory):
    """
    Tests that a provider can be instantiated via the ProviderService,
    which correctly handles dependency injection.
    """
    provider_dict = {
        "my_huggingface": {
            "target": "culicidaelab.providers.huggingface_provider.HuggingFaceProvider",
            "dataset_url": "http://example.com",
        },
    }
    with open(user_config_dir / "providers.yaml", "w") as f:
        yaml.dump(provider_dict, f)

    settings = settings_factory(config_dir=user_config_dir)

    provider_service = ProviderService(settings=settings)
    instance = provider_service.get_provider("my_huggingface")

    assert instance is not None
    assert instance.get_provider_name() == "huggingface"
    assert instance.dataset_url == "http://example.com"
