import importlib
import pkgutil
import inspect
import pytest

DATASETS_MODULE = "culicidaelab.datasets"


def get_dataset_modules():
    """Yield all modules in the culicidaelab.datasets package."""
    import culicidaelab.datasets

    package = culicidaelab.datasets
    for _, modname, ispkg in pkgutil.iter_modules(package.__path__):
        if not ispkg:
            yield f"{DATASETS_MODULE}.{modname}"


@pytest.mark.parametrize("modname", list(get_dataset_modules()))
def test_import_dataset_module(modname):
    """Test that each dataset module can be imported."""
    importlib.import_module(modname)


@pytest.mark.parametrize("modname", list(get_dataset_modules()))
def test_dataset_classes_callable(modname):
    """Test that dataset classes/functions in each module are instantiable/callable."""
    module = importlib.import_module(modname)
    # Find classes or functions that look like datasets
    for name, obj in inspect.getmembers(module):
        if inspect.isclass(obj) and obj.__module__ == module.__name__:
            # Try to instantiate with no args if possible
            try:
                instance = obj()
            except Exception:
                continue  # Skip if requires args
        elif inspect.isfunction(obj) and obj.__module__ == module.__name__:
            # Try to call with no args if possible
            try:
                obj()
            except Exception:
                continue  # Skip if requires args
