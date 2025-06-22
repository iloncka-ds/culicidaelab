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
def test_dataset_classes_callable(modname, caplog):
    """Test that dataset classes/functions in each module are instantiable/callable.

    Args:
        modname: Name of the module to test
        caplog: pytest fixture for capturing log output
    """
    module = importlib.import_module(modname)

    for name, obj in inspect.getmembers(module):
        # Skip imported members
        if obj.__module__ != module.__name__:
            continue

        if inspect.isclass(obj):
            try:
                instance = obj()
                assert instance is not None  # Basic sanity check
            except (TypeError, ValueError) as e:
                # Expected for classes that require arguments
                pytest.skip(f"Skipping {name}: {e}")
            except Exception as e:
                pytest.fail(f"Failed to instantiate {name}: {e}")

        elif inspect.isfunction(obj):
            try:
                _ = obj()  # Call the function but ignore the result
                # Optional: If you need to validate the return type, use:
                # result = obj()
                # assert isinstance(result, expected_type)
            except (TypeError, ValueError) as e:
                # Expected for functions that require arguments
                pytest.skip(f"Skipping {name}: {e}")
            except Exception as e:
                pytest.fail(f"Failed to call {name}: {e}")
