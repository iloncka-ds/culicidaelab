# Testing Approach for CulicidaeLab

This document outlines the testing strategy and approach used in the CulicidaeLab project, with a focus on the `BasePredictor` test suite.

## Test Structure

### Test Organization
- Tests are organized in the `tests/` directory, mirroring the project's source structure
- Each test file is named `test_<module_name>.py`
- Test classes are named `Test<ClassName>`
- Test methods are named `test_<method_name>_<scenario>[_when_<condition>]`

### Key Testing Patterns

1. **Fixtures**
   - Defined in `conftest.py` or at the module level
   - Used for test setup and teardown
   - Common fixtures include:
     - `dummy_settings`: Creates a mock Settings object
     - `dummy_predictor`: Creates a test instance of DummyPredictor

2. **Mocks and Patching**
   - `unittest.mock` is used to mock external dependencies
   - Patching is used to replace real implementations with mocks during tests
   - Common patches include:
     - `ModelWeightsManager` to prevent actual downloads
     - File system operations to avoid side effects

3. **Temporary Files**
   - `tmp_path` fixture is used for creating temporary files and directories
   - Ensures tests don't pollute the filesystem
   - Automatically cleaned up after tests complete



## Running Tests

Run all tests:
```bash
pytest
```

Run a specific test file:
```bash
pytest tests/core/test_base_predictor.py -v
```

Run tests with coverage:
```bash
pytest --cov=culicidaelab tests/
```

## Best Practices

1. **Isolation**
   - Each test should be independent
   - Use fixtures to set up required state
   - Clean up after tests using teardown

2. **Readability**
   - Use descriptive test names
   - Group related tests in classes
   - Add docstrings to explain test purpose

3. **Maintainability**
   - Keep test code DRY (Don't Repeat Yourself)
   - Use helper functions for common assertions
   - Keep tests focused on a single behavior

4. **Performance**
   - Use mocks for slow operations
   - Keep tests fast to encourage frequent running
   - Avoid unnecessary I/O in tests
