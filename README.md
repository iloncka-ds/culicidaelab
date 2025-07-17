<div align="center">

# CulicidaeLab 🦟

**A configuration-driven Python library for advanced mosquito analysis, featuring pre-trained models for detection, segmentation, and species classification.**

</div>

<p align="center">
  <a href="https://pypi.org/project/culicidaelab/"><img alt="PyPI" src="https://img.shields.io/pypi/v/culicidaelab?color=blue"></a>
  <a href="https://github.com/iloncka-ds/culicidaelab/blob/main/LICENSE"><img alt="License" src="https://img.shields.io/pypi/l/culicidaelab"></a>
  <a href="#">
  <a href="https://github.com/astral-sh/ruff"><img src="https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json" alt="Ruff" /></a>
</p>

---

`CulicidaeLab` provides a robust, extensible framework designed to streamline the pipeline of mosquito image analysis. Built on a powerful configuration system, it allows researchers and developers to easily manage datasets, experiment with models, and process images for classification, detection, and segmentation tasks.

## Key Features

- **Configuration-Driven Workflow**: Manage all settings—from file paths to model parameters—through simple YAML files. Override defaults easily for custom experiments.
- **Ready-to-Use Models**: Leverage pre-trained models for:
    - **Species Classification**: Identify mosquito species using a high-accuracy classifier.
    - **Mosquito Detection**: Localize mosquitoes in images with a YOLO-based detector.
    - **Instance Segmentation**: Generate precise pixel-level masks with a SAM-based segmenter.
- **Unified API**: All predictors share a consistent interface (`.predict()`, `.visualize()`, `.evaluate()`) for a predictable user experience.
- **Automatic Resource Management**: The library intelligently manages local storage, automatically downloading and caching model weights and datasets on first use.
- **Extensible Provider System**: Seamlessly connect to data sources. A `HuggingFaceProvider` is built-in, with an easy-to-implement interface for adding more providers.
- **Powerful Visualization**: Instantly visualize model outputs with built-in, configurable methods for drawing bounding boxes, classification labels, and segmentation masks.

## Installation

It is recommended to use a modern package manager like `uv` or `pip`.

```bash
# Using uv
uv pip install culicidaelab

# Or using pip
pip install culicidaelab
```

## Quick Start

Here's how to classify the species of a mosquito in just a few lines of code. The library will automatically download the necessary model on the first run.

```python
import cv2
from culicidaelab.core import get_settings

# 1. Get the central settings object
# This loads all default configurations for the library.
settings = get_settings()

# 2. Instantiate the classifier
# The settings object knows how to configure the classifier.
classifier = settings.instantiate_from_config("predictors.classifier")

# 3. Load an image
# (Ensure you have an image file at 'path/to/your/image.jpg')
try:
    image = cv2.imread("path/to/your/image.jpg")
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
except (cv2.error, AttributeError):
    print("Error: Could not load image. Please check the file path.")
    exit()

# 4. Make a prediction
# The model is lazy-loaded (downloaded and loaded into memory) here.
with classifier.model_context():
    predictions = classifier.predict(image_rgb)

# 5. Print the results
# The output is a list of (species_name, confidence_score) tuples.
print("Top 3 Predictions:")
for species, confidence in predictions[:3]:
    print(f"- {species}: {confidence:.4f}")

# Example Output:
# Top 3 Predictions:
# - Aedes aegypti: 0.9876
# - Aedes albopictus: 0.0112
# - Culex quinquefasciatus: 0.0009
```

## Going Deeper

`CulicidaeLab` can do much more. You can easily chain predictors, run batch predictions, and visualize results.

```python
# --- Continuing from the Quick Start ---

# Use the detector to find the mosquito first
detector = settings.instantiate_from_config("predictors.detector")
with detector.model_context():
    detections = detector.predict(image_rgb) # Returns [(x, y, w, h, conf)]

# Visualize the detection results
with detector.model_context():
    viz_image = detector.visualize(image_rgb, detections)
    cv2.imwrite("detection_result.png", cv2.cvtColor(viz_image, cv2.COLOR_RGB2BGR))
```

## Practical Applications

`CulicidaeLab` is more than just a set of models; it's a powerful engine for building real-world solutions. Here are some of the ways it can be applied:

-   **Automation in Scientific Laboratories:**
    -   **Bulk Data Processing:** Automatically analyze thousands of images from camera traps or microscopes to assess mosquito populations without manual intervention.
    -   **Reproducible Research:** Standardize the data analysis process, allowing other scientists to easily reproduce and verify research results published using the library.

-   **Integration into Governmental and Commercial Systems:**
    -   **Epidemiological Surveillance:** Use the library as the core "engine" for national or regional monitoring systems to track vector-borne disease risks.
    -   **Custom Solution Development:** Rapidly prototype and create specialized software products for pest control services, agro-industrial companies, or environmental organizations.

-   **Advanced Analytics and Data Science:**
    -   **Geospatial Analysis:** Write scripts to build disease vector distribution maps by processing geotagged images.
    -   **Predictive Modeling:** Use the library's outputs as features for larger models that forecast disease outbreaks based on vector presence and density.

## Documentation

For complete guides, tutorials, and the full API reference, **[visit the documentation site](https://iloncka-ds.github.io/culicidaelab/)**.

The documentation includes:
- In-depth installation and configuration guides.
- Detailed tutorials for each predictor.
- Architectural deep-dives for contributors.
- A full, auto-generated API reference.

## Contributing

Contributions are what make the open-source community such an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**.

Please see our **[Contributing Guide](https://github.com/iloncka-ds/culicidaelab/blob/main/CONTRIBUTING.md)** for details on our code of conduct, development setup, and the pull request process.

## Development Setup

To get a development environment running:

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/iloncka-ds/culicidaelab.git
    cd culicidaelab
    ```

2.  **Install dependencies with `uv` or `pip`**:
    ```bash
    # This installs the library in editable mode and includes all dev tools
    uv pip install -e ".[dev]"
    ```

3.  **Set up pre-commit hooks**:
    ```bash
    pre-commit install
    ```
    This will run linters and formatters automatically on each commit to ensure code quality and consistency.

## License

This project is distributed under the MIT License. See the [LICENSE](https://github.com/iloncka-ds/culicidaelab/blob/main/LICENSE) file for more information.

## Citation

If you use `CulicidaeLab` in your research, please cite it as follows:

```bibtex
@software{culicidaelab2024,
  author = {Ilona Kovaleva},
  title = {{CulicidaeLab: A Configuration-Driven Python Library for Mosquito Analysis}},
  year = {2024},
  publisher = {GitHub},
  journal = {GitHub repository},
  url = {https://github.com/iloncka-ds/culicidaelab}
}
```

## Contact

- **Issues**: Please use the [GitHub issue tracker](https://github.com/iloncka-ds/culicidaelab/issues).
- **Email**: [iloncka.ds@gmail.com](mailto:iloncka.ds@gmail.com)
