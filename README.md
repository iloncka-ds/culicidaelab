# CulicidaeLab

A comprehensive Python library for mosquito detection, segmentation, and classification using state-of-the-art deep learning models.

## Features

- **Mosquito Detection**: Detect mosquitoes in images using YOLOv8
- **Mosquito Segmentation**: Segment mosquito instances using the Segment Anything Model (SAM)
- **Species Classification**: Classify mosquito species using FastAI and timm models
- **Easy-to-use API**: Simple and intuitive interface for all functionalities
- **Comprehensive Documentation**: Detailed tutorials and API documentation
- **Visualization Tools**: Built-in visualization capabilities for detection, segmentation, and classification results

## Installation

It is recommended to use [uv](https://docs.astral.sh/uv/) to manage Python projects:

```bash
uv add culicidaelab
```

Alternatively, add it with `pip`:

```bash
pip install culicidaelab
```
## Quick Start

```python
from culicidaelab.detection import MosquitoDetector
from culicidaelab.segmentation import MosquitoSegmenter
from culicidaelab.classification import MosquitoClassifier

# Initialize models
detector = MosquitoDetector()
segmenter = MosquitoSegmenter()
classifier = MosquitoClassifier()

# Process an image
image_path = "path/to/your/image.jpg"
detections = detector.detect(image_path)
segments = segmenter.segment(image_path)
species = classifier.classify(image_path)
```

## Documentation

For detailed documentation and tutorials, visit the [documentation site](https://your-docs-site.com). The documentation includes:

- Installation guide
- API reference
- Usage tutorials
- Model training guides
- Best practices

## Tutorials

The following Jupyter notebooks provide step-by-step tutorials:

1. `examples/tutorial_part_1_mosquito_detection.ipynb`: Learn how to detect mosquitoes in images
2. `examples/tutorial_part_2_mosquito_segmentation.ipynb`: Explore mosquito segmentation capabilities
3. `examples/tutorial_part_3_mosquito_classification.ipynb`: Understand species classification

## Requirements

- Python 3.11+
- PyTorch 2.0+
- Other dependencies are listed in `pyproject.toml`

## Development Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/culicidaelab.git
cd culicidaelab
```

2. Install development dependencies with `uv`:
```bash
uv pip install --editable ".[dev]"
```

This will install all development dependencies including:
- Code formatting tools (black, ruff)
- Testing frameworks (pytest)
- Documentation tools (mkdocs)
- Jupyter notebook support (ipykernel)

## Contributing

Contributions are welcome! See the [Contributing Guide](CONTRIBUTING.md) for details.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If CulicidaeLab is used in research, please cite:

```bibtex
@software{culicidaelab2024,
  author = {Ilona Kovaleva},
  title = {CulicidaeLab: A Python Library for Mosquito Analysis},
  year = {2024},
  url = {https://github.com/iloncka-ds/culicidaelab}
}
```

## Contact

- Issues: Please use the [GitHub issue tracker](https://github.com/iloncka-ds/culicidaelab/issues)
- Email: iloncka.ds@gmail.com
