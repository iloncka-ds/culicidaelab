# Changelog

All notable changes to CulicidaeLab will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.3.1] - 2024-12-XX

### 🚀 Major Features

#### New Backend Architecture
- **Added pluggable inference backend system** with `BaseInferenceBackend` abstract class
- **Added backend factory** with intelligent backend selection (PyTorch vs ONNX)
- **Added concrete backend implementations**:
  - `ClassifierFastAIBackend` and `ClassifierONNXBackend` for classification
  - `DetectorYOLOBackend` for YOLO-based detection  
  - `SegmenterSAMBackend` for Segment Anything Model segmentation

#### High-Performance Serving API
- **Added `serve()` function** for production-optimized inference with automatic model caching
- **Added `clear_serve_cache()`** for explicit memory management
- **Automatic ONNX backend selection** for lightweight, fast inference
- **In-memory predictor caching** eliminates model loading overhead

#### Multiple Input Format Support Added

**Enhancement**: CulicidaeLab predictors now accept multiple image input formats instead of only NumPy arrays:

1. **File Paths** (str or Path): `"image.jpg"` or `Path("image.jpg")`
2. **PIL Images**: `Image.open("image.jpg")` (automatically in RGB)
3. **NumPy Arrays**: `np.array(image)` (RGB format)
4. **Image Bytes**: Raw image bytes for web API integration

This enhancement applies to:
- All predictor `.predict()` methods (`MosquitoDetector`, `MosquitoClassifier`, `MosquitoSegmenter`)
- The high-performance `serve()` API
- Batch processing methods (`.predict_batch()`)

**Benefits**:
- **Simplified Integration**: No need to convert images to NumPy arrays manually
- **Web API Friendly**: Direct support for image bytes from HTTP uploads
- **Flexible Development**: Use the most convenient format for your workflow
- **Backward Compatible**: Existing NumPy array code continues to work

#### Structured Prediction Outputs
- **Added Pydantic-based prediction models** for type-safe, validated outputs:
  - `ClassificationPrediction` with `Classification` objects
  - `DetectionPrediction` with `Detection` and `BoundingBox` objects  
  - `SegmentationPrediction` with NumPy mask and pixel count
- **Built-in JSON serialization** via Pydantic
- **Confidence score validation** (0.0-1.0 range enforcement)

### 🔧 Installation System Redesign

#### New Installation Profiles
- **`[serve]`** - Lightweight ONNX-only serving (default, ~200MB vs ~2GB+)
- **`[serve-gpu]`** - GPU-accelerated ONNX serving
- **`[full]`** - Complete PyTorch + ONNX development environment (CPU)
- **`[full-gpu]`** - Complete development environment with GPU support
- **`[dev]`** - Full development setup with docs, tests, examples

#### Core Dependencies Restructure
- **Minimal core dependencies** - ONNX Runtime, configuration, utilities only
- **Optional PyTorch** - Only installed with `[full]` profiles
- **Layered dependency system** - Low-level backends + high-level profiles

### ✨ New Utility Functions
- **Added `list_models()`** - Discover available model types programmatically
- **Added `list_datasets()`** - Discover available dataset types programmatically
- **Enhanced Settings class** with `list_model_types()` and `list_datasets()` methods

### 🔄 API Changes

#### Breaking Changes
- **Predictor output format changed** from tuples/lists to structured Pydantic models
  ```python
  # Old (v0.2.2)
  predictions = classifier.predict("image.jpg")
  species, confidence = predictions[0]
  
  # New (v0.3.1)  
  result = classifier.predict("image.jpg")
  species = result.top_prediction().species_name
  confidence = result.top_prediction().confidence
  ```

#### New APIs
- **`serve(image, predictor_type="classifier", **kwargs)`** - High-performance inference
- **`clear_serve_cache()`** - Memory management for serving
- **`list_models()`** - Model discovery
- **`list_datasets()`** - Dataset discovery

### 📦 Dependencies

#### Added
- `onnxruntime>=1.22.1` - Core ONNX Runtime support
- `fastprogress==1.0.3` - Progress bars for batch processing
- `pydantic-settings>=2.9.1` - Enhanced settings management

#### Moved to Optional
- `torch>=2.3.1` - Now optional, only in `[full]` profiles
- `torchvision>=0.18.1` - Now optional, only in `[full]` profiles  
- `fastai>=2.7.0,<=2.8.0` - Now optional, only in `[full]` profiles
- `ultralytics>=8.3.0` - Now optional, only in `[full]` profiles

### 🏗️ Architecture Improvements

#### Backend System
- **Automatic backend selection** based on installation profile and configuration
- **Environment detection** - Falls back to ONNX if PyTorch not available
- **Configuration override** - Per-predictor backend specification via YAML
- **Code override** - Force backend selection via `mode` parameter

#### Resource Management
- **Enhanced model weight management** with backend-specific paths
- **Improved caching** with structured directory layout
- **Memory optimization** for serving environments

### 📚 Documentation Updates

#### Installation Guide
- **Updated installation instructions** with profile-specific examples
- **Added performance comparison** between profiles
- **Added deployment examples** for different use cases

#### API Documentation  
- **Updated code examples** for new structured outputs
- **Added serving API examples** for production use
- **Added migration guide** from v0.2.2 to v0.3.1

### 🧪 Testing

#### New Tests
- **Backend factory tests** - Verify intelligent backend selection
- **Serve module tests** - Test caching and performance optimizations
- **Prediction model tests** - Validate Pydantic model behavior
- **Installation profile tests** - Verify dependency isolation

### 🐛 Bug Fixes
- **Improved error handling** in backend selection with clear error messages
- **Enhanced configuration validation** with better error reporting
- **Fixed resource path construction** for cross-platform compatibility

### 🔒 Security
- **Updated dependency versions** to address security vulnerabilities
- **Enhanced input validation** via Pydantic models
- **Improved error handling** to prevent information leakage

---

## Migration Guide: v0.2.2 → v0.3.1

### Installation Changes

#### For Production/Serving
```bash
# Old
pip install culicidaelab

# New (same command, but much lighter!)
pip install culicidaelab
# or explicitly
pip install culicidaelab[serve]

# For GPU serving
pip install culicidaelab[serve-gpu]
```

#### For Development/Research
```bash
# Old  
pip install culicidaelab

# New
pip install culicidaelab[full] --extra-index-url https://download.pytorch.org/whl/cpu
# or for GPU
pip install culicidaelab[full-gpu]
```

### Code Changes

#### Prediction Output Format
```python
# Old format
predictions = classifier.predict("image.jpg")
top_species, confidence = predictions[0]

# New format
result = classifier.predict("image.jpg")
top_species = result.top_prediction().species_name
confidence = result.top_prediction().confidence

# Access all predictions
for pred in result.predictions:
    print(f"{pred.species_name}: {pred.confidence:.3f}")

# JSON serialization
json_output = result.model_dump_json(indent=2)
```

#### High-Performance Serving (New)
```python
# New serving API for production
from culicidaelab.serve import serve, clear_serve_cache

# Fast inference with caching
result = serve("image.jpg", predictor_type="classifier")
print(result.top_prediction().species_name)

# Clean up when done
clear_serve_cache()
```

### Compatibility
- **Backward Compatible**: Existing predictor classes (`MosquitoClassifier`, etc.) still work
- **Output Format**: Only breaking change is the structured output format
- **Installation**: Default installation is now much lighter but fully functional

---

## [0.2.2] - Previous Release

### Features
- Basic predictor classes for classification, detection, and segmentation
- Configuration-driven architecture with YAML files
- HuggingFace integration for models and datasets
- Comprehensive documentation and examples

### Dependencies
- All dependencies bundled in single installation
- PyTorch-based inference only
- Larger installation footprint (~2GB+)