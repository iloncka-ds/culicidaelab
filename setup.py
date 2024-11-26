from __future__ import annotations

from setuptools import find_packages
from setuptools import setup

setup(
    name="culicidaelab",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.21.0",
        "opencv-python>=4.5.0",
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "scikit-image>=0.19.0",
        "pillow>=9.0.0",
        "fastai>=2.7.0",
        "timm>=0.9.0",
        "segment-anything>=1.0",
        "ultralytics>=8.0.0",
        "pyyaml>=6.0.0",
        "huggingface-hub>=0.16.0",
        "datasets>=2.13.0",
    ],
    author="CulicidaeLab Team",
    author_email="your.email@example.com",
    description="A Python library for mosquito detection, segmentation, and classification in images",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/culicidaelab",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Image Recognition",
    ],
    python_requires=">=3.8",
)
