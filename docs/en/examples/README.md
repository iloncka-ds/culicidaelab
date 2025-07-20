# Tutorials & Usage Examples

Welcome to the Usage Examples section, where we move from theory to practice. Here, you will find a series of hands-on tutorials designed to walk you through the core functionalities of `CulicidaeLab`. Each example is self-contained, executable, and builds upon the last, demonstrating a complete and logical workflow for mosquito image analysis.

We will follow a common real-world scenario, taking an image and processing it from start to finish. You will learn how to:

1.  **Understand and Use the `settings` Module:** We'll start with the foundation. You will learn how to access configurations and understand how the library manages resources. This is the central nervous system of `CulicidaeLab`.
2.  **Managing Datasets**: This guide shows you how to use the DatasetsManager to discover, load, and cache the datasets configured within the library. It is essential for anyone looking to perform large-scale model evaluation or exploratory data analysis (EDA).
3.  **Detect Mosquitoes:** Next, we'll use the `MosquitoDetector` to answer the first critical question: "Is there a mosquito in this image, and where is it?"
4.  **Segment Mosquitoes:** For the most detailed analysis, we'll use the `MosquitoSegmenter` to generate a precise, pixel-level mask of the mosquito's exact shape.
5.  **Classify Mosquito Species:** Once a mosquito is found, we'll use the `MosquitoClassifier` to identify its species, a crucial step for epidemiological studies.

Before diving into the examples, please ensure you have successfully installed `CulicidaeLab`. If you haven't, please see the **Installation Guide**.

The examples use sample images that are assumed to be in a local `test_imgs/` directory. You can download these from our GitHub repository or simply replace the file paths with your own images.

We encourage you to run the code snippets yourself in a Jupyter Notebook or a Python script to get a feel for how everything works together. Let's get started
