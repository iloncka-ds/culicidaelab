"""
# Managing and Loading Datasets

This tutorial demonstrates how to use the `DatasetsManager` in CulicidaeLab
to interact with the datasets defined in the library's configuration.
"""

# %%
# Install the `culicidaelab` library if not already installed
# !pip install -q culicidaelab

# %%
# Third-party imports
import matplotlib.pyplot as plt
import numpy as np
import requests

from collections import defaultdict

# CulicidaeLab imports
from culicidaelab import get_settings, DatasetsManager

# %% [markdown]
# ## 1. Initializing the DatasetsManager
#
# The `DatasetsManager` is the high-level interface for all dataset operations.
# It requires the `settings` object and a `ProviderService` to function.

# %%
print("--- 1. Initializing DatasetsManager ---")
settings = get_settings()
manager = DatasetsManager(settings)
print("DatasetsManager initialized successfully.")

# %% [markdown]
# ## 2. Listing Available Datasets
#
# You can easily see all datasets configured in the library.

# %%
print("\n--- 2. Listing all available datasets ---")
available_datasets = manager.list_datasets()
print(f"Available datasets found in configuration: {available_datasets}")

# %% [markdown]
# ## 3. Getting Information for a Specific Dataset
#
# Before loading, you can retrieve the configuration metadata for any dataset.

# %%
print("\n--- 3. Getting info for 'classification' dataset ---")
try:
    info = manager.get_dataset_info("classification")
    print(f"  - Name: {info.name}")
    print(f"  - Hugging Face Repository: {info.repository}")
    print(f"  - Data Format: {info.format}")
    print(f"  - Provider: {info.provider_name}")
    # print(f"  - Classes: {info.classes}") # This can be long, so we'll omit it here.
except KeyError as e:
    print(e)

# %% [markdown]
# ## 4. Loading a Dataset
#
# When you load a dataset for the first time, the manager performs several actions:
# 1. It locates the appropriate data provider (e.g., `HuggingFaceProvider`).
# 2. It instructs the provider to download the dataset to a local cache.
# 3. It loads the dataset from the local cache into memory.
#
# On subsequent calls, the manager will use the cached version, making loading much faster.

# %%
print("\n--- 4. Loading 'classification' dataset's 'test' split for the first time ---")
print("This may take a moment as it triggers a download from Hugging Face.")
classification_data = manager.load_dataset("classification", split="test")
print("\nDataset loaded successfully!")
print(f"Returned data type: {type(classification_data)}")
print(f"Dataset features: {classification_data.features}")
print(f"Number of samples in test split: {len(classification_data)}")


# %% [markdown]
# ## 5. Listing Loaded Datasets
#
# The manager keeps an internal cache of datasets that have been loaded during the session.

# %%
print("\n--- 5. Listing currently loaded (cached) datasets ---")
loaded_list = manager.list_loaded_datasets()
print(f"Manager reports these datasets are loaded: {loaded_list}")

# %% [markdown]
# ---
# ## Advanced: Exploring Dataset Statistics with the Hugging Face API
#
# The rest of this tutorial goes beyond the core `culicidaelab` library functionality.
# It demonstrates how you can directly query the Hugging Face Datasets Server API
# to get detailed statistics and create insightful visualizations for mosquito species dataset.
# This is useful for exploratory data analysis (EDA).
#
# **Note:** The following code does not use the `DatasetsManager` and is provided as a supplementary example.

# %%
# Define the dataset name we want to explore
repo_id = "iloncka/mosquito-species-classification-dataset"

# %% [markdown]
# ### Fetching Dataset Metadata and Statistics
# We'll use helper functions to query the API endpoints for general metadata and detailed statistics.


# %%
def get_dataset_metadata(repo_id):
    """Fetch general metadata for a given dataset from Hugging Face."""
    api_url = f"https://datasets-server.huggingface.co/croissant-crumbs?dataset={repo_id}"
    response = requests.get(api_url, timeout=10)
    response.raise_for_status()
    return response.json()


def get_dataset_statistics(repo_id, config_name="default", split_name="test"):
    """Fetch detailed column statistics for a dataset split."""
    api_url = (
        f"https://datasets-server.huggingface.co/statistics?dataset={repo_id}&config={config_name}&split={split_name}"
    )
    response = requests.get(api_url, timeout=10)
    response.raise_for_status()
    return response.json()


print(f"--- Fetching statistics for '{repo_id}' ---")
dataset_info = get_dataset_statistics(repo_id)
print("Statistics fetched successfully.")


# %% [markdown]
# ### Visualizing Class Distribution
# A balanced dataset is crucial for training a good model. Let's visualize the number of samples per species.


# %%
def get_label_stats(dataset_info):
    """Extract label statistics from dataset_info."""
    label_stats = None
    for column in dataset_info["statistics"]:
        if column["column_type"] == "string_label":
            label_stats = column["column_statistics"].get("frequencies", {})
            break

    return label_stats


def create_distribution_plot(
    dataset_info,
    repo_id,
    color="teal",
    figsize=(15, 10),
    output_file="class_distribution.png",
):
    # (Code from original script remains unchanged here)
    # Get label frequencies from dataset_info
    label_stats = get_label_stats(dataset_info)

    if not label_stats:
        print("No label statistics found in dataset_info")
        return
    # Sort classes by sample count
    sorted_items = sorted(label_stats.items(), key=lambda x: x[1], reverse=True)
    classes, counts = zip(*sorted_items)
    # Create figure with custom size
    _, ax = plt.subplots(figsize=figsize)
    # Create horizontal bars
    y_pos = np.arange(len(classes))
    ax.barh(y_pos, counts, align="center", color=color, alpha=0.8)
    # Customize the plot
    ax.set_yticks(y_pos)
    # Format class names by replacing underscores with spaces and capitalize
    formatted_classes = [c.replace("_", " ").title() for c in classes]
    ax.set_yticklabels(formatted_classes, fontsize=16)
    # Add value labels on the bars
    for i, v in enumerate(counts):
        ax.text(v + 0.5, i, str(v), va="center", fontsize=20)
    # Add title and labels
    plt.title(f"Distribution of Mosquito Species in {repo_id}", pad=20, fontsize=18)
    plt.xlabel("Number of Samples", fontsize=14)
    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    # Save the plot
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"Distribution plot saved as {output_file}")
    # Display the plot
    plt.show()


# Create the plot
create_distribution_plot(dataset_info, repo_id)


# %% [markdown]
# ### Visualizing Taxonomic Distribution
# We can also visualize the data in a more structured, tree-like format to see how species are grouped by genus.


# %%
def create_tree_visualization(dataset_info, figsize=(15, 10), output_file="tree_distribution.png"):
    # (Code from original script remains unchanged here)
    # Get label frequencies from dataset_info
    label_stats = get_label_stats(dataset_info)

    if not label_stats:
        print("No label statistics found in dataset_info")
        return
    # Group species by genus
    genus_groups = defaultdict(list)
    genus_totals = defaultdict(int)
    for species, count in label_stats.items():
        genus = species.split("_")[0]
        genus_groups[genus].append((species, count))
        genus_totals[genus] += count
    # Sort genera by total count
    sorted_genera = sorted(genus_totals.items(), key=lambda x: x[1], reverse=True)
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    # Calculate scaling factors
    max_count = max(label_stats.values())
    min_count = min(label_stats.values())
    max_genus_count = max(genus_totals.values())
    min_genus_count = min(genus_totals.values())
    # Calculate positions
    total_species = sum(len(group) for group in genus_groups.values())
    y_positions = np.linspace(0.1, 0.9, total_species)
    trunk_x = 0.15  # Position of main vertical line
    max_branch_length = 0.4  # Maximum branch length
    current_y_index = 0
    text_offset = 0.02
    # Color map for genera
    colors = plt.cm.tab20(np.linspace(0, 1, len(genus_groups)))
    # Draw main trunk segments between genera
    for (genus, _), color in zip(sorted_genera, colors):
        species_count = len(genus_groups[genus])
        start_y = y_positions[current_y_index]
        end_y = y_positions[current_y_index + species_count - 1]
        # Draw main trunk segment for this genus
        ax.plot([trunk_x, trunk_x], [start_y, end_y], color="k", linewidth=3)
        current_y_index += species_count
    # Reset current_y_index for species drawing
    current_y_index = 0
    # Draw branches for each genus
    for (genus, total_count), color in zip(sorted_genera, colors):
        species_list = genus_groups[genus]
        species_count = len(species_list)
        # Calculate genus branch position and length
        genus_y = np.mean(y_positions[current_y_index : current_y_index + species_count])
        genus_branch_length = 0.02  # Fixed length for genus branches
        # Calculate line thickness based on count
        thickness = 1 + 3 * (total_count - min_genus_count) / (max_genus_count - min_genus_count)
        # Draw genus branch
        ax.plot([trunk_x, trunk_x + genus_branch_length], [genus_y, genus_y], "-", color=color, linewidth=thickness)
        # Add genus name
        ax.text(
            trunk_x - 0.02,
            genus_y,
            f"{genus.title()}\n({total_count} total)",
            horizontalalignment="right",
            verticalalignment="center",
            fontsize=18,
            fontweight="bold",
        )
        # Draw vertical connector for species
        if species_count > 1:
            ax.plot(
                [trunk_x + genus_branch_length, trunk_x + genus_branch_length],
                [y_positions[current_y_index], y_positions[current_y_index + species_count - 1]],
                "-",
                color=color,
                linewidth=1,
                alpha=1,
            )
        # Draw species branches
        for i, (species, count) in enumerate(sorted(species_list, key=lambda x: x[1], reverse=True)):
            y_pos = y_positions[current_y_index + i]
            # Calculate species branch length based on count
            species_branch_length = max_branch_length * 0.5 * (count - min_count) / (max_count - min_count)
            # Draw species branch
            species_thickness = 0.5 + 2 * (count - min_count) / (max_count - min_count)
            ax.plot(
                [trunk_x + genus_branch_length, trunk_x + genus_branch_length + species_branch_length],
                [y_pos, y_pos],
                "-",
                color=color,
                linewidth=species_thickness,
            )
            # Add species name with genus
            species_name = species.replace("_", " ").title()
            ax.text(
                trunk_x + genus_branch_length + species_branch_length + text_offset,
                y_pos,
                f"{species_name} ({count})",
                verticalalignment="center",
                fontsize=16,
            )
        current_y_index += species_count
    # Customize the plot
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    # Add title
    plt.suptitle("Mosquito Species Distribution by Genus and Species", y=0.95, fontsize=18)
    # Add total samples count and legend
    total_samples = sum(label_stats.values())
    text_img = f"""Total samples: {total_samples}\n
    Number of genera: {len(genus_groups)}\n
    Number of species: {len(label_stats)}\n
    Branch length ∝ sample count"""
    # Add text at the bottom of the figure
    plt.figtext(
        0.02,
        0.02,
        text_img,
        fontsize=18,
    )
    # Save the plot
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"Tree visualization saved as {output_file}")
    # Display the plot
    plt.show()


# Example usage
create_tree_visualization(dataset_info)
