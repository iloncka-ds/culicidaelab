"""
# CulicidaeLab Datasets Module Example

This notebook demonstrates how to use the datasets module in CulicidaeLab.
"""

# %%
# Third-party imports
import matplotlib.pyplot as plt
import numpy as np
import requests

from datasets import load_dataset
from collections import defaultdict

from culicidaelab.core.settings import get_settings
from culicidaelab.core.provider_service import ProviderService
from culicidaelab.datasets.datasets_manager import DatasetsManager

# %%
print("\n--- 1. Initializing DatasetsManager ---")
settings = get_settings()
provider_service = ProviderService(settings)
manager = DatasetsManager(settings, provider_service)
print("DatasetsManager initialized successfully.")

# %%
print("\n--- 2. Listing all available datasets ---")
available_datasets = manager.list_datasets()
print(f"Available datasets found in configuration: {available_datasets}")

# %%
print("\n--- 3. Getting info for 'classification' dataset ---")
try:
    info = manager.get_dataset_info("classification")
    print(f"  - Name: {info.name}")
    print(f"  - Path/ID: {info.path}")
    print(f"  - Provider: {info.provider_name}")
    print(f"  - Classes: {info.classes}")
except KeyError as e:
    print(e)

# %%
print("\n--- 4. Loading 'classification' dataset for the first time ---")
print("This will trigger the provider to 'download' and then 'load' the data.")
classification_data = manager.load_dataset("classification", split="test")
print("\nDataset loaded successfully!")
print(f"Returned data: {classification_data}")

# %%
print("\n--- 5. Listing currently loaded (cached) datasets ---")
loaded_list = manager.list_loaded_datasets()
print(f"Manager reports these datasets are loaded: {loaded_list}")
print(f"Internal cache state: {manager.loaded_datasets}")

# %%
# Define the dataset name
dataset_name = "iloncka/mosquito-species-classification-dataset"
API_URL = f"https://datasets-server.huggingface.co/croissant-crumbs?dataset={dataset_name}"


def get_metadata(dataset_name):
    """Fetch metadata for a given dataset from Hugging Face."""
    api_url = f"https://datasets-server.huggingface.co/croissant-crumbs?dataset={dataset_name}"
    response = requests.get(api_url, timeout=10)
    response.raise_for_status()
    return response.json()


metadata = get_metadata(dataset_name)
print(metadata)

# %%
dataset = load_dataset(
    "iloncka/mosquito-species-classification-dataset",
    split="test",
)  # , streaming=True, trust_remote_code=True
dataset_head = dataset.take(2)

# %%
set(dataset["label"])

# %%
# Specify the dataset, config, and split you want to query
dataset_name = "iloncka/mosquito-species-classification-dataset"  # e.g., "nyu-mll/glue"
config_name = "default"  # e.g., "cola"
split_name = "test"  # e.g., "train", "validation"

# Construct the API URL
API_URL = (
    f"https://datasets-server.huggingface.co/statistics?dataset={dataset_name}&config={config_name}&split={split_name}"
)


# Function to query the API
def query():
    response = requests.get(API_URL, timeout=10)
    return response.json()


# Fetch and print unique labels
dataset_info = query()


# %%
def get_dataset_summary(dataset_name, dataset_info):
    """
    Generate a summary of dataset information from the dataset statistics.

    Parameters:
    -----------
    dataset_info : dict
        Dictionary containing dataset statistics and information

    Returns:
    --------
    dict
        Organized summary of the dataset
    """
    summary = {
        "dataset_name": dataset_name,
        "total_samples": dataset_info["num_examples"],
        "columns": {},
        "label_distribution": None,
        "image_info": None,
    }

    # Process each column's statistics
    for column in dataset_info["statistics"]:
        col_name = column["column_name"]
        col_type = column["column_type"]
        stats = column["column_statistics"]

        # Special handling for label column
        if col_type == "string_label":
            summary["label_distribution"] = {
                "num_classes": stats["n_unique"],
                "class_distribution": stats["frequencies"],
            }

        # Special handling for image column
        elif col_type == "image":
            summary["image_info"] = {
                "dimensions": f"{int(stats['min'])}x{int(stats['max'])}",
                "num_images": dataset_info["num_examples"],
            }

        # Store basic column information
        summary["columns"][col_name] = {
            "type": col_type,
            "missing_values": {
                "count": stats.get("nan_count", 0),
                "percentage": stats.get("nan_proportion", 0) * 100,
            },
        }

        # Add additional statistics if available
        if "mean" in stats:
            summary["columns"][col_name]["statistics"] = {
                "mean": stats["mean"],
                "std": stats["std"],
                "min": stats["min"],
                "max": stats["max"],
            }

    return summary


# Print summary in a readable format
def print_dataset_summary(summary):
    print(f"Dataset Summary {summary['dataset_name']}:")
    print(f"Total samples: {summary['total_samples']}")
    print("\nImage Information:")
    if summary["image_info"]:
        print(f"Dimensions: {summary['image_info']['dimensions']}")
        print(f"Number of images: {summary['image_info']['num_images']}")

    print("\nLabel Distribution:")
    if summary["label_distribution"]:
        print(f"Number of classes: {summary['label_distribution']['num_classes']}")
        print("\nSamples per class:")
        for class_name, count in summary["label_distribution"]["class_distribution"].items():
            print(f"  {class_name}: {count}")

    print("\nColumns:")
    for col_name, col_info in summary["columns"].items():
        print(f"\n{col_name}:")
        print(f"  Type: {col_info['type']}")
        # Print missing values info, keeping line length <= 120
        missing_count = col_info["missing_values"]["count"]
        missing_pct = col_info["missing_values"]["percentage"]
        print(f"  Missing values: {missing_count} ({missing_pct:.2f}%)")
        if "statistics" in col_info:
            print("  Statistics:")
            print(f"    Mean: {col_info['statistics']['mean']:.2f}")
            print(f"    Std: {col_info['statistics']['std']:.2f}")
            print(f"    Min: {col_info['statistics']['min']}")
            print(f"    Max: {col_info['statistics']['max']}")


# Example usage:
summary = get_dataset_summary(
    dataset_name,
    dataset_info,
)  # Replace dataset_info with your response

# Print the summary
print_dataset_summary(summary)


# %%
def find_min_max_classes(dataset_info):
    """
    Find classes with minimum and maximum number of samples in the dataset.

    Parameters:
    -----------
    dataset_info : dict
        Dictionary containing dataset statistics and information

    Returns:
    --------
    dict
        Information about minimum and maximum classes
    """
    # Get the label frequencies from the statistics
    label_stats = None
    for column in dataset_info["statistics"]:
        if column["column_type"] == "string_label":
            label_stats = column["column_statistics"]["frequencies"]
            break

    if not label_stats:
        return None

    # Find min and max classes
    min_class = min(label_stats.items(), key=lambda x: x[1])
    max_class = max(label_stats.items(), key=lambda x: x[1])

    result = {
        "minimum": {"class_name": min_class[0], "sample_count": min_class[1]},
        "maximum": {"class_name": max_class[0], "sample_count": max_class[1]},
        "difference": max_class[1] - min_class[1],
    }

    return result


# Example usage:
min_max_info = find_min_max_classes(
    dataset_info,
)  # Replace dataset_info with your response

# Print the results in a readable format
if min_max_info:
    print("Class Distribution Analysis:")
    print("\nMinimum samples per class:")
    print(f"  Class: {min_max_info['minimum']['class_name']}")
    print(f"  Count: {min_max_info['minimum']['sample_count']} samples")

    print("\nMaximum samples per class:")
    print(f"  Class: {min_max_info['maximum']['class_name']}")
    print(f"  Count: {min_max_info['maximum']['sample_count']} samples")

    print(f"\nDifference between max and min: {min_max_info['difference']} samples")


# %%
def create_distribution_plot(
    dataset,
    dataset_info,
    color="green",
    figsize=(15, 10),
    output_file="class_distribution.png",
):
    """
    Create a horizontal bar plot of class distribution.

    Parameters:
    -----------
    dataset : HuggingFace IterableDataset
        The dataset containing images and labels
    dataset_info : dict
        Dictionary containing dataset statistics and information
    figsize : tuple, default=(15, 10)
        Figure size (width, height)
    output_file : str, default='class_distribution.png'
        Output file name
    """
    # Get label frequencies from dataset_info
    label_stats = None
    for column in dataset_info["statistics"]:
        if column["column_type"] == "string_label":
            label_stats = column["column_statistics"]["frequencies"]
            break

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
    plt.title("Distribution of Mosquito Species in Dataset", pad=20, fontsize=18)
    plt.xlabel("Number of Samples", fontsize=14)

    # Adjust layout to prevent label cutoff
    plt.tight_layout()

    # Save the plot
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"Distribution plot saved as {output_file}")

    # Display the plot
    plt.show()


# Example usage - create both plots
create_distribution_plot(dataset, dataset_info)


# %%
def create_tree_visualization(
    dataset_info,
    figsize=(15, 10),
    output_file="tree_distribution.png",
):
    """
    Create a tree-like visualization with branch lengths proportional to species count.
    """
    # Get label frequencies from dataset_info
    label_stats = None
    for column in dataset_info["statistics"]:
        if column["column_type"] == "string_label":
            label_stats = column["column_statistics"]["frequencies"]
            break

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
    # Draw main trunk segments between genera
    # prev_end removed (was unused)
    for (genus, _), color in zip(sorted_genera, colors):
        species_count = len(genus_groups[genus])
        start_y = y_positions[current_y_index]
        end_y = y_positions[current_y_index + species_count - 1]

        # Draw main trunk segment for this genus
        ax.plot([trunk_x, trunk_x], [start_y, end_y], color="k", linewidth=3)

        # prev_end = end_y  # Removed unused variable
        current_y_index += species_count

    # Reset current_y_index for species drawing
    current_y_index = 0

    # Draw branches for each genus
    for (genus, total_count), color in zip(sorted_genera, colors):
        species_list = genus_groups[genus]
        species_count = len(species_list)

        # Calculate genus branch position and length
        genus_y = np.mean(
            y_positions[current_y_index : current_y_index + species_count],
        )
        genus_branch_length = 0.02  # Fixed length for genus branches

        # Calculate line thickness based on count
        thickness = 1 + 3 * (total_count - min_genus_count) / (max_genus_count - min_genus_count)

        # Draw genus branch
        ax.plot(
            [trunk_x, trunk_x + genus_branch_length],
            [genus_y, genus_y],
            "-",
            color=color,
            linewidth=thickness,
        )

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
                [
                    y_positions[current_y_index],
                    y_positions[current_y_index + species_count - 1],
                ],
                "-",
                color=color,
                linewidth=1,
                alpha=1,
            )

        # Draw species branches
        for i, (species, count) in enumerate(
            sorted(species_list, key=lambda x: x[1], reverse=True),
        ):
            y_pos = y_positions[current_y_index + i]

            # Calculate species branch length based on count
            species_branch_length = max_branch_length * 0.5 * (count - min_count) / (max_count - min_count)

            # Draw species branch
            species_thickness = 0.5 + 2 * (count - min_count) / (max_count - min_count)
            ax.plot(
                [
                    trunk_x + genus_branch_length,
                    trunk_x + genus_branch_length + species_branch_length,
                ],
                [y_pos, y_pos],
                "-",
                color=color,
                linewidth=species_thickness,
            )

            # Add species name with genus
            species_name = species.replace(
                "_",
                " ",
            ).title()  # Include full name with genus
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
    plt.suptitle(
        "Mosquito Species Distribution by Genus and Species",
        y=0.95,
        fontsize=18,
    )

    # Add total samples count and legend
    total_samples = sum(label_stats.values())
    plt.figtext(
        0.02,
        0.02,
        f"Total samples: {total_samples}\n"
        f"Number of genera: {len(genus_groups)}\n"
        f"Number of species: {len(label_stats)}\n"
        f"Branch length ∝ sample count",
        fontsize=18,
    )

    # Save the plot
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"Tree visualization saved as {output_file}")

    # Display the plot
    plt.show()


# Example usage
create_tree_visualization(dataset_info)
