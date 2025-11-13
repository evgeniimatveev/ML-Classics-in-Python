"""
ml_classics_intro.py

Quick overview script for the "ML Classics in Python (Google Colab)" repository.
This script prints a short description, shows the main parts of the project,
and lists the recommended Python packages.
"""

from textwrap import indent

# 1. Short message about the project ----------------------------------------


def print_header() -> None:
    print("📊 ML Classics in Python (Google Colab)")
    print(
        "Collection of classic machine learning algorithms implemented in Python, "
        "structured into five main parts."
    )
    print(
        "Covers: data preprocessing, regression, classification, clustering, "
        "and association rule learning."
    )
    print("Inspired by: SuperDataScience — Machine Learning A–Z (Python).\n")


# 2. Overview of parts and example algorithms -------------------------------


ML_PARTS = [
    {
        "part": "Part 1 - Data Preprocessing",
        "examples": "Missing values, encoding categorical data, feature scaling",
    },
    {
        "part": "Part 2 - Regression",
        "examples": "Simple & Multiple Linear, Polynomial, SVR, Decision Tree, Random Forest",
    },
    {
        "part": "Part 3 - Classification",
        "examples": "Logistic, KNN, SVM, Decision Tree, Random Forest",
    },
    {
        "part": "Part 4 - Clustering",
        "examples": "K-Means, Hierarchical Clustering",
    },
    {
        "part": "Part 5 - Association Rule Learning",
        "examples": "Apriori, Eclat",
    },
]


def print_structure() -> None:
    print("📂 High-level project structure:\n")
    for block in ML_PARTS:
        print(f"• {block['part']}")
        print(indent(f"- Examples: {block['examples']}\n", prefix="  "))


# 3. Recommended Python libraries -------------------------------------------


REQUIRED_PACKAGES = [
    "numpy",
    "pandas",
    "matplotlib",
    "seaborn",
    "scikit-learn",
    "scipy",
    "mlxtend",
]


def print_dependencies() -> None:
    print("🔧 Recommended Python packages:\n")
    for pkg in REQUIRED_PACKAGES:
        print(f"- {pkg}")
    print(
        "\nTip: in Google Colab you can install any missing package via "
        "`!pip install <package_name>`."
    )


# 4. Main entry point --------------------------------------------------------


def main() -> None:
    """Run all info sections."""
    print_header()
    print_structure()
    print_dependencies()


if __name__ == "__main__":
    main()
