import os

# Define the folder structure
folder_structure = {
    "my_data_science_project": [
        "data/raw",
        "data/processed",
        "notebooks",
        "src",
        "models",
        "results/figures",
        "tests",
    ]
}

# Function to create the folders
def create_folders(structure):
    for parent, children in structure.items():
        for child in children:
            path = os.path.join(parent, child)
            os.makedirs(path, exist_ok=True)
            print(f"Created folder: {path}")

# Create the folders
create_folders(folder_structure)

# Create a requirements.txt and README.md file
with open("my_data_science_project/requirements.txt", "w") as f:
    f.write("# Requirements\n\n")
    f.write("# List your project dependencies here.\n")

with open("my_data_science_project/README.md", "w") as f:
    f.write("# My Data Science Project\n\n")
    f.write("## Overview\n\n")
    f.write("A brief description of the project.\n\n")
    f.write("## Folder Structure\n\n")
    f.write("This project follows the following folder structure:\n\n")
    f.write("```\n")
    f.write("my_data_science_project/\n")
    f.write("├── data/\n")
    f.write("│   ├── raw/\n")
    f.write("│   └── processed/\n")
    f.write("├── notebooks/\n")
    f.write("├── src/\n")
    f.write("├── models/\n")
    f.write("├── results/\n")
    f.write("│   └── figures/\n")
    f.write("├── tests/\n")
    f.write("├── requirements.txt\n")
    f.write("└── README.md\n")
    f.write("```\n")

print("Folder structure created successfully!")
