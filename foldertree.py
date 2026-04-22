import os
import sys

IGNORE_FOLDERS = {"node_modules","__pycache__",".git",".vscode","__MACOSX", "data","outputs"}

def generate_tree(root_path, prefix=""):
    items = [
        item for item in sorted(os.listdir(root_path))
        if item not in IGNORE_FOLDERS
    ]

    pointers = ["├── "] * (len(items) - 1) + ["└── "] if items else []

    for pointer, item in zip(pointers, items):
        path = os.path.join(root_path, item)
        print(prefix + pointer + item)

        if os.path.isdir(path):
            extension = "│   " if pointer == "├── " else "    "
            generate_tree(path, prefix + extension)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python folder_tree.py <folder_path>")
        sys.exit(1)

    root = sys.argv[1]

    if not os.path.isdir(root):
        print("Error: Provided path is not a folder.")
        sys.exit(1)

    print(os.path.basename(root) + "/")
    generate_tree(root)
