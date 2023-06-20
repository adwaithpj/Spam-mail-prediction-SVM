#!/bin/bash

# Get the name of the Jupyter notebook file.
notebook_file="file_path.ipynb"

# Get the name of the output file.
output_file="file_path.py"

# Import the necessary modules.
import nbformat
import os

# Read the Jupyter notebook file.
with open(notebook_file, "r") as f:
    nb = nbformat.read(f, as_version=4)

# Get the cells from the Jupyter notebook.
cells = nb.cells

# Merge the cells into a single code.
code = ""
for cell in cells:
    code += cell.source

# Save the code to a file.
with open(output_file, "w") as f:
    f.write(code)

# Print a message to indicate that the script has finished.
print("The code has been saved to the file {}.".format(output_file))
