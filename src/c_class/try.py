import numpy as np
import os
from python_wrapper import CFunctionWrapper  # Use absolute import

# Get the absolute path to the shared library
current_dir = os.path.dirname(os.path.abspath(__file__))
library_path = os.path.join(current_dir, "C_functions.so")

# Print the absolute path for debugging
print(f"Using library path: {library_path}")

# Initialize the wrapper with the absolute path
wrapper = CFunctionWrapper(library_path)