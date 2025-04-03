import numpy as np
import os
from .python_wrapper import CFunctionWrapper  # Use absolute import

# Print the files in the current directory
print("Files in the current directory:")
print(os.listdir(os.path.dirname(__file__)))

# Initialize the wrapper
wrapper = CFunctionWrapper("C_functions.so")