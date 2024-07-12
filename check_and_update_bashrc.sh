#!/bin/bash

# Define the lines to add
cuda_path_line='export PATH=/usr/local/cuda/bin${PATH:+:${PATH}}'
cuda_ld_library_path_line='export LD_LIBRARY_PATH=/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}'

# Add the lines to .bashrc if they are not already present
if ! grep -Fxq "$cuda_path_line" ~/.bashrc; then
    echo "$cuda_path_line" >> ~/.bashrc
    echo "Added CUDA path to .bashrc"
else
    echo "CUDA path already exists in .bashrc"
fi

if ! grep -Fxq "$cuda_ld_library_path_line" ~/.bashrc; then
    echo "$cuda_ld_library_path_line" >> ~/.bashrc
    echo "Added CUDA library path to .bashrc"
else
    echo "CUDA library path already exists in .bashrc"
fi

# Source the updated .bashrc
source ~/.bashrc
echo ".bashrc updated and sourced"

