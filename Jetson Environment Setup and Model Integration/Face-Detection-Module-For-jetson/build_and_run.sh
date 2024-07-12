#!/bin/bash

# Remove the existing build directory
if [ -d "build" ]; then
    rm -r build
fi

# Create a new build directory
mkdir build

# Change to the build directory
cd build

# Run CMake to configure the build system
cmake ..

# Build the project using all available cores
make -j$(nproc)

# Run the resulting executable
./ultraface

