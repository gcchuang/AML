#!/bin/bash
K_value=10
job="NPM1_upsample_V3_K${K_value}"
directory="../${job}"
convergence_file="convergence.csv"
predictions_file="predictions.csv"

# Create the directory
mkdir "$directory"

# Create the convergence.csv file
touch "$directory/$convergence_file"

# Create the predictions.csv file
touch "$directory/$predictions_file"

echo "Directory and files created successfully."
