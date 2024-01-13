#!/bin/bash

# Define the name of the virtual environment
VENV_NAME="myenv"

# Check if the virtual environment exists, create it if it doesn't
if [ ! -d "$VENV_NAME" ]; then
    echo "Creating virtual environment..."
    python3 -m venv $VENV_NAME
fi

# Activate the virtual environment
source $VENV_NAME/bin/activate

# Install the package from your GitHub repository
# Replace 'your_github_repo_url' with your actual GitHub repo URL
pip install git+https://github.com/EvertvanNoort/MyTranscriber

echo "Setup complete. Virtual environment '$VENV_NAME' is ready and the package is installed."
