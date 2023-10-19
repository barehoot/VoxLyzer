# setup.sh
#!/bin/bash

# Create a virtual environment
python3 -m venv myenv

# Activate the virtual environment
source myenv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Make sure to include 'plotly' in your requirements.txt file.
