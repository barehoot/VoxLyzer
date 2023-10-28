#!/bin/bash

# Update and upgrade the system packages
sudo apt-get update
sudo apt-get upgrade -y

# Install FFmpeg
sudo apt-get install -y ffmpeg

# Install Python dependencies from requirements.txt
pip install -r requirements.txt

# Additional commands or installations can be added as needed
# For example, if you have other system dependencies, you can install them here.

# Start your Streamlit app
streamlit run Sent.py
