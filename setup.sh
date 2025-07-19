#!/bin/bash

# Create the .streamlit directory
mkdir -p ~/.streamlit/

# Create the credentials.toml file (can be empty if no specific credentials are needed)
echo "" > ~/.streamlit/credentials.toml

# Create the config.toml file with server settings for deployment
echo "[server]
headless = true
port = \$PORT
enableCORS = false
" > ~/.streamlit/config.toml
