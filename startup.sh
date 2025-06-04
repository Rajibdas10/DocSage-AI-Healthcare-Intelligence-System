#!/bin/bash

# Create the streamlit config directory
mkdir -p /tmp/.streamlit

# Set environment variables
export STREAMLIT_CONFIG_DIR=/tmp/.streamlit
export STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
export STREAMLIT_WATCH_FILE=false
export STREAMLIT_SERVER_HEADLESS=true

# Start the streamlit app
streamlit run app.py --server.port=$PORT --server.address=0.0.0.0
