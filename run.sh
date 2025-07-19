#!/bin/bash

# Activate virtual environment
if [ -d ".venv" ]; then
    source .venv/bin/activate
else
    echo "Virtual environment not found. Creating one..."
    python3 -m venv .venv
    source .venv/bin/activate
    
    # Install dependencies
    pip install -r requirements.txt
fi

# Run the Streamlit app
streamlit run app.py
