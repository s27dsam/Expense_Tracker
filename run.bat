@echo off

:: Check if virtual environment exists
if exist .venv\Scripts\activate.bat (
    call .venv\Scripts\activate.bat
) else (
    echo Virtual environment not found. Creating one...
    python -m venv .venv
    call .venv\Scripts\activate.bat
    
    :: Install dependencies
    pip install -r requirements.txt
)

:: Run the Streamlit app
streamlit run app.py
