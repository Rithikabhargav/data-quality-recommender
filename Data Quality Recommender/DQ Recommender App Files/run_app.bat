@echo off
REM Starts the DQ-Recommender demo at http://localhost:8501
cd /d "%~dp0"
python -m streamlit run app.py
