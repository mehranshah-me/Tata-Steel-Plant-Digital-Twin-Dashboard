#!/bin/bash
# Launch the Digital Safety Twin Dashboard

cd "$(dirname "$0")"
python -m streamlit run app.py

