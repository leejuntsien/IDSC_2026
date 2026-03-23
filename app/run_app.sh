#!/bin/bash
cd "$(dirname "$0")/.."
python -m streamlit run app/streamlit_app.py --server.maxUploadSize 50
