#!/bin/bash
echo "Starting Streamlit app to test cost display..."
echo "1. Open http://localhost:8501 in your browser"
echo "2. Send a test query like 'What is RAG?'"
echo "3. Check that cost appears at BOTTOM of message (not in status)"
echo ""
source venv/bin/activate
streamlit run app/frontend.py
