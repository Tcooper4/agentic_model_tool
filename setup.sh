#!/bin/bash
echo "Upgrading pip..."
pip install --upgrade pip

echo "Installing OpenAI..."
pip install openai==0.27.4 --force-reinstall

echo "Installing other dependencies..."
pip install -r requirements.txt --upgrade --force-reinstall
