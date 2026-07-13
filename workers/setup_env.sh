#!/usr/bin/env bash
# Creates a venv and installs everything needed to run build_all_figures.py
# Usage:  cd brisc_gui && bash setup_env.sh && source .venv/bin/activate
set -e

python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements_figures.txt

echo ""
echo "Environment ready. Run:"
echo "  source .venv/bin/activate"
echo "  python build_all_figures.py"
