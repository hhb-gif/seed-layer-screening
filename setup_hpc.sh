#!/bin/bash
set -e

echo "=== Creating conda env 'seed-layer' with Python 3.11 ==="
conda create -n seed-layer python=3.11 -y

echo "=== Activating env ==="
source activate seed-layer

echo "=== Installing dependencies ==="
pip install pyyaml pymatgen mp-api numpy pandas ase matplotlib

echo "=== Installing MACE (ML potential) ==="
pip install mace-torch

echo "=== Installing project in editable mode ==="
cd /data/home/2025030902017/seed-layer-screening
pip install -e ".[dev]"

echo "=== Done! Test with: ==="
echo "source activate seed-layer"
echo "cd /data/home/2025030902017/seed-layer-screening"
echo "python src/main.py --help"
