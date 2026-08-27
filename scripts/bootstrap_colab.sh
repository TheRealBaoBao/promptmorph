#!/usr/bin/env bash
set -euo pipefail

python -m pip install --upgrade pip
python -m pip install -e ".[dev,sim,ml]"
python -m pytest -q
python -m promptmorph.sim.mujoco_demo --seed 7 --output-dir artifacts
