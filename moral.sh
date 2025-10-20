#!/usr/bin/env bash
set -euo pipefail

python3 main.py --fair_model moral --model gae --dataset credit --device cuda:1 --epochs 500
python3 main.py --fair_model moral --model gae --dataset german --device cuda:1 --epochs 500
python3 main.py --fair_model moral --model gae --dataset nba --device cuda:1 --epochs 500
python3 main.py --fair_model moral --model gae --dataset facebook --device cuda:1 --epochs 500
python3 main.py --fair_model moral --model gae --dataset pokec_n --device cuda:0 --epochs 500
python3 main.py --fair_model moral --model gae --dataset pokec_z --device cuda:2 --epochs 500
