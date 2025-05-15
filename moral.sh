python3 main.py --fair_model moral --model gae --dataset credit  --device cuda:1 --sim_coeff 10 --epoch 500; 
python3 main.py --fair_model moral --model gae --dataset german --device cuda:1 --sim_coeff 10 --epoch 500;
python3 main.py --fair_model moral --model gae --dataset nba  --device cuda:1 --sim_coeff 10 --epoch 500; 
python3 main.py --fair_model moral --model gae --dataset facebook  --device cuda:1 --sim_coeff 10 --epoch 500; 
python3 main.py --fair_model moral --model gae --dataset pokec_n  --device cuda:0 --sim_coeff 10 --epoch 500; 
python3 main.py --fair_model moral --model gae --dataset pokec_z  --device cuda:2 --sim_coeff 10 --epoch 500;
