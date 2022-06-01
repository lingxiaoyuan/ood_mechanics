#!/bin/bash -l

stepn=50001
ann=0
lr=0.001
pweight=0
echo "ERM:"
python main.py \
  --lr=$lr \
  --penalty_anneal_iters=$ann \
  --penalty_weight=$pweight \
  --steps=$stepn \
  --algorithm='erm' \
  --model='lenet'


stepn=50001
ann=15000
lr=0.001
pweight=0.01
echo "REx:"
python main.py \
  --lr=$lr \
  --penalty_anneal_iters=$ann \
  --penalty_weight=$pweight \
  --steps=$stepn \
  --algorithm='rex' \
  --model='lenet'
