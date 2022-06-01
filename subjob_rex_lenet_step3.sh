#!/bin/bash -l
#quest 1 core. This will set NSLOTS=1
#$ -pe omp 4
# Request 1 GPU
#$ -l gpus=1
# Request at least compute capability 3.5
#$ -l gpu_c=6.0
# Terminate after 12 hours
#$ -l h_rt=12:00:00

# Join output and error streams
#$ -j y
# Specify Project
#$ -P slme
# Give the job a name
#$ -N rex

stepn=50001
ann=15000
lr=0.001
# load modules
module load python3/3.8.10
module load pytorch/1.9.0

echo "REx:"
pweight=0.01
python main.py \
  --lr=$lr \
  --penalty_anneal_iters=$ann \
  --penalty_weight=$pweight \
  --steps=$stepn \
  --algorithm='rex' \
  --model='lenet'
