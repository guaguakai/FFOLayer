#!/bin/bash
#SBATCH -J bilevel_sudoku                  # job name
#SBATCH -A gts-kwang692-paid                  # account to which job is charged
#SBATCH -N 1 --ntasks-per-node=1         # number of nodes and cores per node required
#SBATCH --array=1-50                      # run 1 jobs with seeds 1
#SBATCH --mem-per-cpu=512G                 # memory per core
#SBATCH -t 12:00:00                           # duration of the job
#SBATCH --cpus-per-task=4
#SBATCH --output=/storage/home/hcoda1/3/sho73/sudoku_sbatch_output/%j.out

HOME=/storage/home/hcoda1/3/sho73
REPO_PATH=$HOME/r-kwang692-0/FFOLayer


cd $REPO_PATH
source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate bilevel

methods=("ffocp_eq" "cvxpylayer" "lpgd" "ffoqp_eq_schur" "qpth")

method_index=$(( (SLURM_ARRAY_TASK_ID - 1) /  ))
seed=$(( (SLURM_ARRAY_TASK_ID - 1) % 10 + 1 ))
method=${methods[$method_index]}

echo "SLURM_ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID"
echo "Running method=$method, seed=$seed"

epochs=1
batchSize=8
n=3
python sudoku/main_sudoku.py --method=$method --seed=$seed --epochs=$epochs --seed=$seed --batch_size=$batchSize --n=$n

# epoch=1
# batchSize=8
# ydim=900
# python synthetic_task/main_synthetic.py --batch_size=$batchSize --epochs=$epochs --method=$method --learn_constraint=0 --ydim=$ydim --seed=$seed