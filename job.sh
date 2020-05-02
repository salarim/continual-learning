#!/bin/bash
#SBATCH --mail-user=salari.m1375@gmail.com
#SBATCH --mail-type=ALL
#SBATCH --account=def-mori_gpu
#SBATCH --job-name=CL
#SBATCH --output=%x-%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=01:00:00
#SBATCH --mem=0
#SBATCH --gres=gpu:p100l:4
#SBATCH --cpus-per-task=4

cd $SLURM_TMPDIR
cp -r ~/scratch/continual-learning .
cd continual-learning

module load python/3.7
virtualenv --no-download venv
source venv/bin/activate
pip install --no-index --upgrade pip
pip install --no-index -r requirements.txt


python main.py --batch-size 128 --epochs 2 --model-type softmax \
--tasks 2 --dataset mnist --exemplar-size 1000 --oversample --log-interval 100 \
--multiprocessing-distributed --world-size 1 --rank 0

cp experiments/* ~/scratch/continual-learning/experiments
