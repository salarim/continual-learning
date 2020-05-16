#!/bin/bash
#SBATCH --mail-user=salari.m1375@gmail.com
#SBATCH --mail-type=ALL
#SBATCH --account=def-mori_gpu
#SBATCH --job-name=CL-cifar100-parallel
#SBATCH --output=%x-%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=04:00:00
#SBATCH --mem=0
#SBATCH --gres=gpu:p100l:4
#SBATCH --cpus-per-task=12

cd $SLURM_TMPDIR
cp -r ~/scratch/continual-learning .
cd continual-learning

module load python/3.7
virtualenv --no-download venv
source venv/bin/activate
pip install --no-index --upgrade pip
pip install --no-index -r requirements.txt


python main.py --batch-size 128 --epochs 200 --model-type softmax \
--tasks 1 --dataset cifar100 --log-interval 100 --test-interval 5

cp experiments/* ~/scratch/continual-learning/experiments
