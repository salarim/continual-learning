#!/bin/bash
#SBATCH --mail-user=salari.m1375@gmail.com
#SBATCH --mail-type=ALL
#SBATCH --account=def-mori_gpu
#SBATCH --job-name=CL-cifar100
#SBATCH --output=%x-%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=03:00:00
#SBATCH --mem=0
#SBATCH --gres=gpu:p100:1
#SBATCH --cpus-per-task=12

cd $SLURM_TMPDIR
cp -r ~/scratch/continual-learning .
cd continual-learning

# cp ~/scratch/imagenet/imagenet_object_localization_patched2019.tar .
# echo 'Copied'
# tar -xf imagenet_object_localization_patched2019.tar ILSVRC/Data/CLS-LOC --checkpoint=1000000
# echo 'Extracted!'

module load python/3.7 nixpkgs/16.09  gcc/7.3.0 cuda/10.1
virtualenv --no-download venv
source venv/bin/activate
pip install --no-index --upgrade pip
pip install --no-index -r requirements.txt


python main.py --batch-size 128 --epochs 200 --model-type softmax \
--tasks 1 --dataset cifar100 --unlabeled-dataset cifar100 \
--log-interval 100 --test-interval 5 --gpu 0

cp experiments/* ~/scratch/continual-learning/experiments

