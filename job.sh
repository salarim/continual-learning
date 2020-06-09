#!/bin/bash
#SBATCH --mail-user=salari.m1375@gmail.com
#SBATCH --mail-type=ALL
#SBATCH --account=def-mori_gpu
#SBATCH --job-name=contrastive-1task-self-sup-lr0.001-pietz-aug
#SBATCH --output=%x-%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=02:59:00
#SBATCH --mem=0
#SBATCH --gres=gpu:p100l:4
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


# sup_coef=`echo "$SLURM_ARRAY_TASK_ID/20" | bc -l`

python main.py --lr 1e-3 --batch-size 512 --weight-decay 0.0 --epochs 80 \
--model-type contrastive --tasks 1 --dataset cifar10 --unlabeled-dataset cifar10 \
--log-interval 90 --test-interval 5 --milestones 5 --gamma 0.1 \
--sup-coef 0.0 --save-model

cp experiments/* ~/scratch/continual-learning/experiments
cp *.pt ~/scratch/continual-learning/
cp -r plots ~/scratch/continual-learning/

