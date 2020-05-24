# Continual Learning
Imagenet training:
`python main.py --batch-size 256 --epochs 90 --model-type softmax --tasks 1 --dataset imagenet --unlabeled-dataset imagenet --data-path ILSVRC/Data/CLS-LOC --log-interval 100 --test-interval 5 --gpu 0 --weight-decay 1e-4 --milestones 31 61 --gamma 0.1`

Cifar100 training:
`python main.py --batch-size 128 --epochs 200 --model-type softmax --tasks 1 --dataset cifar100 --unlabeled-dataset cifar100 --log-interval 100 --test-interval 5 --gpu 0 --weight-decay 5e-4 --milestones 60 120 160 --gamma 0.2`
