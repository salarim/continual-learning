from models.simple import SimpleNet
from models.resnet import resnet18

def get_model(args):

    if args.dataset == 'mnist':
        model = SimpleNet()
    elif args.dataset == 'cifar10':
        model = resnet18(num_classes=10, high_resolusion=False)
    elif args.dataset == 'cifar100':
        model = resnet18(num_classes=100, high_resolusion=False)
    elif args.dataset == 'imagenet':
        model = resnet18(num_classes=1000, high_resolusion=True)
    else:
        raise ValueError('dataset is not supported.')

    return model
