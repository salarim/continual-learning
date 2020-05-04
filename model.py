from models.simple import SimpleNet
from models.resnet import resnet

def get_model(args):

    if args.dataset == 'mnist':
        model = SimpleNet()
    elif args.dataset == 'cifar10':
        model = ResNet(depth=args.depth, num_classes=10)
    elif args.dataset == 'cifar100':
        model = ResNet(depth=args.depth, num_classes=100)
    elif args.dataset == 'imagenet':
        model = ResNet(depth=args.depth, num_classes=1000)
    else:
        raise ValueError('dataset is not supported.')
    return model
