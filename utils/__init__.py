import sys
sys.path.insert(0, '/home/lpetrini/git/diffeo-sota/')
from collections import OrderedDict
from models import *

def load_net_from_run(r, trained, device='cpu', shuffle_channels=False):
    args = r['args']
    args.device = device
    state = OrderedDict([(k[7:], r['best']['net'][k]) for k in r['best']['net']])
    if shuffle_channels:
    ## permute channels of filter weights!!!
        for k in state:
            if len(state[k].shape) == 4:
                state[k] = state[k][torch.randperm(state[k].shape[0])]
    if 'VGG' in args.net and 'bn' not in args.net:
        for k in state:
            if 'running' in k:
                args.net += 'bn'
                break
    net = select_net(args)
    if trained:
        net.load_state_dict(state)
    else:
        for c in net.modules():
            if 'batchnorm' in str(type(c)):
                c.track_running_stats = False
                c.running_mean = None
                c.running_var = None
    net.eval()
    net = net.to(device)
    return net

def select_net(args):
    num_ch = 1 if 'mnist' in args.dataset or 'twopoints' in args.dataset else 3
    nc = 200 if 'tiny' in args.dataset else 10
    nc = 2 if 'diffeo' in args.dataset else nc
    num_classes = 1 if args.loss == 'hinge' else nc
    imsize = 28 if 'mnist' in args.dataset or 'twopoints' in args.dataset else 32
    try:
        args.fcwidth
    except:
        args.fcwidth = 64
    try:
        args.param_list
    except:
        args.param_list = 0
    try:
        args.width
    except:
        args.width = args.fcwidth
    try:
        args.width_factor
    except:
        args.width_factor = 1
    try:
        args.pretrained
    except:
        args.pretrained = 0
    if not args.pretrained: # and not args.scattering_mode
        if 'VGG' in args.net:
            if 'bn' in args.net:
                bn = True
                net_name = args.net[:-2]
            else:
                bn = False
                net_name = args.net
            net = VGG(net_name, num_ch=num_ch, num_classes=num_classes, batch_norm=bn, param_list=args.param_list, width_factor=args.width_factor)
        if args.net == 'AlexNet':
            net = AlexNet(num_ch=num_ch, num_classes=num_classes)
        if args.net == 'ResNet18':
            net = ResNet18(num_ch=num_ch, num_classes=num_classes)
        if args.net == 'ResNet34':
            net = ResNet34(num_ch=num_ch, num_classes=num_classes)
        if args.net == 'ResNet50':
            net = ResNet50(num_ch=num_ch, num_classes=num_classes)
        if args.net == 'ResNet101':
            net = ResNet101(num_ch=num_ch, num_classes=num_classes)
        if args.net == 'LeNet':
            net = LeNet(num_ch=num_ch, num_classes=num_classes)
        if args.net == 'GoogLeNet':
            net = GoogLeNet(num_ch=num_ch, num_classes=num_classes)
        if args.net == 'MobileNetV2':
            net = MobileNetV2(num_ch=num_ch, num_classes=num_classes)
        if args.net == 'DenseNet121':
            net = DenseNet121(num_ch=num_ch, num_classes=num_classes)
        if args.net == 'EfficientNetB0':
            net = EfficientNetB0(num_ch=num_ch, num_classes=num_classes)
        if args.net == 'MinCNN':
            net = MinCNN(num_ch=num_ch, num_classes=num_classes, h=args.width, fs=args.filter_size, ps=args.pooling_size)
        if args.net == 'LCN':
            net = MinCNN(num_ch=num_ch, num_classes=num_classes, h=args.width, fs=args.filter_size, ps=args.pooling_size)
        if args.net == 'DenseNetL2':
            net = DenseNetL2(num_ch=num_ch * imsize ** 2, num_classes=num_classes, h=args.width)
        if args.net == 'DenseNetL4':
            net = DenseNetL4(num_ch=num_ch * imsize ** 2, num_classes=num_classes, h=args.width)
        if args.net == 'DenseNetL6':
            net = DenseNetL6(num_ch=num_ch * imsize ** 2, num_classes=num_classes, h=args.width)
        if args.net == 'FC':
            net = FC()
    else:
        raise ValueError
    return net