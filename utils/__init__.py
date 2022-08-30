import sys
sys.path.insert(0, '/home/lpetrini/git/diffeo-sota/')
from collections import OrderedDict
from models import *
import numpy as np
import matplotlib.pyplot as plt

def std(x, vec=True):
    if vec:
        x = np.vstack(x)
        return np.std(x, axis=0)
    else:
        return np.std(x)

def plot_std(y, std, x=None, c='C0', alpha=.3):
    log10e = math.log10(math.exp(1))
    if x is None:
        x = y.keys()
    rerr = log10e * std / y
    plt.fill_between(x, 10 ** (np.log10(y) - rerr), 10 ** (np.log10(y) + rerr), color=c, alpha=alpha)

def load_net_from_run(r, trained, device='cpu', shuffle_channels=False, mean_field_weights=0, best=1):
    args = r['args']
    args.device = device
    if best:
        lnet = r['best']['net']
    else:
        if r['dynamics'] is not None:
            lnet = r['dynamics'][-1]['net']
        else:
            lnet = r['last']
    state = OrderedDict([(k[7:], lnet[k]) for k in lnet if 'last_bias' not in k])
    if shuffle_channels:
    ## permute conv. layers channels
        for k in state:
            if len(state[k].shape) == 4:
                state[k] = state[k][:, torch.randperm(state[k].shape[1])]
                if shuffle_channels == 2:
                    perm = torch.randperm(state[k].shape[0])
                    state[k] = state[k][perm]
            # if 'running' in k and shuffle_channels == 2:
            #     state[k] = state[k][perm]
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

    if mean_field_weights == 1:
        net = mean_field_params(net)
    if mean_field_weights == 2:
        net = gaussian_params(net)
    net.eval()
    net = net.to(device)
    return net

def select_net(args):
    num_ch = 1 if 'mnist' in args.dataset or 'twopoints' in args.dataset else 3
    nc = 200 if 'tiny' in args.dataset else 10
    nc = 2 if 'diffeo' in args.dataset else nc
    num_classes = 1 if args.loss == 'hinge' else nc
    if 'mnist' in args.dataset:
        imsize = 28
    elif 'twopoints' in args.dataset:
        try:
            imsize = args.d
        except AttributeError:
            imsize = 28
    else:
        imsize = 32
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
        args.batch_norm
    except:
        args.batch_norm = 0
    try:
        args.width_factor
    except:
        args.width_factor = 1
    try:
        args.pretrained
    except:
        args.pretrained = 0
    try:
        args.last_bias
    except:
        args.last_bias = 0
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
        if args.net == 'ConvGAP':
            net = ConvNetGAPMF(n_blocks=args.depth, input_ch=num_ch, h=args.width, filter_size=args.filter_size,
                               stride=args.stride, pbc=args.pbc, out_dim=num_classes, batch_norm=args.batch_norm,
                               last_bias=args.last_bias)

    else:
        raise ValueError
    return net

def vgg_layer_names(net, li):
    cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],}
    i = 0
    layer_names = {}
    for k in cfg:
        for bn in ['', 'bn']:
            kbn = k + bn
            layer_names[kbn] = []
            for l in cfg[k]:
                if l == 'M':
                    layer_names[kbn].append('M')
                    i += 1
                else:
                    layer_names[kbn].append('Conv')
                    if bn == 'bn':
                        layer_names[kbn].append('BN')
                    layer_names[kbn].append('Relu')
                    i += 3 if bn == 'bn' else 2
            layer_names[kbn].append('A')
            layer_names[kbn].append('fl')
            layer_names[kbn].append('classifier')

    return layer_names[net][li]

def mean_field_params(f):
    for p in f.parameters():
        if len(p.shape) == 4:
            if p.shape[1] == 3:
                p.data = p.sum(dim=0, keepdim=True)
            else:
                p.data = p.sum(dim=[0, 1], keepdim=True)
        elif len(p.shape) == 1:
            p.data = p.sum(dim=0, keepdim=True)
        elif len(p.shape) == 2:
            p.data = p.sum(dim=[0, 1], keepdim=True)
    for c in f.modules():
        if 'batchnorm' in str(type(c)):
            c.running_mean = c.running_mean.mean(dim=0, keepdim=True)
            c.running_var = c.running_var.mean(dim=0, keepdim=True)
    return f

def gaussian_params(f):
    for p in f.parameters():
        pn = torch.randn(p.shape)
        if len(p.shape) == 4:
            mean = p.mean(dim=[0, 1], keepdim=True)
            std = p.std(dim=[0, 1], keepdim=True)
        elif len(p.shape) == 1:
            mean = p.mean(dim=0, keepdim=True)
            std = p.std(dim=0, keepdim=True)
        elif len(p.shape) == 2:
            mean = p.mean(dim=[0, 1], keepdim=True)
            std = p.std(dim=[0, 1], keepdim=True)
        p.data = pn * std + mean
    return f