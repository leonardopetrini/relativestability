import argparse
import os
import pickle
from grid import load
from filtering_function import filtering_args, filtering_data

import pandas as pd

from data import *
from perturbations import *
from stability import *
from utils import load_net_from_run

from torchvision.models.feature_extraction import get_graph_node_names
from torchvision.models.feature_extraction import create_feature_extractor

import sys
sys.path.insert(0, '/home/lpetrini/git/diffeo-sota/')
from main import test
from datasets import dataset_initialization

prefix = '/home/lpetrini/results/'
# prefix = '/scratch/izar/lpetrini/results/'

torch.set_default_dtype(torch.float64)

def execute(args):

    predicate = lambda a: filtering_args(a, args) and \
                          a['net'] == args['net'] and \
                          a['dataset'] == args['dataset'] and \
                          a['xi'] == args['xi'] and \
                          a['gap'] == args['gap'] and \
                          a['norm'] == args['norm'] and \
                          a['pbc'] == args['pbc'] and \
                          a['gaussian_corruption_std'] == args['gaussian_corruption_std']


    runs = load(args['filename'], pred_args=predicate, pred_run=filtering_data)
    print(f'Loaded {len(runs)} runs')

    if 'mnist' in args['dataset']:
        x, _ = load_mnist(p=args['P'], fashion='fashion' in args['dataset'], device=args['device'])
    elif 'cifar' in args['dataset']:
        x, _ = load_cifar(p=args['P'], device=args['device'],
                          gaussian_corruption_std=args['gaussian_corruption_std'] if args['corrupt_test'] else 0)
    elif 'twopoints' in args['dataset']:
        x, _ = load_twopoints(p=args['P'], seed=0, xi=args['xi'], imsize=args['d'], pbc=args['pbc'],
                              gap=args['gap'], norm=args['norm'], train=False,
                              shuffle=True, device=args['device'], bkg_noise=args['background_noise'],
                              labelling=args['labelling'])
    else:
        raise ValueError('Dataset not in the list!')

    x *= args['input_variance']

    if 'twopoints' in args['dataset'] and args['interp'] != 'nearest':
        xd, _ = load_twopoints(p=args['P'], seed=0, xi=args['xi'], imsize=args['d'], pbc=args['pbc'], gap=args['gap'], norm=args['norm'],
                               train=False, shuffle=True, device=args['device'], local_translations=1, bkg_noise=args['background_noise'],
                               labelling=args['labelling'])
    else:
        xd = diffeo_batch(x, args['delta'], args['cut'], args['interp'])
    xn = noisy_batch(x, xd, args['noise_magnitude'])

    df = pd.DataFrame()

    for r in runs:
        run_args = r['args']

        for tr in [0, 1] if args['init'] else [1]:
            f = load_net_from_run(r, tr, device=args['device'], shuffle_channels=args['shuffle_channels'], mean_field_weights=args['mf_weights'], best=args['best_net'])

            if args['by_layer']:
                nodes, _ = get_graph_node_names(f)
                nodes = [node for node in nodes if node not in ['x', 'size', 'view', 'squeeze', 'dropout'] and
                         'size' not in node and 'sub' not in node and
                         'floordiv' not in node and 'numel' not in node and
                         'pow' not in node]
                l = create_feature_extractor(f, return_nodes=nodes)

                with torch.no_grad():
                    o = l(x)
                    od = l(xd)
                    on = l(xn)
            else:
                with torch.no_grad():
                    o = {'output': f(x)}
                    od = {'output': f(xd)}
                    on = {'output': f(xn)}



            for i, k in enumerate(o):
                D, deno = stability(o[k], od[k], mean=args['mean_stability'])
                G, _ = stability(o[k], on[k], mean=args['mean_stability'])

                if args['corrupt_test']:
                    print('Re-computing test error...')
                    _, testloader, _, _ = dataset_initialization(run_args)
                    def boo(_, __):
                        return torch.zeros(5).sum()
                    acc = test(run_args, testloader, f, boo)
                else:
                    acc = None

                df = pd.concat([df, pd.DataFrame(args | {
                    'run_args': run_args,
                    'layer': k,
                    'li': i,
                    'trained': tr,
                    'acc': r['best']['acc'],
                    'epoch': r['best']['epoch'],
                    'acc_new': acc,
                    'D': D,
                    'G': G,
                    'deno': deno,
                }, index=[0])])

    return {
        'args': args,
        'df': df
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, required=True)

    parser.add_argument("--filename", type=str, required=False, default='none')
    parser.add_argument("--device", type=str, default='cuda')
    parser.add_argument("--P", type=int, default=100, help='number of probing points')
    parser.add_argument("--init", type=int, required=True, help='compute initialization')
    parser.add_argument("--init_samples", type=int, default=5, help='number of network init. samples')
    parser.add_argument("--by_layer", type=int, default=1, help='compute stabilities layer by layer')

    parser.add_argument("--dataset", type=str, required=True, help='dataset')
    parser.add_argument("--gaussian_corruption_std", type=float, default=0, help='images with background noise')
    parser.add_argument("--corrupt_test", type=float, default=0, help='corrupt also test images with background noise')
    parser.add_argument("--input_variance", type=float, default=1, help='variance of the datapoints')

    # TwoPoints
    parser.add_argument("--labelling", type=str, default='distance', help='labelling function')
    parser.add_argument("--xi", type=int, default=14, help='typical scale')
    parser.add_argument("--gap", type=int, default=2, help='classes gap')
    parser.add_argument("--norm", type=str, default='L2', help='classes gap')
    parser.add_argument("--d", type=int, default=28, help='input dimension')
    parser.add_argument("--pbc", type=int, default=0, help='periodic boundary cond.')
    parser.add_argument("--background_noise", type=float, default=0, help='datasets with background noise')


    parser.add_argument("--net", type=str, required=True, help='network architecture')
    parser.add_argument("--best_net", type=int, default=1, help='network architecture')
    parser.add_argument("--shuffle_channels", type=int, default=0, help='shuffle convolutional channels')
    parser.add_argument("--mf_weights", type=int, default=0, help='average the weights')
    parser.add_argument("--mean_stability", type=int, default=1, help='mean of median in stab. computation')

    # Transformations
    parser.add_argument("--delta", type=float, default=1, help='diffeo avg. pixel displacement')
    parser.add_argument("--cut", type=float, default=3, help='diffeo high freq. cut-off')
    parser.add_argument("--interp", type=str, default='linear', help='interpolation')

    parser.add_argument("--noise_magnitude", type=float, default=-1, help='noise magnitude. If -1 matches diffeo effective noise.')

    args = parser.parse_args().__dict__

    if args['filename'] == 'none':
        args['filename'] = args['output'][:-15]
    else:
        args['filename'] = prefix + args['filename']

    with open(args['output'], 'wb') as handle:
        pickle.dump(args, handle)
    try:
        data = execute(args)
        with open(args['output'], 'wb') as handle:
            pickle.dump(args, handle)
            pickle.dump(data, handle)
    except:
        os.remove(args['output'])
        raise


if __name__ == "__main__":
    main()