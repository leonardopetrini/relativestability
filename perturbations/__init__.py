import sys
try:
    sys.path.insert(0, '/home/lpetrini/git/diffeomorphism/')
except ModuleNotFoundError:
    print("Diffeo Module not found !!! "
                  "Find the Module @ https://github.com/pcsl-epfl/diffeomorphism.")

import math
import torch
from diff import deform
# from functools import partial
# from functorch import vmap

def diffeo_batch(imgs, delta=1, c=3, interp='linear'):
    n = imgs.shape[-1]
    T = typical_temperature(delta, c, n)
    return torch.stack([deform(i, T=T, cut=c, interp=interp) for i in imgs])
    # batched_deform = vmap(partial(deform, T=T, cut=c, interp=interp), randomness='different')
    # return batched_deform(imgs)

def noisy_batch(imgs, timgs, sigma=-1):
    """
    :param imgs: original images
    :param timgs: locally translated images / diffeo
    :param sigma: noise magnitude, if `-1` use the diffeo imgs.
    :return: original images, locally translated images, noisy images
    """
    if sigma == -1:
        sigma = (timgs - imgs).pow(2).sum([1, 2, 3], keepdim=True).sqrt()
    eta = torch.randn(imgs.shape, device=imgs.device)
    eta = eta / eta.pow(2).sum([1, 2, 3], keepdim=True).sqrt() * sigma
    nimgs = imgs + eta
    return nimgs

def typical_temperature(delta, cut, n):
    if isinstance(cut, (float, int)):
        log = math.log(cut)
    else:
        log = cut.log()
    return 4 * delta ** 2 / (math.pi * n ** 2 * log)
