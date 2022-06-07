import torch
import math

def power_spectrum_k(x, sigma=1 / 2 ** .5):
    """
    in:  [P, ch, N, N]
    out: [ch, K]
    """

    N = x.shape[-1]
    P = x.shape[0]
    ch = x.shape[1]

    fi = torch.fft.rfft2(x)
    power = fi.abs().pow(2)
    p = power.reshape(P, ch, -1)

    NN = torch.arange(N, dtype=x.dtype, device=x.device)
    j, i = torch.meshgrid(NN, NN[:len(NN) // 2 + 1])
    k_norm = (i.pow(2) + j.pow(2)).sqrt()
    k = k_norm.flatten()

    ks = torch.logspace(0, math.log10(N), N // 2, dtype=x.dtype, device=x.device)

    def f(k1, k2):
        return (-(k1 - k2) ** 2 / (2 * sigma ** 2)).exp()

    w = f(ks.reshape(-1, 1), k)  # [i,j]
    w /= w.sum(1, keepdim=True)

    ps = torch.einsum('ij,pcj->pci', w, p)
    ps = ps.mean(0)

    return ps