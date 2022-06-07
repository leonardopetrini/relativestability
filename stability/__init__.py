import torch

def stability(o, op):
    """
    compute stability of the function `f` to perturbations `xp` of `x`
    :param o: output on original batch
    :param op: output on perturbed batch
    :return: stability
    """

    o = o.reshape(len(o), -1)
    op = op.reshape(len(o), -1)

    deno = torch.cdist(o, o).pow(2).mean().item() + 1e-10
    return (op - o).pow(2).mean(0).sum().item(), deno

# def stability(f, x, xp):
#     """
#     compute stability of the function `f` to perturbations `xp` of `x`
#     :param f: network function
#     :param x: original batch of image(s)
#     :param xp: perturbed batch of image(s)
#     :return: stability
#     """
#
#     with torch.no_grad():
#
#         f0 = f(x).detach().reshape(len(x), -1)
#         fn = f(xp).detach().reshape(len(x), -1)
#
#         deno = torch.cdist(f0, f0).pow(2).mean().item() + 1e-16
#
#         return (fn - f0).pow(2).mean(0).sum().item() / deno
