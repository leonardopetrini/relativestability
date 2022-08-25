import torch

def laplacian_proj(w):
    psi = laplacian_eigenvectors(N=w.shape[-1])
    wn = w.pow(2).sum(dim=[-1, -2]).sqrt()
        # print('Filtering weights fraction: ', 1 - (wn > 1e-3).float().mean().item())
    w = w[wn > 1e-4]
    return torch.einsum('iyx,jyx->ij', w, psi).pow(2).mean(0)

def laplacian_eigenvectors(N=3):
    """Compute the eigenvectors of the Laplacian on the NxN grid.
    :param int n: grid size
    :returns torch.Tensor: the N^2 eigenvectors of the grid Laplacian. Each eigenvector is a NxN matrix."""

    laplacian = torch.zeros(N**2, N**2)
    for i in range(N**2):
        x = i % N
        y = i // N
        n = 0
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            xx = x + dx
            yy = y + dy
            if 0 <= xx < N and 0 <= yy < N:
                laplacian[i, N * yy + xx] = -1
                n += 1
        laplacian[i, i] = n

    lambd, psi = torch.linalg.eigh(laplacian) # torch.symeig(laplacian, eigenvectors=True)
    return psi.T.reshape(N**2, N, N)

def extract_convolutional_weights(f):
    return [p.detach().flatten(0, -3) for p in f.parameters() if len(p.shape) == 4]

