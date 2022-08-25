import torch

class GaussianNoiseCorruption(torch.nn.Module):
    """
        Add white noise to input image.
    """
    def __init__(self, mean=0., std=1.):
        super().__init__()
        self.std = std
        self.mean = mean

    def __call__(self, tensor):
        noise = torch.randn(tensor.size()).div(torch.tensor(tensor.shape[-3:]).prod().sqrt())
        return tensor + noise * self.std + self.mean

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)