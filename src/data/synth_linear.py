import torch
from torch.utils.data import Dataset

class SynthLinear(Dataset):
    """
    y = w·x + ε synthetic regression.
    """
    def __init__(self, n=1024, in_dim=10):
        super().__init__()
        self.x = torch.randn(n, in_dim)
        w_true = torch.randn(in_dim, 1)
        self.y = self.x @ w_true + 0.1 * torch.randn(n, 1)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]
