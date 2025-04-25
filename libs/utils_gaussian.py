import numpy as np
import torch

def sample_correlated_gaussian(rho=0.5, dim=20, batch_size=128, seed=1, cubic=False):
    torch.manual_seed(seed)
    x, eps = torch.chunk(torch.randn(batch_size, 2 * dim), 2, dim=1)
    y = rho * x + torch.sqrt(torch.tensor(1. - rho**2).float()) * eps

    if cubic is True:
        y = y ** 3

    return x, y

def rho_to_mi(dim, rho):
    return -0.5 * np.log(1 - rho**2) * dim

def mi_to_rho(dim, mi):
    return np.sqrt(1 - np.exp(-2.0 / dim * mi))

def mi_schedule(n_iter):
    mis = np.round(np.linspace(0.5, 5.5 - 1e-9, n_iter)) * 2.0
    return mis.astype(np.float32)

class GaussianDataset(torch.utils.data.Dataset):
    def __init__(self, true_mi, ds, batch_size, cubic=False):
        super().__init__()
        self.true_mi = true_mi
        self.ds = ds
        self.batch_size = batch_size
        self.cubic = cubic

    def __len__(self):
        return len(self.true_mi)

    def __getitem__(self, idx):
        rho = mi_to_rho(self.ds, self.true_mi[idx])
        return sample_correlated_gaussian(rho=rho, dim=self.ds, batch_size=self.batch_size, seed=idx, cubic=self.cubic)
