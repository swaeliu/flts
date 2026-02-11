import math
import random
import torch
from torch.utils.data import Dataset, DataLoader

def set_seed(s=0):
    random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)

def make_series(T, C, seed=0, client_id=0):
    g = torch.Generator().manual_seed(seed + 1000 * client_id)
    t = torch.linspace(0, 1, T)

    ys = []
    for c in range(C):
        # periodic trends 
        trend = (c + 1) * (0.6 + 0.05 * client_id) * t + (0.15 + 0.01 * client_id) * (t**2)
        f1 = 2 + (client_id % 5) + c
        f2 = 11 + 2 * (client_id % 7) + 2 * c
        season = 0.6 * torch.sin(2 * math.pi * f1 * t) + 0.2 * torch.sin(2 * math.pi * f2 * t)
        noise = (0.12 + 0.02 * (client_id % 3)) * torch.randn(T, generator=g)
        y = trend + season + noise
        ys.append(y)

    y = torch.stack(ys, 0)  # [C,T]
    y = (y - y.mean(1, keepdim=True)) / (y.std(1, keepdim=True) + 1e-6)
    return y

class WinDS(Dataset):
    def __init__(self, series_ct, start, end, L, H):
        self.s = series_ct
        self.L, self.H = L, H
        self.idxs = list(range(start, end - (L + H)))
        self.mask = torch.ones(L, dtype=torch.long)

    def __len__(self): 
        return len(self.idxs)

    def __getitem__(self, i):
        k = self.idxs[i]
        x = self.s[:, k:k + self.L]               # [C,L]
        y = self.s[:, k + self.L:k + self.L + self.H]  # [C,H]
        return x, y, self.mask

def collate(b):
    x, y, m = zip(*b)
    return torch.stack(x), torch.stack(y), torch.stack(m)  # [B,C,L],[B,C,H],[B,L]

def make_client_loaders(series, seq_len, horizon, batch_size, train_frac, val_frac):
    T = series.size(1)
    tr_end = int(train_frac * T)
    va_end = int((train_frac + val_frac) * T)

    train_ds = WinDS(series, 0, tr_end, seq_len, horizon)
    val_ds   = WinDS(series, tr_end, va_end, seq_len, horizon)
    test_ds  = WinDS(series, va_end, T, seq_len, horizon)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  collate_fn=collate)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, collate_fn=collate)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, collate_fn=collate)
    return train_loader, val_loader, test_loader
