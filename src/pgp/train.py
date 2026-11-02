from __future__ import annotations

import random
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .data import shifted_images


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_model(model, loader, device, epochs=25, lr=1e-3, weight_decay=1e-4):
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    for _ in range(epochs):
        model.train()
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(x), y)
            loss.backward()
            opt.step()
    return model


@torch.no_grad()
def accuracy(model, loader, device):
    model.eval()
    correct = total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        pred = model(x).argmax(1)
        correct += (pred == y).sum().item()
        total += y.numel()
    return correct / max(total, 1)


@torch.no_grad()
def shifted_accuracy(model, loader, device):
    shifts = [(0, 1), (0, -1), (1, 0), (-1, 0)]
    vals = []
    model.eval()
    for dy, dx in shifts:
        correct = total = 0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            xs = shifted_images(x, dy, dx)
            pred = model(xs).argmax(1)
            correct += (pred == y).sum().item()
            total += y.numel()
        vals.append(correct / max(total, 1))
    return float(np.mean(vals))
