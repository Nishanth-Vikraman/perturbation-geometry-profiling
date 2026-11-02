from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class ProbeModel(nn.Module):
    """Model interface exposing an internal probe state and a continuation map."""

    def probe(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def forward_from_probe(self, h: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class LinearClassifier(ProbeModel):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(64, 10)

    def forward(self, x):
        return self.fc(x.flatten(1))

    def probe(self, x):
        return x.flatten(1)

    def forward_from_probe(self, h):
        return self.fc(h)


class MLP(ProbeModel):
    def __init__(self, hidden=128):
        super().__init__()
        self.fc1 = nn.Linear(64, hidden)
        self.fc2 = nn.Linear(hidden, 64)
        self.out = nn.Linear(64, 10)

    def forward(self, x):
        h = self.probe(x)
        return self.forward_from_probe(h)

    def probe(self, x):
        return F.relu(self.fc1(x.flatten(1)))

    def forward_from_probe(self, h):
        return self.out(F.relu(self.fc2(h)))


class CNN(ProbeModel):
    def __init__(self, scrambled: bool = False, seed: int = 123):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.head = nn.Linear(32 * 4 * 4, 10)
        self.scrambled = scrambled
        g = torch.Generator().manual_seed(seed)
        self.register_buffer("perm", torch.randperm(64, generator=g))

    def _pre(self, x):
        if not self.scrambled:
            return x
        b = x.shape[0]
        return x.flatten(1)[:, self.perm].view(b, 1, 8, 8)

    def probe(self, x):
        return F.relu(self.conv1(self._pre(x)))

    def forward_from_probe(self, h):
        z = F.relu(self.conv2(h))
        z = F.max_pool2d(z, 2)
        return self.head(z.flatten(1))

    def forward(self, x):
        return self.forward_from_probe(self.probe(x))


class TinyViT(ProbeModel):
    def __init__(self, d_model=32, nhead=4, layers=2, patch=2):
        super().__init__()
        self.patch = patch
        self.patch_proj = nn.Conv2d(1, d_model, kernel_size=patch, stride=patch)
        n_patches = (8 // patch) ** 2
        self.cls = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos = nn.Parameter(torch.randn(1, n_patches + 1, d_model) * 0.02)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=64,
            batch_first=True, activation="gelu", dropout=0.0, norm_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=layers)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, 10)

    def probe(self, x):
        z = self.patch_proj(x).flatten(2).transpose(1, 2)
        cls = self.cls.expand(x.shape[0], -1, -1)
        return torch.cat([cls, z], dim=1) + self.pos

    def forward_from_probe(self, h):
        z = self.encoder(h)
        return self.head(self.norm(z[:, 0]))

    def forward(self, x):
        return self.forward_from_probe(self.probe(x))


def build_model(name: str, seed: int = 0) -> ProbeModel:
    key = name.lower()
    if key == "linear":
        return LinearClassifier()
    if key == "mlp":
        return MLP()
    if key == "cnn":
        return CNN(scrambled=False, seed=seed)
    if key in {"cnn_scrambled", "scrambled_cnn"}:
        return CNN(scrambled=True, seed=seed)
    if key in {"tinyvit", "vit"}:
        return TinyViT()
    raise ValueError(f"Unknown model: {name}")
