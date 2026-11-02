#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from pgp.data import load_digits_data, subset_dataset
from pgp.models import build_model
from pgp.perturb import input_perturbation_geometry, internal_perturbation_geometry
from pgp.train import accuracy, set_seed, shifted_accuracy, train_model

MODELS = ["linear", "mlp", "cnn", "cnn_scrambled", "tinyvit"]


def main():
    p = argparse.ArgumentParser(description="Digits proof-of-concept for Perturbation Geometry Profiling")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--output", default="outputs/digits_pgp.json")
    p.add_argument("--fast", action="store_true", help="Smaller perturbation sample counts")
    p.add_argument("--skip-low-data", action="store_true")
    args = p.parse_args()

    set_seed(args.seed)
    train_loader, test_loader, train_ds, test_ds = load_digits_data(seed=args.seed)
    x_test = test_ds.tensors[0]

    results = {}
    pair_samples = 180 if args.fast else 700
    internal_pairs = 48 if args.fast else 160
    max_samples = 24 if args.fast else 64

    for name in MODELS:
        set_seed(args.seed)
        model = build_model(name, seed=args.seed)
        train_model(model, train_loader, args.device, epochs=args.epochs)
        row = {
            "accuracy": accuracy(model, test_loader, args.device),
            "shifted_accuracy": shifted_accuracy(model, test_loader, args.device),
        }
        row["input_geometry"] = input_perturbation_geometry(
            model, x_test, args.device, max_samples=max_samples,
            pair_samples=pair_samples, seed=args.seed
        )
        row["internal_geometry"] = internal_perturbation_geometry(
            model, x_test, args.device, max_samples=min(max_samples, 32),
            pair_samples=internal_pairs, seed=args.seed
        )
        results[name] = row
        print(name, json.dumps(row, indent=2))

    if not args.skip_low_data:
        low_data = {}
        for name in ["mlp", "cnn", "cnn_scrambled", "tinyvit"]:
            clean, shifted = [], []
            for seed in [0, 1, 2]:
                set_seed(seed)
                subset = subset_dataset(train_ds, 300, seed)
                loader = DataLoader(subset, batch_size=64, shuffle=True)
                model = build_model(name, seed=seed)
                train_model(model, loader, args.device, epochs=max(args.epochs, 35))
                clean.append(accuracy(model, test_loader, args.device))
                shifted.append(shifted_accuracy(model, test_loader, args.device))
            low_data[name] = {
                "clean_mean": float(np.mean(clean)), "clean_std": float(np.std(clean)),
                "shifted_mean": float(np.mean(shifted)), "shifted_std": float(np.std(shifted)),
            }
        results["low_data_300"] = low_data

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))

    rows = []
    for name in MODELS:
        r = results[name]
        rows.append({
            "model": name,
            "accuracy": r["accuracy"],
            "shifted_accuracy": r["shifted_accuracy"],
            **{f"input_{k}": v for k, v in r["input_geometry"].items()},
            **{f"internal_{k}": v for k, v in r["internal_geometry"].items()},
        })
    csv_out = out.with_suffix(".csv")
    pd.DataFrame(rows).to_csv(csv_out, index=False)
    print(f"\nWrote {out} and {csv_out}")


if __name__ == "__main__":
    main()
