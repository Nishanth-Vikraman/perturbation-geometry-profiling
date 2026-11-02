from __future__ import annotations

import itertools
import math
import numpy as np
import torch


def _target_score(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return logits.gather(1, targets[:, None]).squeeze(1)


def _effective_rank(matrix: np.ndarray, eps: float = 1e-12) -> float:
    s = np.linalg.svd(matrix, compute_uv=False)
    s = s[s > eps]
    if len(s) == 0:
        return 0.0
    # Participation ratio on singular-value energy.
    e = s ** 2
    return float((e.sum() ** 2) / (np.square(e).sum() + eps))


def _mean_abs_cosine_rows(matrix: np.ndarray, max_rows: int = 128) -> float:
    if matrix.shape[0] < 2:
        return 0.0
    if matrix.shape[0] > max_rows:
        idx = np.linspace(0, matrix.shape[0] - 1, max_rows).astype(int)
        matrix = matrix[idx]
    n = np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-12
    z = matrix / n
    sim = np.abs(z @ z.T)
    iu = np.triu_indices(sim.shape[0], 1)
    return float(sim[iu].mean()) if len(iu[0]) else 0.0


@torch.no_grad()
def input_perturbation_geometry(
    model,
    x: torch.Tensor,
    device,
    delta: float = 0.25,
    max_samples: int = 64,
    pair_samples: int = 500,
    seed: int = 0,
):
    """Finite first/second-order perturbation geometry on 8x8 inputs.

    Metrics are computed on the model's baseline-predicted class logit, avoiding
    label leakage in the geometric probe itself.
    """
    rng = np.random.default_rng(seed)
    model.eval().to(device)
    x = x[:max_samples].to(device)
    base_logits = model(x)
    targets = base_logits.argmax(1)
    base = _target_score(base_logits, targets)

    first = []
    for i in range(64):
        xp = x.clone()
        r, c = divmod(i, 8)
        xp[:, 0, r, c] = torch.clamp(xp[:, 0, r, c] + delta, 0.0, 1.0)
        d = _target_score(model(xp), targets) - base
        first.append(d.cpu().numpy())
    first = np.stack(first, axis=1)  # samples x pixels

    all_pairs = [(i, j) for i in range(64) for j in range(i + 1, 64)]
    if pair_samples < len(all_pairs):
        choose = rng.choice(len(all_pairs), size=pair_samples, replace=False)
        pairs = [all_pairs[k] for k in choose]
    else:
        pairs = all_pairs

    inter_strength = []
    distances = []
    normalized = []
    for i, j in pairs:
        xp = x.clone()
        ri, ci = divmod(i, 8)
        rj, cj = divmod(j, 8)
        xp[:, 0, ri, ci] = torch.clamp(xp[:, 0, ri, ci] + delta, 0.0, 1.0)
        xp[:, 0, rj, cj] = torch.clamp(xp[:, 0, rj, cj] + delta, 0.0, 1.0)
        dij = _target_score(model(xp), targets) - base
        second = dij.cpu().numpy() - first[:, i] - first[:, j]
        a = float(np.mean(np.abs(second)))
        denom = float(np.mean((np.abs(first[:, i]) + np.abs(first[:, j])) / 2.0)) + 1e-12
        inter_strength.append(a)
        normalized.append(a / denom)
        distances.append(math.hypot(ri - rj, ci - cj))

    inter_strength = np.asarray(inter_strength)
    distances = np.asarray(distances)
    near = inter_strength[distances <= 1.5]
    far = inter_strength[distances >= 4.0]
    ratio = float(near.mean() / (far.mean() + 1e-12)) if len(near) and len(far) else float("nan")
    corr = float(np.corrcoef(distances, inter_strength)[0, 1]) if np.std(inter_strength) > 0 else 0.0

    return {
        "near_far_interaction_ratio": ratio,
        "interaction_distance_correlation": corr,
        "pair_interaction_ratio": float(np.mean(normalized)),
        "mean_first_order_abs": float(np.mean(np.abs(first))),
        "mean_second_order_abs": float(np.mean(inter_strength)),
    }


@torch.no_grad()
def internal_perturbation_geometry(
    model,
    x: torch.Tensor,
    device,
    max_samples: int = 32,
    unit_samples: int = 64,
    pair_samples: int = 128,
    perturb_scale: float = 0.5,
    seed: int = 0,
):
    """Intervene on sampled coordinates of an exposed internal probe state.

    Coordinate-level metrics are architecture-dependent, so downstream use should
    focus on aggregate quantities (rank, redundancy, interaction, context variance)
    rather than comparing individual unit identities across model families.
    """
    rng = np.random.default_rng(seed)
    model.eval().to(device)
    x = x[:max_samples].to(device)
    h = model.probe(x)
    flat = h.reshape(h.shape[0], -1)
    n_units = flat.shape[1]
    units = np.arange(n_units)
    if n_units > unit_samples:
        units = np.sort(rng.choice(n_units, size=unit_samples, replace=False))

    base_logits = model.forward_from_probe(h)
    targets = base_logits.argmax(1)
    base_target = _target_score(base_logits, targets)
    global_scale = float(flat.std().item()) + 1e-6
    delta = perturb_scale * global_scale

    # unit x sample x class response tensor
    response = []
    target_first = []
    for u in units:
        hp = flat.clone()
        hp[:, u] += delta
        logits = model.forward_from_probe(hp.reshape_as(h))
        response.append((logits - base_logits).cpu().numpy())
        target_first.append((_target_score(logits, targets) - base_target).cpu().numpy())
    response = np.stack(response, axis=0)
    target_first = np.stack(target_first, axis=0)

    # Aggregate response rank across all sample/class contexts.
    response_matrix = response.reshape(response.shape[0], -1)
    eff_rank = _effective_rank(response_matrix)
    redundancy = _mean_abs_cosine_rows(response_matrix)

    # Sample dependence: compare each unit's output-direction across samples.
    sample_dep = []
    for ui in range(response.shape[0]):
        R = response[ui]
        norms = np.linalg.norm(R, axis=1, keepdims=True) + 1e-12
        Z = R / norms
        sim = Z @ Z.T
        iu = np.triu_indices(sim.shape[0], 1)
        if len(iu[0]):
            sample_dep.append(1.0 - float(np.mean(np.abs(sim[iu]))))
    context_dependence = float(np.mean(sample_dep)) if sample_dep else 0.0

    pairs = list(itertools.combinations(range(len(units)), 2))
    if len(pairs) > pair_samples:
        choose = rng.choice(len(pairs), size=pair_samples, replace=False)
        pairs = [pairs[k] for k in choose]

    ratios = []
    for a, b in pairs:
        hp = flat.clone()
        hp[:, units[a]] += delta
        hp[:, units[b]] += delta
        logits = model.forward_from_probe(hp.reshape_as(h))
        dij = (_target_score(logits, targets) - base_target).cpu().numpy()
        second = dij - target_first[a] - target_first[b]
        num = np.mean(np.abs(second))
        den = np.mean((np.abs(target_first[a]) + np.abs(target_first[b])) / 2.0) + 1e-12
        ratios.append(num / den)

    return {
        "effective_response_rank": eff_rank,
        "pair_interaction_ratio": float(np.mean(ratios)) if ratios else 0.0,
        "response_redundancy": redundancy,
        "sample_dependence": context_dependence,
        "probe_units_total": int(n_units),
        "probe_units_sampled": int(len(units)),
    }
