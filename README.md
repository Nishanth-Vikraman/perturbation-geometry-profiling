# Perturbation Geometry Profiling (PGP)

A small research repo for testing whether **finite perturbation geometry** can diagnose how naturally a model architecture fits a dataset.

The core claim is deliberately narrower than generic interpretability:

> Architecture suitability may be reflected not only in perturbation magnitude, but in the **geometry of perturbational interactions**.

The initial proof-of-concept uses `sklearn.datasets.load_digits` (8x8 handwritten digits) and compares:

- linear classifier
- MLP
- CNN
- the same CNN after a fixed random permutation of pixels
- a tiny Vision Transformer

The scrambled CNN is the key control. It preserves the convolutional architecture while deliberately destroying the alignment between convolutional neighborhoods and real image-space locality.

## Core finite perturbations

For input coordinate `i`,

\[
R_i(x)=f(x+\delta e_i)-f(x).
\]

For a pair `(i,j)`,

\[
I_{ij}(x)=f(x+\delta e_i+\delta e_j)-f(x+\delta e_i)-f(x+\delta e_j)+f(x).
\]

The repo summarizes these maps with quantities such as:

- near-vs-far interaction ratio
- correlation between spatial distance and interaction strength
- normalized pair interaction strength
- effective response rank
- response redundancy
- sample dependence of perturbation responses
- robustness to one-pixel shifts

These are intended as **geometry summaries**, not as a single universal architecture score.

## Reported exploratory result

The originating exploratory run produced a striking locality signature:

| Model | Nearby / distant interaction | Interaction vs distance |
|---|---:|---:|
| Linear | 0.99x | +0.08 |
| MLP | 1.09x | -0.07 |
| CNN | **8.35x** | **-0.81** |
| CNN, scrambled pixels | **0.92x** | -0.01 |
| Tiny ViT | 1.72x | -0.13 |

The same CNN architecture therefore changed from a strongly local perturbation geometry to essentially no image-space locality when the input coordinate system was scrambled.

Those values are preserved in `reported_results.json` for provenance. They are **exploratory values, not certified reproduced outputs of this repository build**. Run the experiment again before relying on exact numbers.

## Setup

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
pip install -e .
```

## Run

From the repository root:

```bash
python experiments/digits_pgp.py
```

For a quicker smoke test:

```bash
python experiments/digits_pgp.py --fast --epochs 8 --skip-low-data
```

GPU is used automatically when available. Override with:

```bash
python experiments/digits_pgp.py --device cpu
```

Results are written to:

```text
outputs/digits_pgp.json
outputs/digits_pgp.csv
```

## What the experiment tests

A useful PGP diagnostic should distinguish at least three cases:

1. **Architecture fingerprint only** — trivial and not enough.
2. **Architecture actually exploiting matching task structure** — useful.
3. **Architecture capable of fitting the task but constructing unnecessarily complex or unstable geometry** — potentially useful for architecture selection.

The scrambled-CNN control targets case 2 directly.

## Repository layout

```text
.
├── README.md
├── pyproject.toml
├── requirements.txt
├── reported_results.json
├── docs/
│   ├── research_note.md
│   └── trading_extension.md
├── experiments/
│   └── digits_pgp.py
├── outputs/
└── src/pgp/
    ├── __init__.py
    ├── data.py
    ├── models.py
    ├── perturb.py
    └── train.py
```

## Important caveats

- Coordinate-level internal interventions are parameterization-dependent. Compare aggregate summaries, not neuron identities, across model families.
- Finite perturbation size matters. A robustness study should sweep `delta` / perturbation scale.
- The digits dataset is small and clean; success here does not establish usefulness for noisy financial data.
- A high effective rank is not inherently bad. The hypothesis is about **task-matched structure and stable complexity**, not minimal complexity.
- Data geometry does not uniquely identify an architecture. At best it constrains useful architectural primitives.

See `docs/research_note.md` for the research formulation and `docs/trading_extension.md` for the proposed market-data version.
