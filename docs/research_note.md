# Research note: Perturbation Geometry Profiling

## Question

Can the geometry of finite perturbation responses reveal whether a model architecture is naturally suited to a dataset, rather than merely capable of fitting it?

## Motivation

Validation accuracy confounds several possibilities:

- the data may contain weak or unstable signal;
- the model family may be structurally mismatched;
- the model may fit in-distribution data but rely on brittle geometry;
- several architectures may perform similarly while realizing the task very differently.

PGP treats a trained network as a perturbable computational system. It does **not** require semantic labels for hidden units.

## Perturbation hierarchy

For an input `x` and finite perturbation size `delta`, define the first-order response

\[
\mathcal T^{(1)}_i(x)
= f(x+\delta e_i)-f(x).
\]

The second-order finite interaction is

\[
\mathcal T^{(2)}_{ij}(x)
= f(x+\delta e_i+\delta e_j)
-f(x+\delta e_i)
-f(x+\delta e_j)
+f(x).
\]

Higher orders are possible in principle, but scale combinatorially and should usually be sampled or approximated.

The raw tensors are not the final object. PGP compresses them into a **Perturbation Geometry Signature**

\[
G = [r_{eff}, H_{int}, L_{locality}, A_{anisotropy}, R_{redundancy}, V_{context}, C_{conditioning}, ...].
\]

The exact metrics are domain-dependent.

## Internal perturbations

Let a model expose an internal state `h` and continuation map `g`, so that

\[
f(x)=g(h(x)).
\]

Intervene on coordinate `i`:

\[
\Delta_i y = g(h+\delta e_i)-g(h).
\]

The internal second-order interaction is

\[
\Delta^2_{ij} y
= g(h+\delta e_i+\delta e_j)
-g(h+\delta e_i)
-g(h+\delta e_j)
+g(h).
\]

Raw coordinates are not invariant to neuron permutation, rescaling, or representation rotation. Cross-model claims should therefore rely on aggregate quantities such as response spectra, effective rank, redundancy, interaction distributions, and context dependence.

## Digits proof-of-concept

The exploratory experiment compared five models on 8x8 handwritten digits:

| Model | Test accuracy |
|---|---:|
| Linear | 92.7% |
| MLP | 98.9% |
| CNN | 99.1% |
| CNN with scrambled pixels | 97.3% |
| Tiny Vision Transformer | 96.7% |

The key control was the scrambled CNN. It retained the same convolutional family while a fixed random permutation destroyed the correspondence between convolutional neighborhoods and the original image geometry.

### Input interaction geometry

| Model | Nearby / distant interaction strength | Interaction vs distance correlation |
|---|---:|---:|
| Linear | 0.99x | +0.08 |
| MLP | 1.09x | -0.07 |
| CNN | **8.35x** | **-0.81** |
| CNN-scrambled | **0.92x** | -0.01 |
| TinyViT | 1.72x | -0.13 |

The result suggests that the perturbation map was not merely recognizing a CNN parameterization. It was detecting whether the learned function actually organized interactions according to image-space locality.

### Interaction magnitude alone was insufficient

Approximate normalized pair interaction:

| Model | Pair interaction |
|---|---:|
| Linear | ~0 |
| MLP | 0.039 |
| CNN | 0.070 |
| CNN-scrambled | 0.069 |
| TinyViT | 0.162 |

CNN and scrambled CNN had similar total nonlinear interaction, but radically different spatial organization. **Where interaction occurs** carried more diagnostic information than total interaction magnitude.

### Internal geometry

Exploratory aggregate metrics:

| Model | Effective response rank | Pair interaction | Response redundancy | Sample dependence |
|---|---:|---:|---:|---:|
| MLP | 12.7 | 0.018 | 0.394 | 0.11 |
| CNN | 14.9 | 0.022 | 0.337 | 0.13 |
| CNN-scrambled | 17.5 | 0.024 | 0.340 | 0.11 |
| TinyViT | 48.4 | 0.080 | 0.155 | 0.76 |

The TinyViT built a much more high-dimensional and sample-dependent computation on this small task. That is not automatically evidence of mismatch, but it is a candidate diagnostic when combined with sample efficiency and robustness.

### Shift robustness

Average accuracy after a one-pixel translation:

| Model | Clean | Shifted |
|---|---:|---:|
| Linear | 92.7% | 43.3% |
| MLP | 98.9% | 46.1% |
| CNN | **99.1%** | **55.2%** |
| CNN-scrambled | 97.3% | 44.2% |
| TinyViT | 96.7% | 44.8% |

The model with the strongest spatial locality signature was also the most robust to spatial displacement.

### Low-data test

With only 300 training examples, three-seed exploratory results were:

| Model | Clean accuracy | Shifted accuracy |
|---|---:|---:|
| MLP | 94.6 +/- 1.1% | 43.9 +/- 2.9% |
| CNN | **94.4 +/- 0.6%** | **50.3 +/- 2.0%** |
| CNN-scrambled | 91.6 +/- 0.9% | 37.2 +/- 2.0% |
| TinyViT | 85.9 +/- 2.3% | 34.2 +/- 4.7% |

The central hypothesis is therefore:

> A suitable architecture may be characterized by stable computational complexity whose perturbational geometry matches the structure of the task.

Not minimum rank. Not minimum interaction. **The right complexity in the right directions.**

## Data-first extension

The more ambitious formulation estimates a task geometry before choosing an architecture:

\[
D \rightarrow G_D \rightarrow \text{candidate architectural primitives}.
\]

Since finite data cannot identify an arbitrary target function without assumptions, `G_D` should be estimated through a deliberately heterogeneous probe ensemble and bootstrap stability analysis.

A useful decomposition is

\[
\underbrace{D \rightarrow G_D}_{\text{what the problem appears to demand}}
\qquad\leftrightarrow\qquad
\underbrace{M \rightarrow G_M}_{\text{what the model constructed}}.
\]

A mismatch measure `d(G_D, G_M)` could become an architecture-diagnostic quantity.

## Falsification criteria

The idea weakens substantially if any of the following occurs:

1. PGP metrics mostly encode parameterization artifacts and fail under function-preserving reparameterizations.
2. Geometry does not predict robustness, sample efficiency, or OOD behavior beyond ordinary validation metrics.
3. Geometry is unstable across seeds or perturbation magnitudes for otherwise equivalent models.
4. The data-first signature depends strongly on the chosen probe family.
5. PGP cannot distinguish deliberately matched and mismatched controls across multiple domains.

## Next experiments

1. Reproduce the digits results with perturbation-size sweeps and confidence intervals.
2. Repeat on full MNIST / Fashion-MNIST / CIFAR-10 with matched-capacity models.
3. Add function-preserving neuron permutation and scaling controls.
4. Estimate data-first geometry from a heterogeneous probe ensemble.
5. Test whether geometry predicts which model wins in low-data and distribution-shift regimes.
6. Move to market data only after the controlled benchmarks survive.

## Provenance warning

The numerical tables above are the values from the original exploratory analysis that motivated this repository. They are stored in `reported_results.json`. They should be considered **reported exploratory observations until regenerated from a clean run** of this codebase.
