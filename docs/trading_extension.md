# Extending Perturbation Geometry Profiling to trading data

## Why this domain matters

In trading, poor out-of-sample performance is ambiguous. It can mean:

- the market state contains little stable predictive information;
- the feature representation is poor;
- the architecture is mismatched;
- the signal is real but too weak for the available sample size;
- the geometry changes by regime.

PGP is interesting here only if it helps separate those failure modes.

## Replace image geometry with market geometry

For images, the first proof-of-concept used Euclidean pixel distance. For market data, define pair metadata such as:

- temporal lag separation
- same vs different asset
- same vs different book level
- flow vs depth vs spread vs volatility feature type
- same event window vs cross-window interaction
- signed direction / side of book
- regime labels, if available only for analysis and not training

Then estimate whether second-order perturbation strength obeys structure such as

\[
|I_{ij}| \approx g(|\tau_i-\tau_j|)
\]

or whether interactions are instead state-dependent and nonlocal.

## Candidate task-geometry quantities

A market-oriented signature could include:

- **memory spectrum**: predictive contribution vs lag
- **interaction order**: additive, pairwise, or higher-order dependence
- **temporal locality**: whether strong interactions cluster in lag
- **cross-feature sparsity**: how many feature families interact materially
- **state dependence**: how much geometry changes with volatility/liquidity regime
- **effective dimension**: dimensionality of stable predictive directions
- **directionality**: causal/time-asymmetric propagation
- **geometry stability**: variation across months, assets, and bootstraps

## Data-first workflow

The preferred workflow is not to begin with a Transformer.

1. Construct a leak-safe supervised target and feature panel.
2. Split chronologically.
3. Train a heterogeneous set of weak probe estimators.
4. Estimate the same structural quantities from each probe.
5. Bootstrap / roll the estimation through time.
6. Keep only geometry that is stable across probe families and periods.
7. Translate stable geometry into architectural primitives.
8. Train candidate architectures and compare their learned geometry `G_M` with the data-first estimate `G_D`.

## Example architecture implications

If the data repeatedly indicates:

```text
low effective dimension
+ long causal memory
+ strong second-order flow x liquidity interaction
+ multiple decay time constants
```

then a plausible architecture is not necessarily a generic Transformer. It might be a small state-space core with multiplicative liquidity gating.

If instead it indicates:

```text
weak stationary locality
+ many nonlocal interactions
+ interaction graph strongly conditioned on current state
```

then attention-like dynamic coupling becomes more defensible.

The goal is therefore not

```text
PGP -> one uniquely correct model
```

but

```text
PGP -> constraints on useful computational primitives
```

## Crucial controls for finance

- No random train/test split.
- Perturbations must respect market-support constraints; independent feature perturbations can create impossible book states.
- Use conditional or manifold-respecting perturbations where needed.
- Report stability across time, not only average geometry.
- Keep probe models small enough that the geometry is not simply inherited from a giant universal approximator.
- Compare against null targets and time-shuffled controls.
- Test whether PGP predicts OOS ranking beyond ordinary CV score.

## A decisive first trading experiment

Take one existing ETH prediction panel and compare four families with similar parameter budgets:

- regularized MLP
- temporal CNN
- small SSM
- small attention model

For each:

1. measure OOS predictive performance;
2. measure input perturbation geometry by lag and feature type;
3. measure internal aggregate perturbation geometry;
4. evaluate geometry stability across rolling windows;
5. compare with the data-first consensus geometry;
6. test whether geometry mismatch predicts OOS degradation.

A useful positive result would be that geometry mismatch ranks architectures or failure modes **before** PnL differences become statistically obvious.
