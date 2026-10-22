# Asian option: Monte Carlo and randomized Sobol

[Français](README.fr.md) · [English](README.md) · [Home](../../README.md)

By Pascal Montmory for ENINCA CONSULTING. **CPU pilot executed; native TPU compilation and performance remain unvalidated.** This is not a validated banking pricing engine. It uses unchanged historical engines and the [Sobol32 prototype](../sobol/README.md). See the [rights notice](../../LICENSE) and [third-party notices](../../THIRD_PARTY_NOTICES.md).

[Open in Colab](https://colab.research.google.com/github/pascalMontmory/montmory-prng-research/blob/main/applications/asian_option/notebooks/Eninca_Asian_Option_TPU.en.ipynb) · [English notebook](notebooks/Eninca_Asian_Option_TPU.en.ipynb) · [Notebook français](notebooks/Eninca_Asian_Option_TPU.ipynb)

## Research question

How long does it take to reach a given error on a financial expectation? The program compares Monte Carlo using Threefry and Montmory with randomized Sobol quasi-Monte Carlo. Accuracy is measured on the price, not just the generator's throughput.

Two Asian calls share the same paths: **arithmetic** average and **geometric** average. The geometric case has an analytic reference and supplies the primary accuracy criterion. Arithmetic pricing uses a separate numerical reference with reported uncertainty.

## Model and reference

Risk-neutral Black–Scholes model: S0=100, K=100, r=0.03, sigma=0.20, maturity=1 year. Actual parameters are the archived float32 values, converted to float64 for reference calculations. There are 128 equally spaced observations from T/128 through T; **S0 is excluded from the average**. No dividends, Brownian bridge, control variate or parameter selection after observing outcomes.

Log increments use the exact geometric-Brownian-motion solution at the monitoring dates, not Euler discretization of price. Payoffs are discounted to time zero. For geometric average G and n=128:

```text
E[log G] = log S0 + (r - sigma²/2) T (n+1)/(2n)
Var(log G) = sigma² T (n+1)(2n+1)/(6n²)
```

The lognormal geometric price is approximately **5.11913** for this market. The formula is checked with independent numerical quadrature, Brownian covariance and the zero-volatility case. The n=1 case is also checked. Our original implementation is in [pricing.py](pricing.py).

## Compared methods

- Threefry JAX and Montmory JAX; TPU adds their Pallas variants with fixed 8 × 128 pair tiles.
- SciPy-supplied LMS+shift Sobol, 128 dimensions; JAX blocks of 256 points and Pallas blocks of 256 on TPU. Each replicate has a distinct scramble. We retain the first `2^m` points, including the scrambled net's point zero, with no skipping or thinning.
- A `threefry_native` comparator calls `jax.random.normal` directly. Its transformation contract differs from the other paths.
- Other paths map uint32 to `((word >> 9)+0.5)*2^-23`, then apply the inverse normal CDF. This float32 grid excludes 0 and 1 but bounds the tails and precision. It approximates the continuous model; increasing the number of paths does not eliminate this discretization bias.
- Montmory retains T=2048,D=4096 and base `2^32-4097`. Pair i uses the low and high halves of word base+i; Threefry uses partitionable JAX positions 2i and 2i+1. Flattened coordinates are ordered by path then date. Exported Sobol parameters, not the seed alone, define its stream.

## Timing and accuracy

Pilot: 512 and 2,048 paths, four fixed seeds, three timings per seed. Full protocol: 4,096, 16,384 and 65,536 paths, sixteen fixed seeds, five timings. Timing repetitions reuse the same sample: **they do not increase the number of statistical replicates**.

Complete functions are compiled and warmed before measurements. Inputs remain dynamic and resident; variant order is shuffled. Timing includes RNG, normal conversion, paths, payoffs, means and synchronization. Key/Sobol-table preparation and transfers, compilation, references and checks are excluded; preparation times are retained in `setup.csv`. Pallas may materialize intermediate arrays. This is not a cold-call latency measurement including initialization.

Integer arrays are compared to C, the JAX API or SciPy; every timed result is checked against NumPy float64 payoffs from the same coordinates (`atol=rtol=1e-4`). Correctness checks are not PRNG statistical-quality tests. Bitwise reproducibility covers integers, not floating-point prices across architectures.

The report gives geometric-estimate RMSE against the analytic value, and the first size **among tested sizes** reaching empirical errors 0.10, 0.05 or 0.02. It does not extrapolate a time when a target is unmet. This descriptive selection on the same replicates is neither an accuracy guarantee nor independent confirmation.

The arithmetic reference uses different seeds, a float64 transform and full 32-bit Sobol coordinates (a possible zero is mapped to the first bin midpoint). The pilot uses eight nets of 8,192 points; the full protocol uses sixteen nets of 32,768. Its 95% Student interval is computed **across independent nets**, not across points within a net. This is an estimated interval, not a certified bound. RMS differences to this reference are consequently not exact RMSE against the unknown price. No cloud-currency cost target is calculated.

## Retained CPU results

[Report](validation/cpu/report.json) · [Measurements](validation/cpu/measurements.csv) · [Independent estimates](validation/cpu/estimates.csv) · [Summary](validation/cpu/summary.csv).

![CPU pilot error versus time](validation/cpu/error-vs-time.png)

The small pilot favors Sobol on geometric-price error at these sizes. Four replicates and one market establish neither universal superiority, native TPU gains, nor better intrinsic quality of Montmory or Threefry. Summaries retain every engine, including unfavorable results.

## Reproduce

Python 3.11+ and a C compiler, from the repository root:

```bash
python -m pip install -r applications/stochastic/requirements-cpu.txt
python applications/asian_option/run.py --backend cpu --quick --out build/asian-new
# Add --verify-only for checks without a campaign.
```

On an allocated TPU, use the notebook or additionally install `jax[tpu]==0.10.2`, then select `--backend tpu`. Remove `--quick` for the full protocol. Each run needs a new output directory; no CPU fallback or automatic results upload. Notebooks use the reader's account and quotas. Sources and versions are hashed and recorded; Sobol parameters are exported as NPZ files.

## References

- [QuantLib Asian-option example](https://github.com/lballabio/QuantLib/blob/master/Examples/AsianOption/AsianOption.cpp); our code does not copy its engine.
- [SciPy Sobol: scrambling, powers of two and Joe–Kuo references](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.qmc.Sobol.html). Sobol and its tables are not ENINCA inventions.
- [JAX Pallas PRNG](https://docs.jax.dev/en/latest/pallas/tpu/prng.html).
