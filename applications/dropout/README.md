# Small MLP training with Montmory and Threefry dropout

[Français](README.fr.md) · [English](README.md)

This experiment asks whether a software PRNG optimized for TPU helps an actual training step. It adds a fixed 25% dropout to a small classifier. It changes no historical PRNG source, constants, rounds or logical Montmory geometry. Developed by Pascal Montmory for ENINCA CONSULTING; the repository [rights notice](../../LICENSE) applies.

**Status: native validation passed on one TPU v5 lite on 24 October 2026.** Pallas reduces Montmory step time by 5.10–6.27% against Montmory JAX; timings remain close to Threefry. [Results, raw data and limitations](validation/tpu-20261024/README.md) are retained separately from raw-generator measurements.

[Open in Colab — English](https://colab.research.google.com/github/pascalMontmory/montmory-prng-research/blob/main/applications/dropout/notebooks/Montmory_Dropout_TPU.en.ipynb) · [Ouvrir dans Colab — français](https://colab.research.google.com/github/pascalMontmory/montmory-prng-research/blob/main/applications/dropout/notebooks/Montmory_Dropout_TPU.ipynb). The notebooks embed the sources and their hashes; running them requires an actual TPU allocation on the reader’s account. The files can also be inspected [in English](notebooks/Montmory_Dropout_TPU.en.ipynb) and [in French](notebooks/Montmory_Dropout_TPU.ipynb).

## What runs

- A 32 → 128 → 128 → 4 MLP with ReLU, dropout after both hidden layers and SGD; 21,252 parameters, batch size 64, 150 updates, seeds 7, 19 and 42.
- A synthetic supervised task: 2,048 training examples and 512 held-out examples labelled by a fixed linear teacher. All paths use the same initial weights, examples and batch order for each seed. This is a learning smoke test, not a benchmark of model quality on real data.
- CPU: train the ordinary JAX implementations of Threefry and Montmory; interpret the Pallas kernels to verify masks, gradients and a complete update. CPU interpretation is not a TPU performance measurement.
- TPU: train all four paths, adding fused Threefry Pallas and Montmory Pallas dropout. The default tile is 32 × 128 for both; there is no tuning sweep or selection of the fastest trial.

The Pallas kernel generates and applies a mask in one call. Its custom reverse-mode derivative regenerates the same mask from the saved RNG parameters. Other parts of the training graph may still save activations. No total memory reduction is claimed without profiling. This prototype supports float32, a fixed dropout probability, one device and reverse-mode differentiation. It does not implement a general Flax layer, forward-mode AD or distributed training.

## Exact mask contract

Keep an activation when its uint32 word is below `0xC0000000`; scale kept values by `1 / 0.75`. The threshold is exact. This does not establish a new statistical quality guarantee for Montmory.

- **Threefry:** the 20-round Threefry2x32 implementation from the frozen kit, using global flattened indices below 2³². It is checked against `jax.random.bernoulli(..., p=0.75, mode="low")` with partitionable Threefry enabled. Keys are derived from the seed, step and layer with JAX `fold_in`.
- **Montmory:** unchanged T=2048, D=4096 stream. Activation `i` uses low32 of word `base + floor(i/2)` when even, high32 when odd. For a fixed workload, reserve `ceil(batch × width / 2)` words per layer. Layer `l` in step `s` starts at `2³² - 4097 + (2s+l) × stride`. Both hidden layers have the same width. The runner's limits keep these addresses within uint64. This is an explicitly defined application mapping; it is not a new mixing algorithm or proof of independent substreams.

Changing physical tiles preserves this mapping. Changing batch size/width changes the reservation stride and therefore subsequent masks. Cross-shape and multi-device mask invariance are not claimed. Every forward/backward pair must use the same parameters. Integer and mask equality is distinct from floating-point training equality across CPU, GPU and TPU. The received CPU and TPU runs are documented separately; no GPU training run of this model was measured.

## Local result

The initial CPU run passed 49,672 fused-mask element comparisons and 49,672 gradient-element checks, with two tile sizes, tails and four seeds. Montmory masks were checked against the independent scalar C reference. Threefry masks were checked against JAX's Bernoulli API. A complete MLP update matched its reference exactly for each engine on this CPU. Six additional schedule checks cover carries and large step numbers.

After 150 updates, held-out accuracy was 82.0–84.2% with Threefry and 81.6–83.2% with Montmory, from 21.3–27.9% initially. Both learned this simple task. These three runs establish neither superiority nor statistical equivalence of the generators. The complete initial verification and training run took about 18 seconds on this VPS, including CPU compilation. This is not a predicted TPU runtime.

The archived local report and learning curves are under [validation/cpu](validation/cpu). The [24 October native TPU results](validation/tpu-20261024/README.md) contain 1,800 measurements and 900 numerically identical JAX/Pallas loss pairs. Comparison with this earlier CPU check remains indicative, not a controlled hardware benchmark.

## Reproduce

Use Python with JAX and jaxlib 0.10.2, NumPy and a C compiler. For a fresh CPU environment:

```sh
python -m pip install 'jax==0.10.2'
python applications/dropout/run.py --backend cpu --out build/dropout-new-cpu
```

On an allocated TPU, install the [official JAX TPU dependencies](https://docs.jax.dev/en/latest/installation.html#google-cloud-tpu), pinned to this experiment's version:

```sh
python -m pip install 'jax[tpu]==0.10.2'
python applications/dropout/run.py --backend tpu --out build/dropout-new-tpu
```

The output directory must be new. A missing TPU fails explicitly. `report.json`, `steps.csv`, `learning_curves.csv` and a diagnostic archive record the execution. A failed run is not a performance result. For a second, still-small workload, use `--batch 256 --width 256` with a new output directory.

For Colab, generate self-contained French/English notebooks and the source kit:

```sh
python applications/dropout/build_notebook.py --out build/dropout-delivery
```

Upload `Montmory_Dropout_TPU.ipynb` (French) or `Montmory_Dropout_TPU.en.ipynb` (English), select a TPU runtime, and run the four code cells. Source hashes are checked before execution. Download diagnostics even if compilation fails. Building the notebook does not validate its native TPU kernels.

## Measurement and interpretation

Compile and warm all complete training-step pipelines before measurement. Randomize variant order within every step. Arrays are resident on the device; synchronize each timed step. Include forward pass, backward pass and parameter update, plus dispatch and synchronization. Exclude dataset creation, batch extraction, compilation and validation evaluation. This is a resident-data training-step benchmark, not whole-job time including input pipelines.

Compare Pallas and JAX loss sequences and final weights separately for each engine, with tolerances fixed in the code (rtol=2e-4, atol=2e-5). Different generators intentionally produce different masks. Report per-seed median step times and their ratios; a ratio above 1 means the Pallas version is faster. No independent repetitions beyond the three seed trajectories, confidence interval, energy measurement or large-model extrapolation is claimed. A tiny model can be dominated by launch overhead. Pallas can also prevent fusion available to ordinary JAX: a slowdown is a valid result.

The implementation follows the use case described in JAX's [Pallas PRNG documentation](https://docs.jax.dev/en/latest/pallas/tpu/prng.html): generating random values in the consuming kernel. The received run measures a Pallas gain over Montmory JAX for this graph, without establishing a general advantage over Threefry. [Analysis and JAX/Pallas distinction](validation/tpu-20261024/README.md#threefry-in-jax-and-pallas).
