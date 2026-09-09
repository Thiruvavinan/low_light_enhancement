# `configs/` — one YAML per run

**Why this folder exists:** so that a run is fully described by a file that
can be diffed, rather than by a command line that has to be remembered. Every
hyperparameter lives here; `scripts/train.py` contains no defaults worth
overriding.

| file | stage | what it is |
|---|---|---|
| [`decom.yaml`](decom.yaml) | 1 | Decom-Net. Trained **once**, shared by both arms. |
| [`enhance_l1.yaml`](enhance_l1.yaml) | 2 | Baseline: the paper's L1 reconstruction loss. |
| [`enhance_ssim.yaml`](enhance_ssim.yaml) | 2 | Variant: L1 + SSIM reconstruction loss. |

```bash
python scripts/train.py --config configs/decom.yaml        # run this first
python scripts/train.py --config configs/enhance_l1.yaml
python scripts/train.py --config configs/enhance_ssim.yaml
```

## The control

The two stage-2 configs differ in **exactly two substantive places**:
`experiment.name`/`output_dir`, and `loss.ssim_weight`. That is checkable, and
it should be checked:

```bash
diff configs/enhance_l1.yaml configs/enhance_ssim.yaml
```

Three further things enforce the control beyond the diff:

- **One Decom-Net.** Both arms load the same `runs/decom/last.pth` and freeze
  it. If each arm trained its own decomposition, the comparison would measure
  two differently-decomposed models, not two losses.
- **Same seed, verified.** Both use `seed: 0`, so the two Enhance-Nets see
  identical crops and augmentations in identical order — checked by
  `python scripts/check_determinism.py`, not assumed. (An earlier version of
  the dataset drew crops from `np.random.default_rng()`, which ignores the
  seed entirely; that is exactly the kind of failure this check exists for.)
- **No early stopping.** A fixed 100-epoch budget for both. Stopping each arm
  at its own best validation loss would confound the loss comparison with
  training length — and the two arms' validation losses are not even on the
  same scale, since the SSIM arm has an extra non-negative term.

## Paper text vs. released code

Two loss weights differ between the paper and the authors' released
implementation. The configs follow the **released code**, because that is what
produced the published results, and both arms use the same values:

| term | paper | released code | here |
|---|:---:|:---:|:---:|
| invariable reflectance `λ_ir` | 0.001 | 0.01 | **0.01** |
| Enhance-Net smoothness | 1.0 | 3.0 | **3.0** |
| cross-reconstruction `λ_ij` | 0.001 | 0.001 | 0.001 |
| illumination smoothness `λ_is` | 0.1 | 0.1 | 0.1 |
| gradient sensitivity `λ_g` | 10 | 10 | 10 |

Reproducing the paper's stated values is a config edit, not a code edit.

## Overrides

Anything can be changed without editing a file, which is how sweeps and
ablations should be run so the committed configs stay canonical:

```bash
# λ_ssim sweep
python scripts/train.py --config configs/enhance_ssim.yaml \
    --set loss.ssim_weight=0.5 --set experiment.output_dir=runs/enhance_ssim_w0.5

# gradient diagnostics (see models/README.md)
python scripts/train.py --config configs/enhance_l1.yaml --set training.grad_log_every=50

# architecture ablation -- NOT part of the headline comparison
python scripts/train.py --config configs/enhance_l1.yaml --set model.norm=group
```
