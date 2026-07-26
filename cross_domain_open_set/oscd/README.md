# Unified Palm-Auth Cross-Domain Open-Set Benchmark

Merges the 10 original standalone baseline scripts (arcface, magface,
compnet, ppnet, palmbridge, ccnet, co3net, sf2net, convnext, dino) into
one project with a single config, one shared data pipeline, and one
runner (`benchmarking.py`) that loops every method over all 12
train/test settings and reports EER / Rank-1 tables.

## Structure

```
config.py         Global settings + one hyperparameter dict per method.
dataset.py         Data scanning, deterministic seeded ID splits, the 12
                    settings, shared augmentation, Single/Paired/Triplet
                    Dataset classes.
utils.py           Seeding, one canonical EER + Rank-1 evaluator,
                    checkpoint helpers, results-table builder.
model.py           One `Baseline` wrapper class per method (uniform
                    train_step / embed / state_dict interface) + REGISTRY.
benchmarking.py    Main driver.
models/            Architectures + losses, extracted programmatically
                    (via Python's `ast` module, not hand-retyped) from
                    the 10 original scripts -- so the actual network
                    code is unchanged from the originals.
```

## Same data split for every method

Every method reads the same cached, seeded `SPLITS_FILE`
(`config.SPLITS_FILE`, default `./palm_auth_openset_splits.json`). It's
generated once (first run) and reused after that, so **all 10 baselines
train and test on identical train/gallery/probe samples** for a given
setting -- this was verified by running `dataset.get_settings()` and
confirming byte-identical sample lists across methods before this
project was assembled.

## Augmentation fairness (baked into `config.py` defaults)

`config.METHODS[name]["augment_factor"]` implements the fairness scheme
worked out earlier in this conversation:

- Methods whose loss needs >1 view per image (SupCon pairs: ccnet,
  co3net, convnext, dino; triplets: sf2net) get `augment_factor=1` --
  their native view count already IS their augmentation budget, no
  extra dataset-length multiplication on top.
- Methods whose loss only needs 1 view (arcface, magface, palmbridge,
  compnet, ppnet) get `augment_factor=2`, so they land at the same ~2
  augmented-views/image/epoch as the SupCon methods.
- **sf2net is the one exception**: its triplet loss has an irreducible
  3-view floor per image, so `augment_factor=1` there means "no
  augmentation beyond what the loss structurally requires," not "equal
  total exposure to the other 9." This is unavoidable without changing
  sf2net's loss.
- convnext/dino were changed from their original 4-view
  `[base, aug1, aug2, aug3]` batch to a 2-view `[aug1, aug2]` batch (see
  `models/convnext.py` / `models/dino.py` + `model.py`'s
  `_BaselineArcSupConBackbone.train_step`), matching ccnet/co3net's
  paired structure exactly.

All other hyperparameters (batch size, epochs, lr, schedule, loss
weights, margins...) are each method's original published defaults --
unchanged.

## Setup

```bash
pip install -r requirements.txt
```

Then edit `config.py`:
- `DATA_ROOT` -> your `smartphone_data` folder (same layout the original
  scripts expected: `<subject_id>/roi_perspective/`,
  `<subject_id>/roi_scanner/`).
- `METHODS["arcface"]["pretrained_weights"]` -> path to your ArcFace
  ONNX backbone.
- `METHODS["magface"]["pretrained_weights"]` -> path to your MagFace
  checkpoint (optional -- falls back to random init with a warning if
  missing, same as the original script).

`convnext` downloads pretrained ConvNeXtV2-Tiny weights via `timm` on
first use; `dino` downloads DINOv2 ViT-S/14 via `torch.hub` -- both need
internet access the first time they run.

## Running

```bash
python benchmarking.py                              # everything: 10 methods x 12 settings
python benchmarking.py --methods arcface,ccnet        # subset of methods
python benchmarking.py --settings S_scanner,S_wet_text # subset of settings
python benchmarking.py --quick                        # smoke test: 1 epoch, eval every epoch
```

Per-`(setting, method)` results (checkpoint, eval score files,
`results.json`) go to `config.BASE_RESULTS_DIR/<setting_tag>/<method>/`.
Initial weights are cached once per method (keyed by number of training
classes) under `<BASE_RESULTS_DIR>/init_weights/`, so every setting for
a given method starts from the same initial weights -- controlling for
init-noise across settings.

At the end, two tables (rows = 12 settings, columns = 10 methods) are
written to `<BASE_RESULTS_DIR>/eer_table.csv`,
`<BASE_RESULTS_DIR>/rank1_table.csv`, and a combined
`<BASE_RESULTS_DIR>/summary.md`. Rank-1 in both tables corresponds to
each run's best-Rank-1 checkpoint (reloaded and re-evaluated at the end
of training), matching the "best performance based on R1" reporting the
project was built for.

## What's been verified in this sandbox

No GPU / real dataset was available here, so full training runs weren't
possible. What WAS verified with a synthetic dataset (matching the real
folder/filename layout) on CPU:

- `dataset.get_settings()` correctly produces all 12 settings with
  sane train/gallery/probe splits.
- `dataset.make_loader()` produces correctly-shaped batches for all
  three view modes (single / paired / triplet).
- Full `train_step` -> `eval` -> `state_dict` round trip runs
  correctly for compnet, ppnet, palmbridge, ccnet, co3net, sf2net.
- The full `benchmarking.py` orchestration (init-weight caching,
  training loop, best-checkpoint tracking, `results.json`) runs
  end-to-end for compnet.
- arcface/magface/convnext's dependency chains are correct: they fail
  at exactly the expected point (missing ONNX file / missing pretrained
  checkpoint with graceful fallback / no internet to `timm`'s hub) and
  not for any other reason.
- One real bug was caught and fixed this way: `models/sf2net.py` was
  missing a `Parameter` import from the extraction step.

Before a real run, it's worth doing a `--quick --methods <one>
--settings S_scanner` smoke test on your actual data/hardware first.
