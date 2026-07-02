# Alzheimer's Detection using CNN

A PyTorch pipeline that classifies brain MRI slices from the
[OASIS](https://www.oasis-brains.org/) dataset into four dementia stages
(*Non-Demented, Very Mild, Mild, Moderate*) with a small convolutional network.

## Why this refactor

The original Kaggle notebooks reported ~99.9% accuracy / AUC 1.0. That was an
artifact of three bugs, all fixed here:

1. **Patient leakage.** OASIS has many 2D slices per subject. The old code split
   at the *image* level, so slices of the same subject appeared in both train and
   test. We now split at the **subject** level (`GroupShuffleSplit`), so no
   subject crosses splits.
2. **Evaluation on training data.** The old "test" cells rebuilt a loader over the
   *entire* dataset. Evaluation now runs **only on the held-out test split**.
3. **Augmentation silently disabled.** `random_split` subsets share one dataset
   object, so reassigning `.dataset.transform` changed the transform for every
   split. Each split now owns its own transform.

Additional improvements: class-weighted loss for the heavy imbalance
(Non-Demented is ~78% of images), best-on-validation checkpointing with early
stopping, BatchNorm, and full seeding for reproducibility.

> Expect substantially lower (but honest) accuracy once the split is
> subject-disjoint — that is the number worth reporting.

## Project layout

```
src/
  config.py     # dataclass config, path/env resolution
  data.py       # subject-level split, per-split transforms, class weights
  model.py      # SimpleCNN + build_model (multi-GPU aware)
  train.py      # training loop, checkpointing, early stopping
  evaluate.py   # held-out metrics, confusion matrix, binary report
  utils.py      # seeding, device, checkpoint I/O
```

The original notebooks are kept for reference.

## Dataset

Expected as an `ImageFolder` tree with one directory per class:

```
<DATA_DIR>/
  Non Demented/       *.jpg
  Very mild Dementia/ *.jpg
  Mild Dementia/      *.jpg
  Moderate Dementia/  *.jpg
```

Point the pipeline at it via `--data-dir` or the `ALZ_DATA_DIR` environment
variable. On Kaggle the default `/kaggle/input/imagesoasis/Data` is auto-detected.

## Install

```bash
pip install -r requirements.txt
```

## Train

```bash
python -m src.train --data-dir /path/to/OASIS/Data --output-dir outputs --epochs 10
```

The best checkpoint (by validation loss) is written to
`<output-dir>/alzheimers_cnn_best.pth`.

## Evaluate

```bash
python -m src.evaluate --data-dir /path/to/OASIS/Data --checkpoint outputs/alzheimers_cnn_best.pth
```

Prints per-class and binary (Alzheimer's vs. Non-Demented) reports and saves
`confusion_matrix.png` to the output directory.

## Tests & leakage demo

```bash
pip install -r requirements-dev.txt
pytest                       # unit tests: subject-disjoint splits, per-split transforms, class weights
python -m scripts.leakage_demo   # empirical before/after proof of the leakage fix
```

The demo builds a synthetic dataset whose label is *arbitrary per subject* (so
it is unlearnable) and shows the old image-level split reporting ~90%+ (pure
memorization) while the subject-level split reports ~chance — the honest result.

## Notes

- Subject IDs are parsed from filenames with `Config.subject_id_regex`
  (default matches OASIS `OAS<n>_<id>`). Adjust it if your filenames differ;
  unmatched files fall back to an image-level split with a warning.
