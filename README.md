# PSEE

This repository contains the implementation of **PSEE**, a point-supervised
temporal activity segmentation method for inertial sensor sequences. PSEE
uses sparse timestamp annotations to generate pseudo temporal boundaries and
then trains a temporal action localization detector.

## Datasets

The experiments use four wearable-sensor benchmarks:

- WEAR
- WetLab
- SBHAR
- RWHAR

Each dataset is evaluated with leave-one-subject-out (LOSO) splits. The input
is preprocessed inertial sensor features with a 50-sample window and 50%
overlap. One Gaussian-sampled point annotation is used for each training
action instance.

## Requirements

The code is intended for Linux with an NVIDIA GPU, CUDA, Python 3, and
PyTorch. The tested environment uses PyTorch 1.13.1. Install the usual
scientific Python packages, including:

```bash
pip install numpy pandas pyyaml scipy scikit-learn
```

The runners also require these upstream project trees to be available on the
machine:

```text
STAD-main/                 SlimSTAD-1D detector and dataset code
detr_e2e_tad/              PSEE localizer and sensor encoder
camera_baseline/actionformer/  ActionFormer implementation
camera_baseline/tridet/        TriDet implementation
```

Set their locations before running:

```bash
export STAD_PROJECT=/path/to/STAD-main
export DETR_PROJECT=/path/to/detr_e2e_tad
export SLIMSTAD_PYTHON=/path/to/python
```

Dataset features, annotations, and pretrained checkpoints must also be placed
according to the paths used by the selected runner.

## Main experiments

### PSEE + SlimSTAD-1D

For WEAR:

```bash
python psee_localizer/run_psee_lowparam_slimstad1d_wear_queue.py queue
```

For WetLab, SBHAR, and RWHAR:

```bash
python psee_localizer/run_psee_lowparam_slimstad1d_extra_queue.py queue \
  --datasets wetlab,sbhar,rwhar
```

The same script accepts `sbhar` and `rwhar`. Use `--help` to view split,
epoch, and smoke-test options.

### PSEE + ActionFormer / TriDet

```bash
python detectors/run_psee_actionformer_weighted.py --help
python detectors/run_psee_tridet_weighted.py --help
```

These runners consume the PSEE pseudo segments and keep the detector
inference pipeline unchanged.

## Verification

Check the experiment manifest and Python syntax with:

```bash
python experiments/check_main_experiment.py
python -m compileall psee_localizer slimstad1d detectors baselines tools
```

The experiment matrix and LOSO split counts are listed in
`experiments/main_experiment_manifest.json`.
