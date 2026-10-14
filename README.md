# TurboDTI: Accelerating Diffusion-based Time Series Imputation via Consistency Model Sampling and Ensemble Distillation

Code and data supplement for the anonymous submission to The ACM Web Conference 2027 (WWW 2027), Web Infrastructure and Agentic Systems track.

TurboDTI accelerates diffusion-based time-series imputation for recurring data preparation in Web of Things (WoT) services. The code covers three diffusion-imputation backbones (CSDI, PriSTI, and Score-CDM) and three datasets (AQI, PeMS04, and PeMS08).

## Method overview

TurboDTI has two stages:

- **Stage 1: consistency-style (CS) sampling.** A frozen pretrained diffusion backbone uses shortened sampling paths selected by evolutionary sampling path optimization (ESPO), together with SNR-guided noise injection (SNR-NI). Stage 1 reduces both denoising steps and ensemble size without updating backbone parameters.
- **Stage 2: one-pass predictor distillation.** Ensemble predictions from DDPM and/or Stage 1 serve as training targets for a one-pass predictor. This moves repeated sampling and aggregation into offline preparation and reduces online inference to one network function evaluation.

## Repository layout

```text
TurboDTI/
├── CSDI-TurboDTI/
├── PriSTI-TurboDTI/
├── ScoreCDM-TurboDTI/
└── requirements.txt
```

Each backbone directory follows the same organization:

```text
<backbone>-TurboDTI/
├── AQI_datamaker/        # Stage 1 sampling and Stage 2 target preparation
├── PeMS04_datamaker/
├── PeMS08_datamaker/
├── AQI_student/          # Stage 2 one-pass predictor
├── PeMS04_student/
├── PeMS08_student/
├── config/               # Backbone and dataset configurations
├── data/                 # Dataset files expected by the loaders
├── save/                 # Included pretrained checkpoints, where available
└── ...                   # Backbone models and shared utilities
```

The package includes the AQI sample data used by the CSDI and PriSTI loaders and pretrained CSDI/PriSTI checkpoints. PeMS04 and PeMS08 data files must be placed in the paths expected by the corresponding dataset loaders, for example:

```text
CSDI-TurboDTI/data/PEMS04/pems04.h5
CSDI-TurboDTI/data/PEMS08/pems08.h5
PriSTI-TurboDTI/data/PEMS04/pems04.h5
PriSTI-TurboDTI/data/PEMS08/pems08.h5
```

Score-CDM additionally expects the dataset-specific adjacency matrices under its `data/PEMS04/` and `data/PEMS08/` directories.

## Environment

The recorded Python dependencies are listed in `TurboDTI/requirements.txt`.

```bash
cd TurboDTI
pip install -r requirements.txt
```

The experiment scripts use PyTorch Lightning and select CUDA devices in the trainer files. Adjust the device index, worker count, data paths, and checkpoint paths for the local environment before execution.

## Example workflow

The commands below use CSDI on PeMS04 as an example and are run from the `TurboDTI/` directory.

### Stage 1 evaluation

In `CSDI-TurboDTI/PeMS04_datamaker/pems04_trainer.py`, enable `trainer.test(...)` and disable `trainer.fit(...)`, then run:

```bash
python CSDI-TurboDTI/PeMS04_datamaker/pems04_trainer.py
```

### Stage 2 target preparation

Enable `trainer.fit(...)` in the same datamaker script. This runs the accelerated teacher and writes the target records consumed by the student data module:

```bash
python CSDI-TurboDTI/PeMS04_datamaker/pems04_trainer.py
```

### Stage 2 predictor

Select `trainer.fit(...)` for training or `trainer.test(...)` for evaluation in the student trainer, then run:

```bash
python CSDI-TurboDTI/PeMS04_student/pems04_student_trainer.py
```

The PriSTI and Score-CDM directories provide the corresponding dataset-specific datamaker and student scripts.

## Configuration notes

- Stage 1 sampling paths are defined in the dataset-specific datamaker scripts.
- Stage 2 target mixtures and loss weights are defined in the student trainer scripts.
- Trainer scripts contain experiment-specific train/test switches and checkpoint locations; review these values before use.
- The included checkpoints and AQI data are retained from the experimental package. Generated caches and local training logs are excluded from this WWW supplement.

## Citation

Citation information will be added after publication. During anonymous review, please refer to the accompanying WWW 2027 manuscript.
