# Self-supervised Pre-training Helps Retinal Disease Progression Modelling Most When Data Is Scarce

Code and derived results for the paper *"Self-supervised Pre-training Helps Retinal Disease Progression Modelling Most When Data Is Scarce"* (Nwabufo, Gervelmeyer, Müller, Berens; [arXiv:2609.12834](https://arxiv.org/abs/2609.12834)).

## Overview

Longitudinal imaging cohorts — the kind needed to model *when* a disease will progress rather than just whether it is present — are scarce and small. Cross-sectional data (one image per participant) is comparatively abundant. This project asks whether self-supervised pre-training on abundant cross-sectional data can close that gap for disease progression modelling.

We study this for age-related macular degeneration (AMD):

- **Pre-training data:** [NAKO] — a large cross-sectional cohort of retinal fundus images (~153k images).
- **Downstream task:** [AREDS] — a longitudinal cohort (8,784 eyes, 55,173 images) used to predict time-to-conversion to late AMD via deep survival analysis.

We evaluate:

- **In-house SSL encoders**, pre-trained on NAKO with three objectives: **SimCLR** (contrastive), **MAE** (masked-autoencoding), and **DINOv2** (self-distillation via LoRA adaptation).
- **Foundation models**: **RETFound** (domain-specific, retinal) and **DINOv2-LVD** (general-purpose, natural images).
- **Supervised baselines**: ResNet18 with random initialisation or from ImageNet weights.

Each encoder is paired with a lightweight Cox proportional-hazards survival head (linear or MLP), used either **frozen** or **fully fine-tuned**, and evaluated across labelled training set sizes from 100 to 32,250 examples.



## Repository structure

```
.
├── pretraining/          # SSL pre-training pipelines (SimCLR, MAE, DINOv2/LoRA) on NAKO
├── survival/             # Cox proportional-hazards survival head, training & evaluation
└── analysis/             # GAM-based statistical decomposition
```


