# MedDefend: Securing Medical IoT With Adaptive Noise-Reduction-Based Adversarial Detection

Paper replication of **MedDefend**, a training-free adversarial detection framework for medical
image classifiers, applied to binary melanoma cancer classification with a ResNet-50 backbone.

> Reference: Joseph et al., *"MedDefend: Securing Medical IoT With Adaptive Noise-Reduction-Based
> Adversarial Detection,"* IEEE Internet of Things Journal, Vol. 12, No. 8, April 2025.

This repository is an independent replication built for study purposes. It is not affiliated
with the paper's authors.

## What this notebook does

MedDefend detects adversarial examples without any additional training. Given an input image, it:

1. Estimates image entropy in the region highlighted by **Grad-CAM** (the classifier's own
   attention map).
2. Injects an **adaptive noise level** (ANIM) chosen from the entropy value, using thresholds
   `H1`/`H2` and noise levels `N1 < N2 < N3 < N4`.
3. Denoises the noisy image with **t-RPCA** (tailored Robust PCA), which replaces the standard
   nuclear norm with a logarithmic Schatten-p norm to more aggressively suppress adversarial
   perturbations while preserving image structure.
4. Classifies both the original and the denoised image, repeating the process for **3 runs**.
   An input is flagged as adversarial if the two predictions disagree on at least one run.

The notebook reproduces the paper's classifier, detector, and evaluation pipeline against five
attacks (FGSM, BIM, PGD, C&W, DeepFool) and reports detection accuracy, precision, recall, and F1,
matching the structure of the paper's Table II.

## Repository contents

| File | Description |
|---|---|
| `MedDefend_Paper_Replication.ipynb` | Main notebook: data loading, classifier, t-RPCA, ANIM, detector, attacks, evaluation, and figures. |
| `best_meddefend_classifier.pth` | Pretrained ResNet-50 classifier checkpoint (add your own file, or train from scratch — see below). |
| `README.md` | This file. |

## Architecture summary

| Component | Detail |
|---|---|
| Backbone | ResNet-50, ImageNet-pretrained |
| Classification head | FC `2048 -> 512 -> 256 -> 128 -> 2`, ReLU activations, dropout `0.5` |
| Attention | Grad-CAM on the last convolutional block |
| Denoiser | t-RPCA, log-Schatten-p norm, solved via ADMM (`lambda=0.01`, `beta=0.1`, `p=0.5`, `max_iter=100`) |
| Noise thresholds | `H1=4.0`, `H2=6.0` (entropy) with `N1=10/255`, `N2=70/255`, `N3=110/255`, `N4=150/255` |
| Detection rule | 3 runs; adversarial if `C(X) != C(X_D)` on at least one run |

## Dataset

**Melanoma Cancer Image Dataset** (binary: benign / malignant), 13,900 images.
Source: [Kaggle - bhaveshmittal/melanoma-cancer-dataset](https://www.kaggle.com/datasets/bhaveshmittal/melanoma-cancer-dataset)

The dataset must be organized as:

```
melanoma_cancer_dataset/
├── train/
│   ├── benign/
│   └── malignant/
└── test/
    ├── benign/
    └── malignant/
```

## Setup

### Requirements

- Python 3.9+
- PyTorch and torchvision (CUDA build recommended for reasonable runtime)
- `torchattacks`, `opencv-python`, `matplotlib`, `seaborn`, `scikit-learn`, `scikit-image`,
  `pillow`, `pandas`, `tqdm`, `scipy`

All packages are also installed from within the notebook's setup cell.

### Option A: Local deployment

1. Download the dataset from Kaggle and extract it locally, or download it once via KaggleHub
   and reuse the local copy.
2. Open `MedDefend_Paper_Replication.ipynb` and run **Section 3A**, setting `DATA_ROOT` to the
   local dataset path.
3. Skip Section 3B and continue from Section 3C onward.

### Option B: Google Colab / Kaggle Notebooks

1. Upload `MedDefend_Paper_Replication.ipynb` to Colab, or open it directly in a Kaggle notebook
   (Kaggle notebooks already have API credentials available, which makes this path the simplest
   entry point).
2. Run **Section 3B**, which installs `kagglehub` and downloads the dataset automatically.
   - On Kaggle, no credential setup is required.
   - On Colab, either call `kagglehub.login()` interactively, or set the `KAGGLE_USERNAME` and
     `KAGGLE_KEY` environment variables with your own Kaggle API token before running the cell.
     **Never commit real credentials to the notebook or the repository.**
3. Continue from Section 3C onward.

### Using a pretrained checkpoint

If you already have a trained classifier checkpoint, place the `.pth` file in the working
directory (or update `MODEL_SAVE` in Section 5 to point to it) before running the training cell.
The notebook checks for an existing checkpoint at that path and loads it directly, reporting its
validation accuracy, instead of retraining from scratch. Delete or rename the checkpoint file if
you want to force a fresh training run.

## Running the notebook

Execute the sections in order:

1. **Environment setup** — imports, seeding, device detection.
2. **Dataset loading** — Section 3A or 3B, then 3C to build the DataLoaders.
3. **Classifier** — model definition and Grad-CAM hooks.
4. **Train / load classifier** — trains for up to 50 epochs, or loads an existing checkpoint.
5. **t-RPCA denoising** — ADMM solver implementing the paper's Eq. 4-11.
6. **Entropy estimation and ANIM** — adaptive noise injection logic.
7. **MedDefend detector** — combines CAM, ANIM, t-RPCA, and the 3-run detection rule.
8. **Adversarial attacks** — FGSM, BIM, PGD, C&W, DeepFool via `torchattacks`.
9. **Evaluation** — replicates Table II (attack success rate, detection accuracy, precision,
   recall, F1) per attack.
10. **Figures** — singular value analysis, noise-level tuning, and detection visualizations,
    replicating Figures 4, 6, and 8.
11. **Computational complexity** — replicates Table IV (FLOPs comparison).

## Target results (from the paper)

Skin Lesion Melanoma Cancer Image Dataset, binary classification, ResNet-50 backbone
(classifier accuracy = 91%):

| Attack | Success Rate % | Detection Accuracy % | Precision % | Recall % | F1 Score % |
|---|---|---|---|---|---|
| FGSM (eps=0.3) | 86.3 | 92.1 | 95.1 | 89.4 | 92.16 |
| PGD (eps=0.03) | 94.3 | 89.5 | 88.76 | 90.29 | 89.51 |
| C&W (k=0) | 97.2 | 91.6 | 91.24 | 92.55 | 91.89 |
| BIM (eps=0.1) | 90.2 | 92.7 | 93.6 | 92.26 | 92.93 |
| DeepFool | 98.1 | 94.4 | 94.32 | 95.13 | 94.72 |

Actual results from this replication may vary slightly depending on hardware, library versions,
and the specific train/test split obtained from the dataset source.

## Notes on this replication

- All parameters (noise thresholds, t-RPCA hyperparameters, attack epsilons, number of detection
  runs) are set to the values reported in the paper.
- The dataset loading section supports two independent, self-contained paths so the notebook can
  be run identically on a local machine, in Google Colab, or in a Kaggle notebook.
- No credentials are stored in this notebook. Supply your own Kaggle API token via
  `kagglehub.login()` or environment variables if using the Colab/Kaggle download path.

## Citation

If you use this replication, please cite the original paper:

```
Joseph et al., "MedDefend: Securing Medical IoT With Adaptive Noise-Reduction-Based
Adversarial Detection," IEEE Internet of Things Journal, Vol. 12, No. 8, April 2025.
```
