# Assignment 5 — ViT LoRA Fine-tuning + Adversarial Attacks

> **Course:** Deep Learning Operations  
> **GPU:** NVIDIA A30 (24 GB VRAM) | CUDA 12.2 | PyTorch 2.2.2  

---

## 🔗 Links

| Resource | Link |
|----------|------|
| 🟡 WandB — Q1 ViT Training | https://wandb.ai/m25csa014-iit-jodhpur/Assignment5_Q1_ViT_CIFAR100 |
| 🟡 WandB — Q1 Optuna Search | https://wandb.ai/m25csa014-iit-jodhpur/Assignment5_Q1_Optuna |
| 🟡 WandB — Q2(i) FGSM | https://wandb.ai/m25csa014-iit-jodhpur/Assignment5_Q2_FGSM |
| 🟡 WandB — Q2(ii) Detection | https://wandb.ai/m25csa014-iit-jodhpur/Assignment5_Q2_Detection |
| 🤗 HuggingFace — Model Weights | https://huggingface.co/Jaishankar02/assignment5-models |

---

## 📁 Repository Structure

```
Assignment5/
├── Dockerfile                # Docker container setup
├── requirements.txt          # Python dependencies
├── q1_train.py               # Q1 – ViT-S fine-tuning (no LoRA + all 9 LoRA combos)
├── q1_optuna.py              # Q1 – Optuna LoRA hyperparameter search (20 trials)
├── q2_fgsm.py                # Q2(i) – ResNet-18 training + FGSM (scratch & ART)
├── q2_detection.py           # Q2(ii) – ResNet-34 adversarial detectors (PGD, BIM)
├── checkpoints/
│   ├── q1/                   # ViT-S weights for all 10 experiments
│   ├── q1_optuna/            # Best Optuna model weights
│   └── q2/                   # ResNet-18 clean + ResNet-34 detectors
├── data/                     # Datasets (auto-downloaded)
└── outputs/                  # FGSM visual comparisons
```

---

## ⚙️ Installation

### Option A — Docker (Required by assignment)

```bash
# Build the image
docker build -t ass5 .

# Run with GPU (mount volumes so data/weights persist)
docker run --gpus all -it \
  -v $(pwd)/checkpoints:/app/checkpoints \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/outputs:/app/outputs \
  ass5 bash
```

### Option B — Local

```bash
pip install -r requirements.txt
# On Debian/Ubuntu managed environments:
pip install -r requirements.txt --break-system-packages
```

---

## 🚀 Running the Code

### Q1 — ViT-S Fine-tuning on CIFAR-100

**Step 1: Train all experiments (no LoRA baseline + 9 LoRA combinations)**

```bash
python q1_train.py \
  --wandb_key YOUR_WANDB_KEY \
  --hf_token  YOUR_HF_TOKEN \
  --hf_repo   Jaishankar02/assignment5-models
```

To skip the no-LoRA baseline:
```bash
python q1_train.py --skip_no_lora --wandb_key YOUR_WANDB_KEY
```

**Step 2: Optuna hyperparameter search**

```bash
python q1_optuna.py \
  --n_trials  20 \
  --wandb_key YOUR_WANDB_KEY \
  --hf_token  YOUR_HF_TOKEN \
  --hf_repo   Jaishankar02/assignment5-models
```

---

### Q2(i) — FGSM Attack (Scratch vs IBM ART)

```bash
python q2_fgsm.py \
  --epochs    30 \
  --wandb_key YOUR_WANDB_KEY
```

To skip training and load saved weights:
```bash
python q2_fgsm.py --skip_train --wandb_key YOUR_WANDB_KEY
```

---

### Q2(ii) — Adversarial Detection (ResNet-34)

> ⚠️ Must run `q2_fgsm.py` first — needs `checkpoints/q2/resnet18_clean.pt`

```bash
python q2_detection.py \
  --epochs    20 \
  --eps       0.03137 \
  --wandb_key YOUR_WANDB_KEY
```

---

## 📊 Q1 Results — ViT-S on CIFAR-100

### Experiment 0: No LoRA (Head Only)

**Trainable Parameters: 38,500 | Test Accuracy: 81.29%**

| Epoch | Train Loss | Val Loss | Train Acc (%) | Val Acc (%) |
|-------|-----------|---------|--------------|------------|
| 1  | 1.0793 | 0.7020 | 72.36 | 79.38 |
| 2  | 0.6349 | 0.6637 | 81.14 | 80.20 |
| 3  | 0.5613 | 0.6386 | 83.07 | 80.70 |
| 4  | 0.5182 | 0.6449 | 84.21 | 80.92 |
| 5  | 0.4890 | 0.6374 | 85.00 | 81.24 |
| 6  | 0.4636 | 0.6230 | 85.70 | 81.40 |
| 7  | 0.4347 | 0.6194 | 86.50 | 81.34 |
| 8  | 0.4207 | 0.6143 | 87.10 | 82.08 |
| 9  | 0.4101 | 0.6169 | 87.41 | 81.42 |
| 10 | 0.4021 | 0.6112 | 87.65 | 82.06 |

---

### Experiment 1: LoRA r=2, α=2, dropout=0.1

**Trainable Parameters: 75,364 | Test Accuracy: 89.76%**

| Epoch | Train Loss | Val Loss | Train Acc (%) | Val Acc (%) |
|-------|-----------|---------|--------------|------------|
| 1  | 0.7348 | 0.4055 | 80.83 | 87.36 |
| 2  | 0.3345 | 0.3640 | 89.44 | 88.46 |
| 3  | 0.2691 | 0.3716 | 91.33 | 88.38 |
| 4  | 0.2269 | 0.3789 | 92.64 | 88.82 |
| 5  | 0.1940 | 0.3742 | 93.66 | 88.58 |
| 6  | 0.1675 | 0.3742 | 94.53 | 89.12 |
| 7  | 0.1495 | 0.3690 | 95.08 | 88.96 |
| 8  | 0.1324 | 0.3640 | 95.72 | 89.22 |
| 9  | 0.1217 | 0.3664 | 96.23 | 89.42 |
| 10 | 0.1154 | 0.3577 | 96.48 | 89.34 |

---

### Experiment 2: LoRA r=2, α=4, dropout=0.1

**Trainable Parameters: 75,364 | Test Accuracy: 89.65%**

| Epoch | Train Loss | Val Loss | Train Acc (%) | Val Acc (%) |
|-------|-----------|---------|--------------|------------|
| 1  | 0.7302 | 0.4158 | 80.78 | 87.10 |
| 2  | 0.3304 | 0.3784 | 89.64 | 88.06 |
| 3  | 0.2649 | 0.3769 | 91.51 | 87.78 |
| 4  | 0.2223 | 0.3781 | 92.76 | 88.68 |
| 5  | 0.1874 | 0.3714 | 93.75 | 88.86 |
| 6  | 0.1597 | 0.3689 | 94.77 | 88.86 |
| 7  | 0.1386 | 0.3759 | 95.55 | 88.72 |
| 8  | 0.1234 | 0.3706 | 96.22 | 89.12 |
| 9  | 0.1135 | 0.3610 | 96.47 | 89.38 |
| 10 | 0.1063 | 0.3544 | 96.90 | 89.16 |

---

### Experiment 3: LoRA r=2, α=8, dropout=0.1

**Trainable Parameters: 75,364 | Test Accuracy: 89.96%**

| Epoch | Train Loss | Val Loss | Train Acc (%) | Val Acc (%) |
|-------|-----------|---------|--------------|------------|
| 1  | 0.7131 | 0.4154 | 81.14 | 87.36 |
| 2  | 0.3327 | 0.3901 | 89.65 | 88.26 |
| 3  | 0.2685 | 0.3789 | 91.30 | 88.56 |
| 4  | 0.2235 | 0.3787 | 92.70 | 88.26 |
| 5  | 0.1894 | 0.3844 | 93.76 | 87.92 |
| 6  | 0.1601 | 0.3783 | 94.71 | 88.84 |
| 7  | 0.1375 | 0.3803 | 95.50 | 88.48 |
| 8  | 0.1192 | 0.3618 | 96.28 | 88.70 |
| 9  | 0.1075 | 0.3663 | 96.72 | 89.16 |
| 10 | 0.1017 | 0.3771 | 97.00 | 88.50 |

---

### Experiment 4: LoRA r=4, α=2, dropout=0.1

**Trainable Parameters: 112,228 | Test Accuracy: 89.69%**

| Epoch | Train Loss | Val Loss | Train Acc (%) | Val Acc (%) |
|-------|-----------|---------|--------------|------------|
| 1  | 0.7337 | 0.3941 | 80.78 | 87.32 |
| 2  | 0.3287 | 0.3785 | 89.73 | 88.08 |
| 3  | 0.2603 | 0.3715 | 91.62 | 88.68 |
| 4  | 0.2197 | 0.3631 | 92.74 | 88.48 |
| 5  | 0.1841 | 0.3788 | 93.98 | 88.78 |
| 6  | 0.1593 | 0.3593 | 94.77 | 89.10 |
| 7  | 0.1375 | 0.3721 | 95.59 | 88.96 |
| 8  | 0.1217 | 0.3601 | 96.14 | 89.14 |
| 9  | 0.1105 | 0.3748 | 96.70 | 88.92 |
| 10 | 0.1036 | 0.3706 | 96.88 | 89.16 |

---

### Experiment 5: LoRA r=4, α=4, dropout=0.1

**Trainable Parameters: 112,228 | Test Accuracy: 90.07%**

| Epoch | Train Loss | Val Loss | Train Acc (%) | Val Acc (%) |
|-------|-----------|---------|--------------|------------|
| 1  | 0.7143 | 0.3976 | 81.17 | 87.30 |
| 2  | 0.3263 | 0.3762 | 89.69 | 88.32 |
| 3  | 0.2565 | 0.3751 | 91.70 | 88.66 |
| 4  | 0.2109 | 0.3718 | 93.07 | 88.46 |
| 5  | 0.1759 | 0.3777 | 94.21 | 88.40 |
| 6  | 0.1459 | 0.3769 | 95.20 | 88.54 |
| 7  | 0.1270 | 0.3733 | 95.89 | 89.22 |
| 8  | 0.1080 | 0.3760 | 96.58 | 88.86 |
| 9  | 0.0998 | 0.3719 | 96.92 | 88.72 |
| 10 | 0.0921 | 0.3735 | 97.27 | 88.84 |

---

### Experiment 6: LoRA r=4, α=8, dropout=0.1

**Trainable Parameters: 112,228 | Test Accuracy: 89.97%**

| Epoch | Train Loss | Val Loss | Train Acc (%) | Val Acc (%) |
|-------|-----------|---------|--------------|------------|
| 1  | 0.7129 | 0.3997 | 80.99 | 87.32 |
| 2  | 0.3196 | 0.3823 | 89.86 | 87.94 |
| 3  | 0.2542 | 0.3736 | 91.68 | 88.42 |
| 4  | 0.2074 | 0.3712 | 93.24 | 88.76 |
| 5  | 0.1723 | 0.3738 | 94.34 | 89.18 |
| 6  | 0.1383 | 0.3827 | 95.60 | 88.70 |
| 7  | 0.1163 | 0.3613 | 96.28 | 89.38 |
| 8  | 0.0987 | 0.3735 | 96.95 | 89.10 |
| 9  | 0.0896 | 0.3722 | 97.39 | 89.54 |
| 10 | 0.0815 | 0.3698 | 97.70 | 89.34 |

---

### Experiment 7: LoRA r=8, α=2, dropout=0.1

**Trainable Parameters: 185,956 | Test Accuracy: 90.01%**

| Epoch | Train Loss | Val Loss | Train Acc (%) | Val Acc (%) |
|-------|-----------|---------|--------------|------------|
| 1  | 0.7283 | 0.4067 | 80.98 | 87.52 |
| 2  | 0.3281 | 0.3867 | 89.69 | 88.04 |
| 3  | 0.2589 | 0.3829 | 91.60 | 88.42 |
| 4  | 0.2158 | 0.3652 | 93.06 | 89.14 |
| 5  | 0.1824 | 0.3766 | 94.00 | 88.62 |
| 6  | 0.1550 | 0.3748 | 94.91 | 89.04 |
| 7  | 0.1316 | 0.3680 | 95.71 | 88.92 |
| 8  | 0.1174 | 0.3662 | 96.29 | 89.18 |
| 9  | 0.1071 | 0.3623 | 96.77 | 89.04 |
| 10 | 0.1010 | 0.3726 | 96.97 | 89.14 |

---

### Experiment 8: LoRA r=8, α=4, dropout=0.1

**Trainable Parameters: 185,956 | Test Accuracy: 90.28%**

| Epoch | Train Loss | Val Loss | Train Acc (%) | Val Acc (%) |
|-------|-----------|---------|--------------|------------|
| 1  | 0.7165 | 0.3932 | 81.01 | 87.78 |
| 2  | 0.3262 | 0.3784 | 89.82 | 88.46 |
| 3  | 0.2526 | 0.3766 | 91.76 | 88.36 |
| 4  | 0.2079 | 0.3838 | 93.23 | 88.56 |
| 5  | 0.1734 | 0.3781 | 94.33 | 88.86 |
| 6  | 0.1402 | 0.3796 | 95.39 | 88.92 |
| 7  | 0.1170 | 0.3711 | 96.32 | 89.22 |
| 8  | 0.1012 | 0.3686 | 96.84 | 89.26 |
| 9  | 0.0924 | 0.3781 | 97.16 | 89.16 |
| 10 | 0.0839 | 0.3656 | 97.60 | 89.22 |

---

### Experiment 9: LoRA r=8, α=8, dropout=0.1 ⭐ BEST

**Trainable Parameters: 185,956 | Test Accuracy: 90.44%**

| Epoch | Train Loss | Val Loss | Train Acc (%) | Val Acc (%) |
|-------|-----------|---------|--------------|------------|
| 1  | 0.6957 | 0.4076 | 81.55 | 87.38 |
| 2  | 0.3169 | 0.3671 | 90.00 | 88.88 |
| 3  | 0.2496 | 0.3675 | 91.99 | 89.14 |
| 4  | 0.1949 | 0.3732 | 93.61 | 88.72 |
| 5  | 0.1555 | 0.3743 | 94.88 | 88.92 |
| 6  | 0.1235 | 0.3777 | 95.94 | 89.18 |
| 7  | 0.1000 | 0.3602 | 96.81 | 89.30 |
| 8  | 0.0834 | 0.3718 | 97.49 | 89.46 |
| 9  | 0.0729 | 0.3741 | 97.92 | 89.30 |
| 10 | 0.0678 | 0.3737 | 98.07 | 89.58 |

---

### Q1 Overall Test Accuracy Summary

| Exp | LoRA | Rank | Alpha | Dropout | Test Acc (%) | Trainable Params |
|-----|------|------|-------|---------|-------------|-----------------|
| 0  | Without LoRA | N/A | N/A | N/A | 81.29 | 38,500 |
| 1  | With LoRA | 2 | 2 | 0.1 | 89.76 | 75,364 |
| 2  | With LoRA | 2 | 4 | 0.1 | 89.65 | 75,364 |
| 3  | With LoRA | 2 | 8 | 0.1 | 89.96 | 75,364 |
| 4  | With LoRA | 4 | 2 | 0.1 | 89.69 | 112,228 |
| 5  | With LoRA | 4 | 4 | 0.1 | 90.07 | 112,228 |
| 6  | With LoRA | 4 | 8 | 0.1 | 89.97 | 112,228 |
| 7  | With LoRA | 8 | 2 | 0.1 | 90.01 | 185,956 |
| 8  | With LoRA | 8 | 4 | 0.1 | 90.28 | 185,956 |
| **9** | **With LoRA** | **8** | **8** | **0.1** | **90.44 ⭐** | **185,956** |

---

## 🔍 Q1 Optuna Hyperparameter Search

**Search Space:** rank ∈ {2,4,8,16}, alpha ∈ {2,4,8,16}, dropout ∈ [0.0, 0.30], lr ∈ [1e-4, 5e-3]  
**Sampler:** TPE | **Trials:** 20 | **Epochs per trial:** 5

| Trial | Rank | Alpha | Dropout | LR | Val Acc (%) |
|-------|------|-------|---------|-----|------------|
| 0  | 4  | 16 | 0.20 | 0.001596 | 89.54 |
| 1  | 4  | 16 | 0.15 | 0.000312 | 88.64 |
| 2  | 2  | 4  | 0.20 | 0.000120 | 87.00 |
| 3  | 16 | 2  | 0.20 | 0.000560 | 89.32 |
| 4  | 16 | 4  | 0.15 | 0.000206 | 88.22 |
| 5  | 2  | 4  | 0.00 | 0.000357 | 88.88 |
| 6  | 8  | 16 | 0.00 | 0.004750 | 89.00 |
| 7  | 16 | 8  | 0.10 | 0.000157 | 88.68 |
| 8  | 2  | 8  | 0.30 | 0.000634 | 89.42 |
| 9  | 8  | 2  | 0.00 | 0.000153 | 87.56 |
| 10 | 4  | 16 | 0.30 | 0.001972 | 89.22 |
| 11 | 2  | 8  | 0.30 | 0.001213 | 89.24 |
| 12 | 4  | 8  | 0.25 | 0.001120 | 89.68 |
| 13 | 4  | 16 | 0.25 | 0.001602 | 88.86 |
| 14 | 4  | 8  | 0.20 | 0.003182 | 89.16 |
| 15 | 4  | 16 | 0.10 | 0.001033 | 89.32 |
| 16 | 4  | 8  | 0.25 | 0.002263 | 88.98 |
| 17 | 4  | 2  | 0.25 | 0.001012 | 89.72 |
| **18** | **4** | **2** | **0.25** | **0.000835** | **89.78 ⭐** |
| 19 | 8  | 2  | 0.25 | 0.000431 | 89.14 |

**Best Config:** rank=4, alpha=2, dropout=0.25, lr=0.000835  
**Optuna Best Full Test Accuracy (10 epochs): 90.01%**

---

## ⚔️ Q2(i) Results — FGSM Attack

### ResNet-18 Training on CIFAR-10

Clean Test Accuracy: **84.36%** ✅ (requirement: ≥ 72%)

### FGSM Comparison Table

| ε | Clean Acc (%) | FGSM Scratch (%) | FGSM ART (%) | Drop Scratch | Drop ART |
|---|--------------|-----------------|-------------|-------------|---------|
| 0.01 | 84.36 | 71.02 | 43.69 | -13.34 pp | -40.67 pp |
| 0.02 | 84.36 | 56.88 | 21.62 | -27.48 pp | -62.74 pp |
| 0.05 | 84.36 | 26.88 | 10.04 | -57.48 pp | -74.32 pp |
| 0.10 | 84.36 | 7.54  | 7.75  | -76.82 pp | -76.61 pp |
| 0.20 | 84.36 | 1.87  | 7.47  | -82.49 pp | -76.89 pp |

---

## 🛡️ Q2(ii) Results — Adversarial Detection

### Detection Summary

| Attack | Requirement | Test Accuracy | Status |
|--------|------------|--------------|--------|
| PGD    | ≥ 70%      | **99.43%**   | ✅ PASS |
| BIM    | ≥ 70%      | **99.64%**   | ✅ PASS |

---

## 🤗 HuggingFace Model Weights

**Repo:** https://huggingface.co/Jaishankar02/assignment5-models

```
assignment5-models/
├── q1/
│   ├── NoLoRA_HeadOnly_best.pt
│   ├── LoRA_r2_a2_do0.1_best.pt
│   ├── LoRA_r2_a4_do0.1_best.pt
│   ├── LoRA_r2_a8_do0.1_best.pt
│   ├── LoRA_r4_a2_do0.1_best.pt
│   ├── LoRA_r4_a4_do0.1_best.pt
│   ├── LoRA_r4_a8_do0.1_best.pt
│   ├── LoRA_r8_a2_do0.1_best.pt
│   ├── LoRA_r8_a4_do0.1_best.pt
│   ├── LoRA_r8_a8_do0.1_best.pt   ← Best manual config
│   └── optuna_best.pt              ← Optuna best (r=4, α=2, drop=0.25)
└── q2/
    ├── resnet18_clean.pt            ← 84.36% on CIFAR-10
    ├── detector_pgd_best.pt         ← 99.43% detection
    ├── detector_pgd_final.pt
    ├── detector_bim_best.pt         ← 99.64% detection
    └── detector_bim_final.pt
```
