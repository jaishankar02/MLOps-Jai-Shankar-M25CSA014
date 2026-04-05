"""
Q2(ii): Adversarial Detection Model using ResNet-34 on CIFAR-10.
  (a) Detector trained on clean + PGD adversarial images → binary classification
  (b) Detector trained on clean + BIM adversarial images → binary classification
  - WandB logging: 10 samples each of clean, FGSM (scratch), FGSM (ART), PGD, BIM
  - Detection accuracy ≥ 70% required
"""

import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import DataLoader, TensorDataset, random_split
import wandb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm

from art.estimators.classification import PyTorchClassifier
from art.attacks.evasion import (ProjectedGradientDescent,
                                  BasicIterativeMethod,
                                  FastGradientMethod)

DEVICE   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_DIR = "./data"
CKPT_DIR = "./checkpoints/q2"
os.makedirs(CKPT_DIR, exist_ok=True)
os.makedirs("./outputs/q2_detect_visuals", exist_ok=True)

CIFAR10_CLASSES = ["airplane","automobile","bird","cat","deer",
                   "dog","frog","horse","ship","truck"]

# ─────────────────────────── Data helpers ──────────────────────
MEAN = np.array([0.4914, 0.4822, 0.4465], dtype=np.float32).reshape(1, 3, 1, 1)
STD  = np.array([0.2470, 0.2435, 0.2616], dtype=np.float32).reshape(1, 3, 1, 1)


def get_raw_numpy(train=False):
    raw_tf = transforms.ToTensor()
    ds     = torchvision.datasets.CIFAR10(DATA_DIR, train=train, download=True, transform=raw_tf)
    loader = DataLoader(ds, batch_size=512, shuffle=False, num_workers=4)
    xs, ys = [], []
    for x, y in loader:
        xs.append(x.numpy())
        ys.append(y.numpy())
    return np.concatenate(xs), np.concatenate(ys)


def normalize_np(x):
    return (x - MEAN) / STD

# ─────────────────────────── ResNet-18 for CIFAR-10 ────────────
def load_base_classifier():
    """Load the ResNet-18 trained in Q2(i) as the base for generating adversarial examples."""
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(512, 10)
    ckpt = os.path.join(CKPT_DIR, "resnet18_clean.pt")
    if not os.path.exists(ckpt):
        raise FileNotFoundError(
            f"Base classifier not found at {ckpt}. "
            "Run q2_fgsm.py first to train and save it."
        )
    model.load_state_dict(torch.load(ckpt, map_location=DEVICE))
    model = model.to(DEVICE).eval()
    return model


def build_art_clf(model):
    crit = nn.CrossEntropyLoss()
    opt  = torch.optim.SGD(model.parameters(), lr=0.01)
    return PyTorchClassifier(
        model=model, loss=crit, optimizer=opt,
        input_shape=(3, 32, 32), nb_classes=10,
        clip_values=(0.0, 1.0),
        preprocessing=(MEAN, STD),
        device_type="gpu" if DEVICE.type == "cuda" else "cpu",
    )

# ─────────────────────────── Attack generation ─────────────────
def generate_adversarial(art_clf, x_raw, attack_name="pgd", eps=8/255, eps_step=2/255, max_iter=20):
    if attack_name == "pgd":
        attack = ProjectedGradientDescent(
            estimator=art_clf, eps=eps, eps_step=eps_step,
            max_iter=max_iter, targeted=False, batch_size=256
        )
    elif attack_name == "bim":
        attack = BasicIterativeMethod(
            estimator=art_clf, eps=eps, eps_step=eps_step,
            max_iter=max_iter, targeted=False, batch_size=256
        )
    elif attack_name == "fgsm":
        attack = FastGradientMethod(estimator=art_clf, eps=eps, batch_size=256)
    else:
        raise ValueError(f"Unknown attack: {attack_name}")

    print(f"Generating {attack_name.upper()} adversarial examples …")
    x_adv = attack.generate(x=x_raw)
    return x_adv

# ─────────────────────────── Detection model ───────────────────
def build_detector():
    """ResNet-34 binary classifier: 0=clean, 1=adversarial."""
    model = models.resnet34(weights=None)
    model.fc = nn.Linear(512, 2)
    return model


def train_detector(clean_np, adv_np, epochs=20, lr=1e-3, tag="pgd"):
    """Train binary detector on normalized images."""
    clean_t = torch.tensor(normalize_np(clean_np), dtype=torch.float32)
    adv_t   = torch.tensor(normalize_np(adv_np),   dtype=torch.float32)

    x = torch.cat([clean_t, adv_t], dim=0)
    y = torch.cat([torch.zeros(len(clean_t), dtype=torch.long),
                   torch.ones( len(adv_t),   dtype=torch.long)], dim=0)

    # Shuffle
    perm = torch.randperm(len(x))
    x, y = x[perm], y[perm]

    val_size   = int(0.1 * len(x))
    train_size = len(x) - val_size
    full_ds    = TensorDataset(x, y)
    train_ds, val_ds = random_split(full_ds, [train_size, val_size],
                                    generator=torch.Generator().manual_seed(42))

    train_l = DataLoader(train_ds, batch_size=128, shuffle=True,  num_workers=4, pin_memory=True)
    val_l   = DataLoader(val_ds,   batch_size=128, shuffle=False, num_workers=4, pin_memory=True)

    model     = build_detector().to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler    = torch.cuda.amp.GradScaler(enabled=DEVICE.type == "cuda")

    best_va, best_ckpt = 0.0, os.path.join(CKPT_DIR, f"detector_{tag}_best.pt")

    for epoch in range(1, epochs + 1):
        model.train()
        tr_loss, tr_corr, tr_tot = 0.0, 0, 0
        for imgs, labels in train_l:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            with torch.autocast(device_type="cuda", enabled=DEVICE.type == "cuda"):
                out  = model(imgs)
                loss = criterion(out, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            tr_loss += loss.item() * imgs.size(0)
            tr_corr += out.argmax(1).eq(labels).sum().item()
            tr_tot  += imgs.size(0)
        scheduler.step()

        model.eval()
        va_corr, va_tot = 0, 0
        with torch.no_grad():
            for imgs, labels in val_l:
                imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
                va_corr += model(imgs).argmax(1).eq(labels).sum().item()
                va_tot  += imgs.size(0)
        va_acc = va_corr / va_tot
        wandb.log({f"det_{tag}_epoch": epoch,
                   f"det_{tag}_train_loss": tr_loss / tr_tot,
                   f"det_{tag}_train_acc":  tr_corr / tr_tot * 100,
                   f"det_{tag}_val_acc":    va_acc * 100})
        print(f"[{tag.upper()}] Ep {epoch:02d} | Loss {tr_loss/tr_tot:.4f} | "
              f"TrAcc {tr_corr/tr_tot*100:.2f}% | VaAcc {va_acc*100:.2f}%")

        if va_acc > best_va:
            best_va = va_acc
            torch.save(model.state_dict(), best_ckpt)

    # Reload best
    model.load_state_dict(torch.load(best_ckpt, map_location=DEVICE))
    return model, best_va


@torch.no_grad()
def test_detector(model, clean_np, adv_np):
    clean_t = torch.tensor(normalize_np(clean_np), dtype=torch.float32)
    adv_t   = torch.tensor(normalize_np(adv_np),   dtype=torch.float32)
    x = torch.cat([clean_t, adv_t], dim=0).to(DEVICE)
    y = torch.cat([torch.zeros(len(clean_t), dtype=torch.long),
                   torch.ones(len(adv_t),    dtype=torch.long)]).to(DEVICE)

    model.eval()
    all_preds = []
    bs = 256
    for i in range(0, len(x), bs):
        all_preds.append(model(x[i:i+bs]).argmax(1))
    preds = torch.cat(all_preds)
    acc   = preds.eq(y).float().mean().item()
    return acc

# ─────────────────────────── WandB image samples ───────────────
def log_samples_to_wandb(x_clean, x_fgsm_scratch_norm, x_fgsm_art, x_pgd, x_bim, y, n=10):
    """Log 10 samples of each attack type as WandB images."""

    def to_img(arr):
        # arr shape (3, 32, 32) in [0,1] or normalized
        img = np.clip(arr.transpose(1, 2, 0), 0, 1)
        return img

    def norm_to_01(arr_norm):
        # denormalize from CIFAR-10 stats
        return np.clip(arr_norm * STD.reshape(3, 1, 1) + MEAN.reshape(3, 1, 1), 0, 1)

    panels = {
        "clean":        [wandb.Image(to_img(x_clean[i]),
                          caption=f"Clean | {CIFAR10_CLASSES[y[i]]}") for i in range(n)],
        "fgsm_art":     [wandb.Image(to_img(x_fgsm_art[i]),
                          caption=f"FGSM ART | {CIFAR10_CLASSES[y[i]]}") for i in range(n)],
        "pgd":          [wandb.Image(to_img(x_pgd[i]),
                          caption=f"PGD | {CIFAR10_CLASSES[y[i]]}") for i in range(n)],
        "bim":          [wandb.Image(to_img(x_bim[i]),
                          caption=f"BIM | {CIFAR10_CLASSES[y[i]]}") for i in range(n)],
    }

    # FGSM scratch is normalized, so denormalize
    if x_fgsm_scratch_norm is not None:
        panels["fgsm_scratch"] = [
            wandb.Image(to_img(norm_to_01(x_fgsm_scratch_norm[i])),
                        caption=f"FGSM Scratch | {CIFAR10_CLASSES[y[i]]}")
            for i in range(n)
        ]

    for key, imgs in panels.items():
        wandb.log({f"samples_{key}": imgs})


# ─────────────────────────── Main ──────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs",    type=int, default=20)
    parser.add_argument("--wandb_key", type=str, default="")
    parser.add_argument("--eps",       type=float, default=8/255)
    args = parser.parse_args()

    if args.wandb_key:
        wandb.login(key=args.wandb_key)
    wandb.init(project="Assignment5_Q2_Detection", name="ResNet34_Detector", reinit=True)

    # Load base classifier
    base_model = load_base_classifier()
    art_clf    = build_art_clf(base_model)

    # ── Generate adversarial examples ─────────────────────────
    x_test_raw, y_test = get_raw_numpy(train=False)

    x_pgd = generate_adversarial(art_clf, x_test_raw, "pgd",  eps=args.eps)
    x_bim = generate_adversarial(art_clf, x_test_raw, "bim",  eps=args.eps)
    x_fgsm_art = generate_adversarial(art_clf, x_test_raw, "fgsm", eps=args.eps)

    # FGSM scratch (on normalized tensors, just for WandB display)
    norm_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914,0.4822,0.4465),(0.2470,0.2435,0.2616))
    ])
    norm_ds = torchvision.datasets.CIFAR10(DATA_DIR, train=False, download=True, transform=norm_tf)
    norm_loader = DataLoader(norm_ds, batch_size=10, shuffle=False, num_workers=2)
    sample_imgs, sample_labels = next(iter(norm_loader))
    sample_imgs = sample_imgs.to(DEVICE)
    sample_imgs.requires_grad = True
    crit = nn.CrossEntropyLoss()
    loss = crit(base_model(sample_imgs), sample_labels.to(DEVICE))
    loss.backward()
    fgsm_scratch_10 = (sample_imgs + args.eps * sample_imgs.grad.sign()).detach().cpu().numpy()

    # ── Log 10 samples for each attack ────────────────────────
    log_samples_to_wandb(
        x_clean=x_test_raw[:10],
        x_fgsm_scratch_norm=fgsm_scratch_10,
        x_fgsm_art=x_fgsm_art[:10],
        x_pgd=x_pgd[:10],
        x_bim=x_bim[:10],
        y=y_test[:10],
    )

    # ── Train detectors ───────────────────────────────────────
    x_train_raw, _ = get_raw_numpy(train=True)

    # Generate adversarial train examples
    print("\n=== Generating training adversarial examples (this may take a while) ===")
    art_clf_train = build_art_clf(base_model)

    x_pgd_train  = generate_adversarial(art_clf_train, x_train_raw, "pgd", eps=args.eps)
    x_bim_train  = generate_adversarial(art_clf_train, x_train_raw, "bim", eps=args.eps)

    print("\n=== Training PGD Detector ===")
    det_pgd, pgd_va = train_detector(x_train_raw, x_pgd_train, epochs=args.epochs, tag="pgd")

    print("\n=== Training BIM Detector ===")
    det_bim, bim_va = train_detector(x_train_raw, x_bim_train, epochs=args.epochs, tag="bim")

    # ── Test detection accuracy ────────────────────────────────
    pgd_test_acc = test_detector(det_pgd, x_test_raw, x_pgd)
    bim_test_acc = test_detector(det_bim, x_test_raw, x_bim)

    print(f"\n=== Detection Results ===")
    print(f"PGD Detector Test Accuracy: {pgd_test_acc*100:.2f}%")
    print(f"BIM Detector Test Accuracy: {bim_test_acc*100:.2f}%")
    assert pgd_test_acc >= 0.70, f"PGD detection {pgd_test_acc*100:.2f}% < 70%!"
    assert bim_test_acc >= 0.70, f"BIM detection {bim_test_acc*100:.2f}% < 70%!"

    wandb.log({
        "pgd_detection_test_acc": pgd_test_acc * 100,
        "bim_detection_test_acc": bim_test_acc * 100,
    })

    # ── Summary table ─────────────────────────────────────────
    wt = wandb.Table(columns=["Attack", "Detection Accuracy (%)"])
    wt.add_data("PGD", round(pgd_test_acc * 100, 2))
    wt.add_data("BIM", round(bim_test_acc * 100, 2))
    wandb.log({"detection_results_table": wt})

    # ── Save detector weights ─────────────────────────────────
    for tag, m in [("pgd", det_pgd), ("bim", det_bim)]:
        path = os.path.join(CKPT_DIR, f"detector_{tag}_final.pt")
        torch.save(m.state_dict(), path)
        print(f"Saved {tag} detector to {path}")

    wandb.finish()
    print("\nQ2(ii) done.")


if __name__ == "__main__":
    main()
