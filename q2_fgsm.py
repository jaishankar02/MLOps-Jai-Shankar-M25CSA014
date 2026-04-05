"""
Q2(i): FGSM Attack — From Scratch vs IBM ART
  - Train ResNet18 from scratch on CIFAR-10 (≥72% test accuracy)
  - FGSM from scratch
  - FGSM via IBM ART
  - Visual comparison + WandB logging of samples and metrics
"""

import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torchvision.models as models
import wandb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm

# IBM ART imports
from art.estimators.classification import PyTorchClassifier
from art.attacks.evasion import FastGradientMethod

DEVICE   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_DIR = "./data"
CKPT_DIR = "./checkpoints/q2"
os.makedirs(CKPT_DIR, exist_ok=True)
os.makedirs("./outputs/q2_fgsm_visuals", exist_ok=True)

CIFAR10_CLASSES = ["airplane","automobile","bird","cat","deer",
                   "dog","frog","horse","ship","truck"]

# ─────────────────────────── Data ──────────────────────────────
def get_dataloaders(batch_size=128):
    mean = (0.4914, 0.4822, 0.4465)
    std  = (0.2470, 0.2435, 0.2616)
    train_tf = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    test_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    train_ds = torchvision.datasets.CIFAR10(DATA_DIR, train=True,  download=True, transform=train_tf)
    test_ds  = torchvision.datasets.CIFAR10(DATA_DIR, train=False, download=True, transform=test_tf)
    train_l  = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=4, pin_memory=True)
    test_l   = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    return train_l, test_l, test_ds


def get_raw_test_arrays(test_ds):
    """Return raw (unnormalized 0-1) images and labels as numpy arrays for ART."""
    imgs, labels = [], []
    raw_tf = transforms.Compose([transforms.ToTensor()])
    raw_ds = torchvision.datasets.CIFAR10(DATA_DIR, train=False, download=True, transform=raw_tf)
    loader = DataLoader(raw_ds, batch_size=256, shuffle=False, num_workers=4)
    for x, y in loader:
        imgs.append(x.numpy())
        labels.append(y.numpy())
    return np.concatenate(imgs), np.concatenate(labels)

# ─────────────────────────── Model ─────────────────────────────
def build_resnet18():
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(512, 10)
    return model

# ─────────────────────────── Training ──────────────────────────
def train(model, train_loader, epochs=30, lr=1e-1):
    model = model.to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[15, 22], gamma=0.1)
    scaler    = torch.cuda.amp.GradScaler(enabled=DEVICE.type == "cuda")

    for epoch in range(1, epochs + 1):
        model.train()
        tr_loss, correct, total = 0.0, 0, 0
        for imgs, labels in tqdm(train_loader, desc=f"Ep {epoch}/{epochs}", leave=False):
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            with torch.autocast(device_type="cuda", enabled=DEVICE.type == "cuda"):
                out  = model(imgs)
                loss = criterion(out, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            tr_loss += loss.item() * imgs.size(0)
            correct += out.argmax(1).eq(labels).sum().item()
            total   += imgs.size(0)
        scheduler.step()
        tr_acc = correct / total
        wandb.log({"epoch": epoch, "train_loss": tr_loss / total, "train_acc": tr_acc * 100})
        print(f"Ep {epoch:02d} | Loss {tr_loss/total:.4f} | Acc {tr_acc*100:.2f}%")

    return model


@torch.no_grad()
def test_clean(model, loader):
    model.eval()
    correct, total = 0, 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        correct += model(imgs).argmax(1).eq(labels).sum().item()
        total   += imgs.size(0)
    return correct / total

# ─────────────────────────── FGSM from scratch ─────────────────
def fgsm_scratch(model, imgs, labels, epsilon, criterion):
    """FGSM without ART."""
    model.eval()
    imgs_adv = imgs.clone().detach().requires_grad_(True)
    loss = criterion(model(imgs_adv), labels)
    loss.backward()
    perturbation = epsilon * imgs_adv.grad.sign()
    adv = torch.clamp(imgs_adv + perturbation, -3.0, 3.0).detach()
    return adv


def eval_fgsm_scratch(model, loader, epsilon, criterion):
    model.eval()
    correct, total = 0, 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        adv = fgsm_scratch(model, imgs, labels, epsilon, criterion)
        correct += model(adv).argmax(1).eq(labels).sum().item()
        total   += imgs.size(0)
    return correct / total

# ─────────────────────────── FGSM via ART ──────────────────────
def build_art_classifier(model):
    mean = np.array([0.4914, 0.4822, 0.4465]).reshape(1, 3, 1, 1).astype(np.float32)
    std  = np.array([0.2470, 0.2435, 0.2616]).reshape(1, 3, 1, 1).astype(np.float32)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    art_clf = PyTorchClassifier(
        model=model,
        loss=criterion,
        optimizer=optimizer,
        input_shape=(3, 32, 32),
        nb_classes=10,
        clip_values=(0.0, 1.0),
        preprocessing=(mean, std),
        device_type="gpu" if DEVICE.type == "cuda" else "cpu",
    )
    return art_clf


def eval_fgsm_art(art_clf, x_test, y_test, epsilon):
    attack = FastGradientMethod(estimator=art_clf, eps=epsilon, batch_size=256)
    x_adv  = attack.generate(x=x_test)
    preds  = np.argmax(art_clf.predict(x_adv), axis=1)
    acc    = (preds == y_test).mean()
    return acc, x_adv

# ─────────────────────────── Visualisation ─────────────────────
def denormalize(tensor):
    mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(3, 1, 1)
    std  = torch.tensor([0.2470, 0.2435, 0.2616]).view(3, 1, 1)
    return torch.clamp(tensor.cpu() * std + mean, 0, 1)


def save_comparison_grid(clean_imgs, adv_scratch, adv_art, labels, preds_clean,
                          preds_scratch, preds_art, epsilon, n=10):
    fig, axes = plt.subplots(3, n, figsize=(2 * n, 7))
    for i in range(n):
        for row, (imgs_row, title_pfx) in enumerate([
            (clean_imgs,  "Clean"),
            (adv_scratch, "Scratch"),
            (adv_art,     "ART"),
        ]):
            ax = axes[row, i]
            if isinstance(imgs_row, np.ndarray):
                img = imgs_row[i].transpose(1, 2, 0)
                img = np.clip(img, 0, 1)
            else:
                img = denormalize(imgs_row[i]).permute(1, 2, 0).numpy()
            ax.imshow(img)
            ax.axis("off")
            if i == 0:
                ax.set_ylabel(title_pfx, fontsize=10, rotation=0, labelpad=50, va="center")

    plt.suptitle(f"FGSM Comparison (ε={epsilon})", fontsize=13)
    plt.tight_layout()
    path = f"./outputs/q2_fgsm_visuals/fgsm_comparison_eps{epsilon:.2f}.png"
    plt.savefig(path, dpi=100, bbox_inches="tight")
    plt.close()
    return path


# ─────────────────────────── Main ──────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs",    type=int,   default=30)
    parser.add_argument("--wandb_key", type=str,   default="")
    parser.add_argument("--skip_train", action="store_true",
                        help="Load pre-trained weights instead of training")
    args = parser.parse_args()

    if args.wandb_key:
        wandb.login(key=args.wandb_key)

    wandb.init(project="Assignment5_Q2_FGSM", name="ResNet18_FGSM", reinit=True)

    train_loader, test_loader, test_ds = get_dataloaders()
    model = build_resnet18()
    ckpt  = os.path.join(CKPT_DIR, "resnet18_clean.pt")

    if args.skip_train and os.path.exists(ckpt):
        model.load_state_dict(torch.load(ckpt, map_location=DEVICE))
        model = model.to(DEVICE)
        print("Loaded pre-trained weights.")
    else:
        model = train(model, train_loader, epochs=args.epochs)
        torch.save(model.state_dict(), ckpt)

    clean_acc = test_clean(model, test_loader)
    print(f"\nClean Test Accuracy: {clean_acc*100:.2f}%")
    assert clean_acc >= 0.72, f"Clean accuracy {clean_acc*100:.2f}% is below 72% threshold!"
    wandb.log({"clean_test_acc": clean_acc * 100})

    criterion = nn.CrossEntropyLoss()

    # ── FGSM Scratch ──────────────────────────────────────────
    epsilons     = [0.01, 0.02, 0.05, 0.1, 0.2]
    scratch_accs = []
    for eps in epsilons:
        acc = eval_fgsm_scratch(model, test_loader, eps, criterion)
        scratch_accs.append(acc)
        wandb.log({f"fgsm_scratch_acc_eps{eps}": acc * 100})
        print(f"FGSM Scratch ε={eps:.2f}: {acc*100:.2f}%")

    # ── FGSM ART ──────────────────────────────────────────────
    x_test, y_test = get_raw_test_arrays(test_ds)
    art_clf        = build_art_classifier(model)
    art_accs       = []

    for eps in epsilons:
        acc, x_adv = eval_fgsm_art(art_clf, x_test, y_test, eps)
        art_accs.append(acc)
        wandb.log({f"fgsm_art_acc_eps{eps}": acc * 100})
        print(f"FGSM ART   ε={eps:.2f}: {acc*100:.2f}%")

    # ── WandB comparison table ─────────────────────────────────
    wt = wandb.Table(columns=["Epsilon", "Clean Acc (%)", "FGSM Scratch Acc (%)", "FGSM ART Acc (%)"])
    for i, eps in enumerate(epsilons):
        wt.add_data(eps, round(clean_acc * 100, 2),
                    round(scratch_accs[i] * 100, 2),
                    round(art_accs[i] * 100, 2))
    wandb.log({"fgsm_comparison_table": wt})

    # ── Visual comparison ─────────────────────────────────────
    # Get 10 sample clean images
    sample_imgs, sample_labels = next(iter(DataLoader(
        torchvision.datasets.CIFAR10(DATA_DIR, train=False,
            transform=transforms.Compose([transforms.ToTensor(),
                transforms.Normalize((0.4914,0.4822,0.4465),(0.2470,0.2435,0.2616))])),
        batch_size=10, shuffle=False)))
    sample_imgs   = sample_imgs.to(DEVICE)
    sample_labels = sample_labels.to(DEVICE)

    for eps in [0.05, 0.1]:
        # scratch adversarial
        adv_s = fgsm_scratch(model, sample_imgs, sample_labels, eps, criterion)

        # art adversarial (on raw 0-1 images)
        raw_tf  = transforms.ToTensor()
        raw_ds  = torchvision.datasets.CIFAR10(DATA_DIR, train=False, download=True, transform=raw_tf)
        raw_10  = torch.stack([raw_ds[i][0] for i in range(10)]).numpy()
        atk     = FastGradientMethod(estimator=art_clf, eps=eps, batch_size=10)
        adv_art = atk.generate(x=raw_10)  # shape (10,3,32,32) in [0,1]

        path = save_comparison_grid(sample_imgs, adv_s, adv_art,
                                    sample_labels.cpu(), None, None, None, eps)

        wandb.log({f"fgsm_visual_eps{eps}": wandb.Image(path,
                    caption=f"Clean | Scratch FGSM | ART FGSM (ε={eps})")})

    # ── Log 10 sample images as WandB media ───────────────────
    panel = []
    for i in range(10):
        raw_img = np.clip(raw_10[i].transpose(1, 2, 0), 0, 1)
        panel.append(wandb.Image(raw_img,
                     caption=f"Clean | {CIFAR10_CLASSES[y_test[i]]}"))
    wandb.log({"clean_samples": panel})

    wandb.finish()
    print("\nQ2(i) done.")


if __name__ == "__main__":
    main()
