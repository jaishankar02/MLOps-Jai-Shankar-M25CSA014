"""
Q1 – Optuna hyperparameter search for LoRA on ViT-S / CIFAR-100.
Searches over rank, alpha (and optionally lr).
Best config is then trained fully and pushed to HuggingFace.
"""

import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import torchvision
import torchvision.transforms as transforms
import timm
import wandb
import optuna
from peft import LoraConfig, get_peft_model
from tqdm import tqdm
from huggingface_hub import HfApi

DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CLASSES = 100
SEED        = 42
DATA_DIR    = "./data"
CKPT_DIR    = "./checkpoints/q1_optuna"
os.makedirs(CKPT_DIR, exist_ok=True)

# ── Short run for Optuna (5 epochs) ─────────────────────────────
OPTUNA_EPOCHS = 5
FULL_EPOCHS   = 10
BATCH_SIZE    = 128


def get_dataloaders():
    mean = (0.5071, 0.4867, 0.4408)
    std  = (0.2675, 0.2565, 0.2761)
    train_tf = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    val_tf = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    full_train = torchvision.datasets.CIFAR100(DATA_DIR, train=True,  download=True, transform=train_tf)
    test_ds    = torchvision.datasets.CIFAR100(DATA_DIR, train=False, download=True, transform=val_tf)
    val_size   = int(0.1 * len(full_train))
    train_ds, val_ds = random_split(full_train, [len(full_train) - val_size, val_size],
                                    generator=torch.Generator().manual_seed(SEED))
    train_l = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=4, pin_memory=True)
    val_l   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    test_l  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    return train_l, val_l, test_l


def build_model(rank, alpha, dropout):
    base = timm.create_model("vit_small_patch16_224", pretrained=True, num_classes=NUM_CLASSES)
    for p in base.parameters():
        p.requires_grad = False
    for p in base.head.parameters():
        p.requires_grad = True
    cfg   = LoraConfig(r=rank, lora_alpha=alpha, lora_dropout=dropout,
                       bias="none", target_modules=["qkv"])
    model = get_peft_model(base, cfg)
    for name, p in model.named_parameters():
        if "head" in name:
            p.requires_grad = True
    return model


def train_eval(model, train_loader, val_loader, lr, epochs):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=1e-2
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler    = torch.cuda.amp.GradScaler(enabled=DEVICE.type == "cuda")

    for _ in range(epochs):
        model.train()
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            with torch.autocast(device_type="cuda", enabled=DEVICE.type == "cuda"):
                loss = criterion(model(imgs), labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        scheduler.step()

    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            correct += model(imgs).argmax(1).eq(labels).sum().item()
            total   += imgs.size(0)
    return correct / total


def objective(trial):
    rank    = trial.suggest_categorical("rank",    [2, 4, 8, 16])
    alpha   = trial.suggest_categorical("alpha",   [2, 4, 8, 16])
    dropout = trial.suggest_float("dropout",       0.0, 0.3, step=0.05)
    lr      = trial.suggest_float("lr",            1e-4, 5e-3, log=True)

    train_l, val_l, _ = get_dataloaders()
    model = build_model(rank, alpha, dropout).to(DEVICE)
    val_acc = train_eval(model, train_l, val_l, lr, OPTUNA_EPOCHS)

    wandb.log({"trial_val_acc": val_acc * 100,
               "trial_rank": rank, "trial_alpha": alpha,
               "trial_dropout": dropout, "trial_lr": lr})
    return val_acc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_trials",  type=int, default=20)
    parser.add_argument("--wandb_key", type=str, default="")
    parser.add_argument("--hf_token",  type=str, default="")
    parser.add_argument("--hf_repo",   type=str, default="your-username/vit-cifar100-lora-best")
    args = parser.parse_args()

    if args.wandb_key:
        wandb.login(key=args.wandb_key)

    wandb.init(project="Assignment5_Q1_Optuna", name="optuna_search", reinit=True)

    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(objective, n_trials=args.n_trials, show_progress_bar=True)

    best = study.best_params
    print(f"\nBest params: {best}")
    print(f"Best val acc: {study.best_value*100:.2f}%")
    wandb.log({"best_val_acc": study.best_value * 100, **best})

    # ── Full training with best config ────────────────────────
    print("\nTraining best config for full epochs …")
    train_l, val_l, test_l = get_dataloaders()
    model = build_model(best["rank"], best["alpha"], best["dropout"]).to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=best["lr"], weight_decay=1e-2
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=FULL_EPOCHS)
    scaler    = torch.cuda.amp.GradScaler(enabled=DEVICE.type == "cuda")
    best_va, best_ckpt = 0.0, os.path.join(CKPT_DIR, "optuna_best.pt")

    for epoch in range(1, FULL_EPOCHS + 1):
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
        wandb.log({"epoch": epoch,
                   "best_train_loss": tr_loss / tr_tot,
                   "best_train_acc":  tr_corr / tr_tot * 100,
                   "best_val_acc":    va_acc * 100})
        print(f"Ep {epoch:02d} ValAcc {va_acc*100:.2f}%")

        if va_acc > best_va:
            best_va = va_acc
            torch.save(model.base_model.state_dict(), best_ckpt)

    # Test
    model.eval()
    te_corr, te_tot = 0, 0
    with torch.no_grad():
        for imgs, labels in test_l:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            te_corr += model(imgs).argmax(1).eq(labels).sum().item()
            te_tot  += imgs.size(0)
    print(f"Test Accuracy (Optuna Best): {te_corr/te_tot*100:.2f}%")
    wandb.log({"optuna_best_test_acc": te_corr / te_tot * 100})
    wandb.finish()

    # Push to HuggingFace
    if args.hf_token:
        api = HfApi()
        api.upload_file(path_or_fileobj=best_ckpt,
                        path_in_repo="optuna_best.pt",
                        repo_id=args.hf_repo,
                        token=args.hf_token,
                        repo_type="model")
        print("Uploaded to HuggingFace.")


if __name__ == "__main__":
    main()
