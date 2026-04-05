"""
Q1: ViT-S Fine-tuning on CIFAR-100
  - Without LoRA (classification head only)
  - With LoRA (PEFT) on Q, K, V attention weights for various rank/alpha combos
  -
 WandB logging: loss/accuracy tables, class-wise histogram, gradient update graphs
"""
import os
import argparse
import itertools
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.datasets.cifar import CIFAR100
import timm
import wandb
from peft import LoraConfig, get_peft_model, TaskType
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from huggingface_hub import HfApi

# ─────────────────────────── Config ────────────────────────────
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CLASSES = 100
EPOCHS      = 10
BATCH_SIZE  = 128
LR          = 1e-3
SEED        = 42
DATA_DIR    = "./data"
CKPT_DIR    = "./checkpoints/q1"
os.makedirs(CKPT_DIR, exist_ok=True)
torch.manual_seed(SEED)

CIFAR100_CLASSES = [
    'apple','aquarium_fish','baby','bear','beaver','bed','bee','beetle','bicycle','bottle',
    'bowl','boy','bridge','bus','butterfly','camel','can','castle','caterpillar','cattle',
    'chair','chimpanzee','clock','cloud','cockroach','couch','crab','crocodile','cup',
    'dinosaur','dolphin','elephant','flatfish','forest','fox','girl','hamster','house',
    'kangaroo','keyboard','lamp','lawn_mower','leopard','lion','lizard','lobster','man',
    'maple_tree','motorcycle','mountain','mouse','mushroom','oak_tree','orange','orchid',
    'otter','palm_tree','pear','pickup_truck','pine_tree','plain','plate','poppy',
    'porcupine','possum','rabbit','raccoon','ray','road','rocket','rose','sea','seal',
    'shark','shrew','skunk','skyscraper','snail','snake','spider','squirrel','streetcar',
    'sunflower','sweet_pepper','table','tank','telephone','television','tiger','tractor',
    'train','trout','tulip','turtle','wardrobe','whale','willow_tree','wolf','woman','worm'
]

# ─────────────────────────── Data ──────────────────────────────
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

    full_train = CIFAR100(DATA_DIR, train=True, download=True, transform=train_tf)
    test_ds    = CIFAR100(DATA_DIR, train=False, download=True, transform=val_tf)
    val_size   = int(0.1 * len(full_train))
    train_size = len(full_train) - val_size
    train_ds, val_ds = random_split(full_train, [train_size, val_size],
                                    generator=torch.Generator().manual_seed(SEED))

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    return train_loader, val_loader, test_loader

# ─────────────────────────── Model helpers ─────────────────────
def build_vit_no_lora():
    """ViT-S/16 pretrained on ImageNet, only classification head trainable."""
    model = timm.create_model("vit_small_patch16_224", pretrained=True, num_classes=NUM_CLASSES)
    for name, param in model.named_parameters():
        if "head" not in name:
            param.requires_grad = False
    return model


def build_vit_lora(rank, alpha, dropout):
    """ViT-S/16 with LoRA injected into Q, K, V of all attention blocks + trainable head."""
    base = timm.create_model("vit_small_patch16_224", pretrained=True, num_classes=NUM_CLASSES)

    # Freeze everything first
    for param in base.parameters():
        param.requires_grad = False
    # Unfreeze head
    for param in base.head.parameters():
        param.requires_grad = True

    # LoRA config targeting query, key, value projections in timm ViT-S naming
    lora_cfg = LoraConfig(
        r=rank,
        lora_alpha=alpha,
        lora_dropout=dropout,
        bias="none",
        target_modules=["qkv"],   # timm ViT-S uses fused qkv
    )
    model = get_peft_model(base, lora_cfg)
    # Ensure head stays trainable after PEFT wrapping
    for name, param in model.named_parameters():
        if "head" in name:
            param.requires_grad = True
    return model


def count_trainable(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# ─────────────────────────── Train / Eval ──────────────────────
def train_one_epoch(model, loader, optimizer, criterion, scaler):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        with torch.autocast(device_type="cuda", enabled=DEVICE.type == "cuda"):
            out  = model(imgs)
            loss = criterion(out, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item() * imgs.size(0)
        correct    += out.argmax(1).eq(labels).sum().item()
        total      += imgs.size(0)
    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []
    for imgs, labels in loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        out  = model(imgs)
        loss = criterion(out, labels)
        total_loss += loss.item() * imgs.size(0)
        preds = out.argmax(1)
        correct += preds.eq(labels).sum().item()
        total   += imgs.size(0)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    return total_loss / total, correct / total, np.array(all_preds), np.array(all_labels)

# ─────────────────────────── Gradient logging ──────────────────
def log_lora_gradients(model, epoch, prefix="lora_grad"):
    """Log gradient norms of LoRA weights to WandB."""
    grad_dict = {}
    for name, param in model.named_parameters():
        if param.requires_grad and param.grad is not None and "lora" in name.lower():
            grad_dict[f"{prefix}/{name}_grad_norm"] = param.grad.norm().item()
    if grad_dict:
        wandb.log(grad_dict, commit=False)

# ─────────────────────────── Class-wise histogram ──────────────
def log_classwise_histogram(all_preds, all_labels, run_name):
    per_class_acc = []
    for c in range(NUM_CLASSES):
        mask = all_labels == c
        acc  = (all_preds[mask] == c).mean() if mask.sum() > 0 else 0.0
        per_class_acc.append(acc)

    fig, ax = plt.subplots(figsize=(20, 5))
    ax.bar(range(NUM_CLASSES), per_class_acc)
    ax.set_xticks(range(NUM_CLASSES))
    ax.set_xticklabels(CIFAR100_CLASSES, rotation=90, fontsize=5)
    ax.set_ylabel("Accuracy")
    ax.set_title(f"Class-wise Test Accuracy — {run_name}")
    plt.tight_layout()
    wandb.log({"classwise_accuracy_histogram": wandb.Image(fig)})
    plt.close(fig)

# ─────────────────────────── Single experiment ─────────────────
def run_experiment(use_lora, rank=None, alpha=None, dropout=0.1,
                   exp_no=0, results_table=None):
    if use_lora:
        run_name = f"LoRA_r{rank}_a{alpha}_do{dropout}"
        lora_desc = f"rank={rank} alpha={alpha} dropout={dropout}"
    else:
        run_name  = "NoLoRA_HeadOnly"
        lora_desc = "None"

    print(f"\n{'='*60}\nExperiment: {run_name}\n{'='*60}")

    wandb.init(
        project="Assignment5_Q1_ViT_CIFAR100",
        name=run_name,
        config=dict(use_lora=use_lora, rank=rank, alpha=alpha,
                    dropout=dropout, epochs=EPOCHS, batch_size=BATCH_SIZE, lr=LR),
        reinit=True,
    )

    train_loader, val_loader, test_loader = get_dataloaders()

    if use_lora:
        model = build_vit_lora(rank, alpha, dropout)
    else:
        model = build_vit_no_lora()
    model = model.to(DEVICE)

    trainable = count_trainable(model)
    print(f"Trainable params: {trainable:,}")
    wandb.config.update({"trainable_params": trainable})

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), lr=LR, weight_decay=1e-2
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    scaler    = torch.cuda.amp.GradScaler(enabled=DEVICE.type == "cuda")

    best_val_acc = 0.0
    epoch_rows   = []

    for epoch in range(1, EPOCHS + 1):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer, criterion, scaler)

        # Log LoRA gradients after backward
        if use_lora:
            log_lora_gradients(model, epoch, prefix=f"{run_name}/lora_grad")

        va_loss, va_acc, _, _ = evaluate(model, val_loader, criterion)
        scheduler.step()

        row = dict(epoch=epoch, train_loss=tr_loss, val_loss=va_loss,
                   train_acc=tr_acc * 100, val_acc=va_acc * 100)
        epoch_rows.append(row)
        wandb.log(row)

        print(f"Ep {epoch:02d} | TrLoss {tr_loss:.4f} TrAcc {tr_acc*100:.2f}% "
              f"| VaLoss {va_loss:.4f} VaAcc {va_acc*100:.2f}%")

        if va_acc > best_val_acc:
            best_val_acc = va_acc
            ckpt_path = os.path.join(CKPT_DIR, f"{run_name}_best.pt")
            # Save underlying model weights (unwrap PEFT if needed)
            save_model = model.base_model if use_lora else model
            torch.save(save_model.state_dict(), ckpt_path)

    # Test evaluation
    te_loss, te_acc, preds, labels = evaluate(model, test_loader, criterion)
    print(f"\nTest Accuracy: {te_acc*100:.2f}%")
    wandb.log({"test_accuracy": te_acc * 100})

    # Class-wise histogram
    log_classwise_histogram(preds, labels, run_name)

    # WandB Table for epoch-level metrics
    columns = ["Epoch", "Train Loss", "Val Loss", "Train Acc (%)", "Val Acc (%)"]
    wt = wandb.Table(columns=columns)
    for r in epoch_rows:
        wt.add_data(r["epoch"], round(r["train_loss"], 4), round(r["val_loss"], 4),
                    round(r["train_acc"], 2), round(r["val_acc"], 2))
    wandb.log({"epoch_metrics_table": wt})

    # Append to global results
    if results_table is not None:
        results_table.append({
            "exp_no":        exp_no,
            "lora":          "With LoRA" if use_lora else "Without LoRA",
            "rank":          rank if use_lora else "N/A",
            "alpha":         alpha if use_lora else "N/A",
            "dropout":       dropout if use_lora else "N/A",
            "test_acc":      round(te_acc * 100, 2),
            "trainable_params": trainable,
        })

    wandb.finish()
    return te_acc, os.path.join(CKPT_DIR, f"{run_name}_best.pt")


# ─────────────────────────── Main ──────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--wandb_key",   type=str, default="")
    parser.add_argument("--hf_token",    type=str, default="")
    parser.add_argument("--hf_repo",     type=str, default="your-username/vit-cifar100-lora")
    parser.add_argument("--skip_no_lora", action="store_true")
    args = parser.parse_args()

    if args.wandb_key:
        wandb.login(key=args.wandb_key)

    results = []
    exp_no  = 0

    # ── Experiment 0: No LoRA ──────────────────────────────────
    if not args.skip_no_lora:
        run_experiment(use_lora=False, exp_no=exp_no, results_table=results)
        exp_no += 1

    # ── Experiments with LoRA ─────────────────────────────────
    ranks   = [2, 4, 8]
    alphas  = [2, 4, 8]
    dropout = 0.1
    best_acc, best_ckpt = 0.0, ""

    for r, a in itertools.product(ranks, alphas):
        acc, ckpt = run_experiment(
            use_lora=True, rank=r, alpha=a, dropout=dropout,
            exp_no=exp_no, results_table=results
        )
        exp_no += 1
        if acc > best_acc:
            best_acc, best_ckpt = acc, ckpt

    # ── Print summary table ────────────────────────────────────
    print("\n\n===== SUMMARY TABLE =====")
    header = f"{'Exp':>4} | {'LoRA':>12} | {'Rank':>4} | {'Alpha':>5} | {'Drop':>5} | {'TestAcc%':>8} | {'Params':>10}"
    print(header)
    print("-" * len(header))
    for row in results:
        print(f"{row['exp_no']:>4} | {row['lora']:>12} | {str(row['rank']):>4} | "
              f"{str(row['alpha']):>5} | {str(row['dropout']):>5} | "
              f"{row['test_acc']:>8.2f} | {row['trainable_params']:>10,}")

    # ── Push best model to HuggingFace ────────────────────────
    if args.hf_token and best_ckpt:
        print(f"\nUploading best model ({best_ckpt}) to HuggingFace …")
        api = HfApi()
        api.upload_file(
            path_or_fileobj=best_ckpt,
            path_in_repo="best_model.pt",
            repo_id=args.hf_repo,
            token=args.hf_token,
            repo_type="model",
        )
        print("Upload complete.")


if __name__ == "__main__":
    main()
