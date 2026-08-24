"""
Two-stage training pipeline for PGCL-guided polSAR classification.

Stage 1: Train a dual-branch RSD Vision Transformer with supervised topic modeling (PGCL).
Stage 2: Train a Cir-domain fusion model that reuses frozen PGCL guidance from Stage 1.

Before running, update DATA_ROOT and the Train/Test list paths in the __main__ block.
"""

import os
import sys
import copy
import time

try:
    sys.stdout.reconfigure(line_buffering=True)
except Exception:
    pass

print("Importing PyTorch and model definitions...", flush=True)

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np

from VisionTransformer_RSD import VisionTransformer as ViT_RSD
from VisionTransformer_Cir import VisionTransformer as ViT_Cir

# CircS-ViT hyper-parameters (Table V & Section IV-E of the paper)
PAPER_DEFAULTS = {
    "num_topics": 15,
    "embed_dim": 64,
    "num_heads": 4,
    "rsd_depth": 3,
    "cir_depth": 4,
    "mlp_feature_depth": 4,
    "topic_lambda": 0.1,
    "kd_lambda": 0.2,
    "batch_size": 32,
    "epochs": 120,
    "lr": 4e-4,
    "weight_decay": 0.0005,
    "patch_size": 7,
    "token_patch_size": 1,
    "dropout": 0.1,
    "pgcl_hidden_dim": 128,
    "fusion_dim": 128,
}


class TxtDataset(Dataset):
    """Dataset that reads (sample_path, label) pairs from a text file."""

    def __init__(self, txt_path):
        samples = []
        with open(txt_path, "r") as fh:
            for line in fh:
                items = line.strip().split()
                if len(items) >= 2:
                    samples.append((items[0], int(items[1])))
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def calc_loss(outputs, labels, device):
    outputs = outputs.to(device)
    labels = labels.to(device)
    criterion = nn.CrossEntropyLoss()
    return criterion(outputs, labels).mean()


def build_dataloader(txt_path, batch_size, shuffle, num_workers=0):
    dataset = TxtDataset(txt_path)
    return DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)


def _load_batch_tensor(inputs_list, device):
    """Load a list of per-sample .pth tensors and stack them into one batch."""
    tensors = []
    for sample_path in inputs_list:
        sample_tensor = torch.load(sample_path, map_location=device)
        if sample_tensor.dim() == 3:
            sample_tensor = sample_tensor.unsqueeze(0)
        tensors.append(sample_tensor)
    return torch.cat(tensors, dim=0)


def train_rsd_with_pgcl(config):
    """Stage 1: train RSD PGCL model and return frozen guidance modules."""
    device = config["device"]
    use_pgcl = True

    print(f"[RSD] Loading train list: {config['train_txt']}", flush=True)
    train_loader_temp = build_dataloader(config["train_txt"], batch_size=1, shuffle=False)
    print(f"[RSD] Train samples: {len(train_loader_temp.dataset)}; probing channel count...", flush=True)

    actual_in_channels = None
    for inputs_list, _ in train_loader_temp:
        if len(inputs_list) > 0:
            sample_tensor = torch.load(inputs_list[0], map_location=device)
            if len(sample_tensor.shape) == 4:
                actual_in_channels = sample_tensor.shape[1]
            elif len(sample_tensor.shape) == 3:
                actual_in_channels = sample_tensor.shape[0]
            break

    if actual_in_channels is None:
        raise RuntimeError("Could not infer input channels from training data.")

    print(
        f"Detected in_channels={actual_in_channels} "
        f"(config rsd_in_channels={config.get('rsd_in_channels', 'N/A')})",
        flush=True,
    )

    use_dual_branch = config.get("use_dual_branch", True)

    model = ViT_RSD(
        img_size=config.get("patch_size", PAPER_DEFAULTS["patch_size"]),
        embed_dim=PAPER_DEFAULTS["embed_dim"],
        num_heads=PAPER_DEFAULTS["num_heads"],
        num_classes=config["num_classes"],
        num_topics=config["num_topics"],
        use_pgcl=use_pgcl,
        depth=config.get("rsd_depth", PAPER_DEFAULTS["rsd_depth"]),
        in_channels=actual_in_channels,
        use_dual_branch=use_dual_branch,
        mlp_feature_depth=config.get("mlp_feature_depth", PAPER_DEFAULTS["mlp_feature_depth"]),
    ).to(device)

    if use_dual_branch:
        print("Dual-branch RSD architecture enabled.", flush=True)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["lr"],
        weight_decay=config.get("weight_decay", PAPER_DEFAULTS["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=config["epochs"])

    train_loader = build_dataloader(config["train_txt"], config["batch_size"], shuffle=True)
    test_loader = build_dataloader(config["test_txt"], config["batch_size"], shuffle=False)

    best_acc = 0
    best_path = None
    os.makedirs(config["save_dir"], exist_ok=True)

    num_train = len(train_loader.dataset)
    num_test = len(test_loader.dataset)
    num_batches = len(train_loader)
    print("=" * 60, flush=True)
    print("Stage-1: RSD PGCL training", flush=True)
    print(
        f"  train={num_train}, test={num_test}, batch_size={config['batch_size']}, "
        f"batches/epoch={num_batches}",
        flush=True,
    )
    print("=" * 60, flush=True)

    log_every = max(1, num_batches // 20)

    for epoch in range(config["epochs"]):
        model.train()
        total_loss = 0
        total_cls_loss = 0
        total_topic_loss = 0
        count = 0
        epoch_t0 = time.time()

        for batch_idx, (inputs_list, labels) in enumerate(train_loader, start=1):
            labels = labels.to(device)
            batch_tensor = _load_batch_tensor(inputs_list, device)

            optimizer.zero_grad()
            logits, cls_features, _ = model(batch_tensor)
            cls_loss = calc_loss(logits, labels, device)

            if hasattr(model, 'use_dual_branch') and model.use_dual_branch:
                branch1_features = model.fusion_pgcl.branch1_projection(batch_tensor)
                branch1_topic_loss = model.fusion_pgcl.branch1_topic_model.compute_supervised_loss(
                    branch1_features, labels
                )
                branch2_topic_loss = model.fusion_pgcl.branch2_topic_model.compute_supervised_loss(
                    cls_features, labels
                )
                topic_loss = branch1_topic_loss + branch2_topic_loss
            else:
                topic_loss = model.topic_model.compute_supervised_loss(cls_features, labels)

            total_loss_batch = cls_loss + config["topic_lambda"] * topic_loss
            total_loss_batch.backward()
            optimizer.step()

            total_loss += total_loss_batch.item()
            total_cls_loss += cls_loss.item()
            total_topic_loss += topic_loss.item()
            count += 1

            if batch_idx == 1 or batch_idx % log_every == 0 or batch_idx == num_batches:
                elapsed = time.time() - epoch_t0
                speed = batch_idx / max(elapsed, 1e-6)
                eta = (num_batches - batch_idx) / max(speed, 1e-6)
                print(
                    f"  [RSD][Epoch {epoch+1}/{config['epochs']}] "
                    f"batch {batch_idx}/{num_batches} loss={total_loss_batch.item():.4f} "
                    f"({speed:.2f} batch/s, ETA {eta/60:.1f} min)",
                    flush=True,
                )

        avg_loss = total_loss / count
        print(
            f"[RSD][Epoch {epoch+1}] loss={avg_loss:.4f} "
            f"(cls={total_cls_loss/count:.4f}, topic={total_topic_loss/count:.4f}) "
            f"time={(time.time()-epoch_t0)/60:.1f} min",
            flush=True,
        )
        scheduler.step()

        acc = evaluate(model, test_loader, device, use_pgcl=True)
        if acc >= best_acc:
            best_acc = acc
            best_path = os.path.join(config["save_dir"], f"pgcl_model_best_{best_acc:.2f}.pth")
            torch.save(model.state_dict(), best_path)
            print(f"  Saved best PGCL checkpoint: {best_path} (acc={best_acc:.2f}%)", flush=True)

    if best_path is None:
        raise RuntimeError("Stage-1 finished without saving any checkpoint.")

    model.load_state_dict(torch.load(best_path, map_location=device))
    print(f"Stage-1 done. Best acc={best_acc:.2f}% checkpoint={best_path}", flush=True)

    if hasattr(model, 'use_dual_branch') and model.use_dual_branch:
        guidance_fusion_pgcl = copy.deepcopy(model.fusion_pgcl).eval()
        for p in guidance_fusion_pgcl.parameters():
            p.requires_grad = False
        guidance_topic = copy.deepcopy(model.fusion_pgcl.branch2_topic_model).eval()
        guidance_pgcl = copy.deepcopy(model.fusion_pgcl.branch2_pgcl).eval()
        for p in guidance_topic.parameters():
            p.requires_grad = False
        for p in guidance_pgcl.parameters():
            p.requires_grad = False
        return guidance_topic, guidance_pgcl, best_path, guidance_fusion_pgcl

    guidance_topic = copy.deepcopy(model.topic_model).eval()
    guidance_pgcl = copy.deepcopy(model.pgcl).eval()
    for p in guidance_topic.parameters():
        p.requires_grad = False
    for p in guidance_pgcl.parameters():
        p.requires_grad = False
    return guidance_topic, guidance_pgcl, best_path, None


def evaluate(model, data_loader, device, use_pgcl, in_channels=None):
    """Evaluate RSD model accuracy on a dataloader."""
    model.eval()
    total_preds = []
    total_labels = []

    print(f"  Evaluating {len(data_loader)} batches...", flush=True)
    with torch.no_grad():
        for inputs_list, labels in data_loader:
            labels = labels.to(device)
            batch_tensor = _load_batch_tensor(inputs_list, device)
            outputs = model(batch_tensor)
            logits = outputs[0] if isinstance(outputs, tuple) else outputs
            total_preds.append(torch.argmax(logits, dim=1).cpu().numpy())
            total_labels.append(labels.cpu().numpy())

    preds = np.concatenate(total_preds)
    labels = np.concatenate(total_labels)
    acc = round((preds == labels).sum() / len(labels) * 100, 2)
    print(f"  Eval acc: {acc:.2f}%", flush=True)
    return acc


class GuidedVisionTransformerCir(nn.Module):
    """
    Cir-domain fusion model: ViT_Cir backbone + frozen PGCL guidance + fusion head.
    """

    def __init__(self, base_model, guidance_topic, guidance_pgcl, num_classes,
                 fusion_dim=PAPER_DEFAULTS["fusion_dim"]):
        super().__init__()
        self.backbone = base_model
        self.guidance_topic = guidance_topic
        self.guidance_pgcl = guidance_pgcl
        self.num_classes = num_classes

        hidden_dim = fusion_dim
        self.fusion_head = nn.Sequential(
            nn.Linear(self.backbone.embed_dim + hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x):
        logits_base, cls_features = self.backbone(x)

        with torch.no_grad():
            topic_representation, _ = self.guidance_topic(cls_features)
            guided_features = self.guidance_pgcl(cls_features, topic_representation)

        fused_feature = torch.cat([cls_features, guided_features], dim=1)
        fusion_logits = self.fusion_head(fused_feature)
        return fusion_logits, logits_base


def train_cir_with_guidance(config, guidance_topic, guidance_pgcl, rsd_model_path):
    """Stage 2: train Cir fusion model with PGCL guidance from Stage 1."""
    device = config["device"]

    base_model = ViT_Cir(
        img_size=config.get("patch_size", PAPER_DEFAULTS["patch_size"]),
        embed_dim=PAPER_DEFAULTS["embed_dim"],
        num_heads=PAPER_DEFAULTS["num_heads"],
        num_classes=config["num_classes"],
        num_topics=config["num_topics"],
        use_pgcl=False,
        depth=config.get("cir_depth", PAPER_DEFAULTS["cir_depth"]),
        rsd_model_path=rsd_model_path,
    ).to(device)

    guided_model = GuidedVisionTransformerCir(
        base_model,
        guidance_topic.to(device),
        guidance_pgcl.to(device),
        num_classes=config["num_classes"]
    ).to(device)

    optimizer = torch.optim.AdamW(
        guided_model.parameters(),
        lr=config["lr"],
        weight_decay=config.get("weight_decay", PAPER_DEFAULTS["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=config["epochs"])

    train_loader = build_dataloader(config["train_txt"], config["batch_size"], shuffle=True)
    test_loader = build_dataloader(config["test_txt"], config["batch_size"], shuffle=False)

    best_acc = 0
    best_path = None
    os.makedirs(config["save_dir"], exist_ok=True)

    num_train = len(train_loader.dataset)
    num_test = len(test_loader.dataset)
    num_batches = len(train_loader)
    print("=" * 60, flush=True)
    print("Stage-2: Cir fusion training with PGCL guidance", flush=True)
    print(f"  train={num_train}, test={num_test}, batches/epoch={num_batches}", flush=True)
    print("=" * 60, flush=True)

    log_every = max(1, num_batches // 20)

    for epoch in range(config["epochs"]):
        guided_model.train()
        total_loss = 0
        count = 0
        epoch_t0 = time.time()

        for batch_idx, (inputs_list, labels) in enumerate(train_loader, start=1):
            labels = labels.to(device)
            batch_tensor = _load_batch_tensor(inputs_list, device)

            optimizer.zero_grad()
            fusion_logits, logits_base = guided_model(batch_tensor)

            ce_loss = calc_loss(fusion_logits, labels, device)
            kd_loss = torch.nn.functional.kl_div(
                torch.log_softmax(fusion_logits, dim=1),
                torch.softmax(logits_base.detach(), dim=1),
                reduction="batchmean"
            )
            total_batch_loss = ce_loss + config["kd_lambda"] * kd_loss
            total_batch_loss.backward()
            optimizer.step()

            total_loss += total_batch_loss.item()
            count += 1

            if batch_idx == 1 or batch_idx % log_every == 0 or batch_idx == num_batches:
                elapsed = time.time() - epoch_t0
                speed = batch_idx / max(elapsed, 1e-6)
                eta = (num_batches - batch_idx) / max(speed, 1e-6)
                print(
                    f"  [Cir][Epoch {epoch+1}/{config['epochs']}] "
                    f"batch {batch_idx}/{num_batches} loss={total_batch_loss.item():.4f} "
                    f"({speed:.2f} batch/s, ETA {eta/60:.1f} min)",
                    flush=True,
                )

        avg_loss = total_loss / count
        print(f"[Cir][Epoch {epoch+1}] loss={avg_loss:.4f} time={(time.time()-epoch_t0)/60:.1f} min", flush=True)
        scheduler.step()

        acc = evaluate_fusion(guided_model, test_loader, device, config["cir_in_channels"])
        if acc >= best_acc:
            best_acc = acc
            best_path = os.path.join(config["save_dir"], f"fusion_model_best_{best_acc:.2f}.pth")
            torch.save(guided_model.state_dict(), best_path)
            print(f"  Saved best fusion checkpoint: {best_path} (acc={best_acc:.2f}%)", flush=True)

    if best_path is None:
        raise RuntimeError("Stage-2 finished without saving any fusion checkpoint.")

    print(f"Stage-2 done. Best acc={best_acc:.2f}% checkpoint={best_path}", flush=True)
    return best_path


def evaluate_fusion(model, data_loader, device, in_channels):
    """Evaluate fusion model accuracy."""
    model.eval()
    total_preds = []
    total_labels = []

    print(f"  Fusion eval: {len(data_loader)} batches...", flush=True)
    with torch.no_grad():
        for inputs_list, labels in data_loader:
            labels = labels.to(device)
            batch_tensor = _load_batch_tensor(inputs_list, device)
            fusion_logits, _ = model(batch_tensor)
            total_preds.append(torch.argmax(fusion_logits, dim=1).cpu().numpy())
            total_labels.append(labels.cpu().numpy())

    preds = np.concatenate(total_preds)
    labels = np.concatenate(total_labels)
    acc = round((preds == labels).sum() / len(labels) * 100, 2)
    print(f"  Fusion eval acc: {acc:.2f}%", flush=True)
    return acc


if __name__ == "__main__":
    # -----------------------------------------------------------------------
    # User-configurable paths (modify before running)
    # -----------------------------------------------------------------------
    DATA_ROOT = "./data"
    CHECKPOINT_DIR = "./checkpoints"

    print("=" * 60, flush=True)
    print("Train_Fusion starting...", flush=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)

    rsd_config = {
        "device": device,
        "train_txt": os.path.join(DATA_ROOT, "RSD", "Train.txt"),
        "test_txt": os.path.join(DATA_ROOT, "RSD", "Test.txt"),
        "batch_size": PAPER_DEFAULTS["batch_size"],
        "epochs": PAPER_DEFAULTS["epochs"],
        "lr": PAPER_DEFAULTS["lr"],
        "weight_decay": PAPER_DEFAULTS["weight_decay"],
        "num_classes": 9,
        "num_topics": PAPER_DEFAULTS["num_topics"],
        "rsd_depth": PAPER_DEFAULTS["rsd_depth"],
        "mlp_feature_depth": PAPER_DEFAULTS["mlp_feature_depth"],
        "topic_lambda": PAPER_DEFAULTS["topic_lambda"],
        "rsd_in_channels": 12,
        "patch_size": PAPER_DEFAULTS["patch_size"],
        "save_dir": CHECKPOINT_DIR,
    }

    cir_config = {
        "device": device,
        "train_txt": os.path.join(DATA_ROOT, "Cir", "Train.txt"),
        "test_txt": os.path.join(DATA_ROOT, "Cir", "Test.txt"),
        "batch_size": PAPER_DEFAULTS["batch_size"],
        "epochs": PAPER_DEFAULTS["epochs"],
        "lr": PAPER_DEFAULTS["lr"],
        "weight_decay": PAPER_DEFAULTS["weight_decay"],
        "num_classes": 9,
        "num_topics": PAPER_DEFAULTS["num_topics"],
        "kd_lambda": PAPER_DEFAULTS["kd_lambda"],
        "cir_depth": PAPER_DEFAULTS["cir_depth"],
        "cir_in_channels": 10,
        "patch_size": PAPER_DEFAULTS["patch_size"],
        "save_dir": CHECKPOINT_DIR,
    }

    result = train_rsd_with_pgcl(rsd_config)
    if len(result) == 4:
        guidance_topic, guidance_pgcl, rsd_best_path, _ = result
    else:
        guidance_topic, guidance_pgcl, rsd_best_path = result

    train_cir_with_guidance(cir_config, guidance_topic, guidance_pgcl, rsd_best_path)
