"""
Inference script for the PGCL-guided fusion model on circular-polarization (Cir) SAR data.

Usage:
    1. Train Stage-1 (RSD PGCL) and Stage-2 (Cir fusion) with Train_Fusion.py.
    2. Place checkpoints under MODEL_DIR (pgcl_model_best_*.pth, fusion_model_best_*.pth).
    3. Update DATA_ROOT / MODEL_DIR and .mat variable keys below, then run this script.
"""

import os
import copy
import numpy as np
import scipy.io as sio
import h5py
import torch
from scipy import ndimage
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, cohen_kappa_score
from time import time

from VisionTransformer_Cir import VisionTransformer as ViT_Cir
from VisionTransformer_RSD import VisionTransformer as ViT_RSD
from Train_Fusion import GuidedVisionTransformerCir

begin_time = time()

# ---------------------------------------------------------------------------
# User-configurable paths and hyper-parameters (modify before running)
# ---------------------------------------------------------------------------
DATA_ROOT = "./data"
MODEL_DIR = "./checkpoints"

# Cir feature map and ground-truth .mat files (HDF5 v7.3 supported via h5py)
test_data_path = os.path.join(DATA_ROOT, "Cir_Test_Span_Normalization.mat")
test_gt_path = os.path.join(DATA_ROOT, "ground_truth.mat")

# Variable names inside the .mat files
test_data_key = "Cir_features"   # Cir feature tensor in the .mat file
test_gt_key = "cdata"            # ground-truth label map

# Model architecture (Table V & Section IV-E; must match Train_Fusion.py)
windowSize = 7          # 7x7 for Barnaul; use 11 for Oberpfaffenhofen / San Francisco
num_classes = 9           # Barnaul has 9 classes
num_topics = 15           # K
embed_dim = 64            # d
num_heads = 4             # h
in_channels_rsd = 12      # RSD polarimetric feature channels
in_channels_cir = 10      # Cir-domain real feature channels
depth_rsd = 3             # L_RSD
depth_cir = 4             # L_Cir
mlp_feature_depth = 4       # D of MLP_f2t
token_patch_size = 1      # P


def addZeroPadding(X, margin=2):
    """Zero-pad a 2D feature map so that patch extraction stays in bounds."""
    newX = np.zeros((
        X.shape[0] + 2 * margin,
        X.shape[1] + 2 * margin,
        X.shape[2]
    ))
    newX[margin:X.shape[0] + margin, margin:X.shape[1] + margin, :] = X
    return newX


def smooth_segmentation_by_connectivity(label_map, num_classes, min_region_size=90, neighbor_radius=9):
    """
    Post-processing: remove small connected components and relabel them
    using the majority class in a local neighborhood.
    """
    h, w = label_map.shape
    label_map = label_map.astype(np.int32).copy()

    for cls in range(1, num_classes + 1):
        mask = (label_map == cls)
        if not mask.any():
            continue

        labeled, num_comp = ndimage.label(mask)
        if num_comp == 0:
            continue

        sizes = np.bincount(labeled.ravel())
        small_ids = [i for i, s in enumerate(sizes) if i != 0 and s < min_region_size]
        if not small_ids:
            continue

        for comp_id in small_ids:
            ys, xs = np.where(labeled == comp_id)
            for y, x in zip(ys, xs):
                y0 = max(0, y - neighbor_radius)
                y1 = min(h, y + neighbor_radius + 1)
                x0 = max(0, x - neighbor_radius)
                x1 = min(w, x + neighbor_radius + 1)

                window = label_map[y0:y1, x0:x1]
                window_flat = window.reshape(-1)
                window_flat = window_flat[window_flat != 0]
                if window_flat.size == 0:
                    continue

                label_map[y, x] = np.bincount(window_flat).argmax()

    return label_map


def find_latest_model(model_dir, prefix):
    """Return the most recently modified checkpoint with the given prefix."""
    if not os.path.exists(model_dir):
        return None
    model_files = [f for f in os.listdir(model_dir) if f.startswith(prefix) and f.endswith('.pth')]
    if not model_files:
        return None
    model_files.sort(key=lambda x: os.path.getmtime(os.path.join(model_dir, x)), reverse=True)
    return os.path.join(model_dir, model_files[0])


def extract_acc_from_path(path):
    """Parse validation accuracy from checkpoint filename, e.g. fusion_model_best_95.00.pth."""
    base = os.path.basename(path)
    try:
        if 'fusion_model_best_' in base:
            return base.replace('fusion_model_best_', '').replace('.pth', '')
        if 'pgcl_model_best_' in base:
            return base.replace('pgcl_model_best_', '').replace('.pth', '')
        return 'unknown'
    except Exception:
        return 'unknown'


def _state_dict_convert_class_prototypes_to_eta(state_dict):
    """
    Convert legacy class_prototypes (num_classes, num_topics) to sLDA eta
    (num_topics, num_classes) for backward-compatible checkpoint loading.
    """
    new_state = {}
    for k, v in state_dict.items():
        if 'class_prototypes' in k:
            new_key = k.replace('class_prototypes', 'eta')
            new_state[new_key] = v.transpose(0, 1)
        else:
            new_state[k] = v
    return new_state


def _infer_embed_dim(state_dict, default_dim):
    """Infer embed_dim from cls_token in the checkpoint."""
    token = state_dict.get("cls_token")
    if token is None:
        token = state_dict.get("backbone.cls_token")
    if token is None:
        return default_dim
    ckpt_dim = int(token.shape[-1])
    if ckpt_dim != default_dim:
        print(f"Warning: config embed_dim={default_dim}, checkpoint embed_dim={ckpt_dim}; using checkpoint value.")
    return ckpt_dim


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Auto-discover the best checkpoints produced by Train_Fusion.py
pgcl_model_path = find_latest_model(MODEL_DIR, 'pgcl_model_best_')
if pgcl_model_path is None:
    raise FileNotFoundError(
        f"No RSD PGCL checkpoint found in {MODEL_DIR}. "
        "Run Train_Fusion.py Stage-1 first (pgcl_model_best_*.pth)."
    )
print(f"Using RSD PGCL checkpoint: {pgcl_model_path}")

fusion_model_path = find_latest_model(MODEL_DIR, 'fusion_model_best_')
if fusion_model_path is None:
    raise FileNotFoundError(
        f"No fusion checkpoint found in {MODEL_DIR}. "
        "Run Train_Fusion.py Stage-2 first (fusion_model_best_*.pth)."
    )
print(f"Using fusion checkpoint: {fusion_model_path}")

fusion_acc = extract_acc_from_path(fusion_model_path)

print("=" * 60)
print("Loading RSD PGCL model for physics-guided guidance...")
print("=" * 60)
rsd_state = torch.load(pgcl_model_path, map_location=device)
if any(k.startswith("backbone.") for k in rsd_state.keys()):
    raise ValueError(
        "The selected file is a fusion checkpoint (contains backbone.* keys), not an RSD PGCL checkpoint.\n"
        f"Current path: {pgcl_model_path}"
    )
rsd_state = _state_dict_convert_class_prototypes_to_eta(rsd_state)
embed_dim = _infer_embed_dim(rsd_state, embed_dim)
print(f"Building models with embed_dim={embed_dim}, num_heads={num_heads}")

rsd_model = ViT_RSD(
    embed_dim=embed_dim,
    num_heads=num_heads,
    num_classes=num_classes,
    num_topics=num_topics,
    use_pgcl=True,
    use_dual_branch=True,
    in_channels=in_channels_rsd,
    depth=depth_rsd,
    img_size=windowSize,
    patch_size=token_patch_size,
    mlp_feature_depth=mlp_feature_depth,
).to(device)
rsd_model.load_state_dict(rsd_state, strict=True)
rsd_model.eval()

if hasattr(rsd_model, 'use_dual_branch') and rsd_model.use_dual_branch:
    print("Dual-branch RSD architecture detected; using fusion PGCL guidance modules.")
    guidance_topic = copy.deepcopy(rsd_model.fusion_pgcl.branch2_topic_model).to(device).eval()
    guidance_pgcl = copy.deepcopy(rsd_model.fusion_pgcl.branch2_pgcl).to(device).eval()
else:
    print("Single-branch RSD architecture detected.")
    guidance_topic = copy.deepcopy(rsd_model.topic_model).to(device).eval()
    guidance_pgcl = copy.deepcopy(rsd_model.pgcl).to(device).eval()

for p in guidance_topic.parameters():
    p.requires_grad = False
for p in guidance_pgcl.parameters():
    p.requires_grad = False

print("Loading fusion model (ViT_Cir + PGCL guidance)...")
base_model = ViT_Cir(
    embed_dim=embed_dim,
    num_heads=num_heads,
    num_classes=num_classes,
    num_topics=num_topics,
    use_pgcl=False,
    in_channels=in_channels_cir,
    depth=depth_cir,
    img_size=windowSize,
    patch_size=token_patch_size,
    rsd_model_path=pgcl_model_path,
).to(device)

fusion_model = GuidedVisionTransformerCir(
    base_model,
    guidance_topic,
    guidance_pgcl,
    num_classes=num_classes
).to(device)

fusion_state = torch.load(fusion_model_path, map_location=device)
fusion_state = _state_dict_convert_class_prototypes_to_eta(fusion_state)
fusion_model.load_state_dict(fusion_state, strict=False)
fusion_model.eval()
print("Fusion model loaded successfully.")

# ---------------------------------------------------------------------------
# Load Cir test data and run pixel-wise classification
# ---------------------------------------------------------------------------
if not os.path.exists(test_data_path):
    raise FileNotFoundError(f"Test data not found: {test_data_path}\nUpdate test_data_path at the top of this file.")
if not os.path.exists(test_gt_path):
    raise FileNotFoundError(f"Ground truth not found: {test_gt_path}\nUpdate test_gt_path at the top of this file.")

cir_data_file = h5py.File(test_data_path, 'r')
cir_matrix = np.array(cir_data_file[test_data_key][:])
cir_data_file.close()

gt_file = h5py.File(test_gt_path, 'r')
data_gt = np.array(gt_file[test_gt_key][:])
gt_file.close()
data_gt = torch.from_numpy(data_gt.transpose(1, 0))

data_hsi1 = cir_matrix
print("Cir feature shape:", np.shape(data_hsi1))
data_hsi1 = torch.from_numpy(data_hsi1.transpose(2, 1, 0))
height1, width1, _ = data_hsi1.shape

margin = (windowSize - 1) // 2
data_hsi1 = addZeroPadding(data_hsi1, margin=margin)
outputs = np.zeros((height1, width1))

print("=" * 60)
print("Classifying Cir polarimetric SAR data (ViT_Cir + PGCL guidance)...")
print(f"Image size: {height1} x {width1}, patch size: {windowSize} x {windowSize}")
print("=" * 60)

with torch.no_grad():
    for i in range(height1):
        if (i + 1) % 100 == 0:
            print(f"Progress: row {i + 1}/{height1} ({100 * (i + 1) / height1:.1f}%)")

        for j in range(width1):
            if int(data_gt[i, j]) != 0:
                image_patch = data_hsi1[i:i + windowSize, j:j + windowSize, :]
                image_patch = image_patch.reshape(1, image_patch.shape[0], image_patch.shape[1], image_patch.shape[2])
                X_test_image = torch.FloatTensor(image_patch.transpose(0, 3, 1, 2)).to(device)
                fusion_logits, _ = fusion_model(X_test_image)
                prediction = torch.argmax(fusion_logits, dim=1).item()
                outputs[i][j] = prediction + 1

output_dir = MODEL_DIR
os.makedirs(output_dir, exist_ok=True)
raw_output_path = os.path.join(output_dir, f'Cir_result_raw_fusion_{fusion_acc}.mat')
sio.savemat(raw_output_path, {'output': outputs})
print(f"\nRaw classification map saved to: {raw_output_path}")

print("\nApplying connectivity-based post-processing...")
outputs_smoothed = smooth_segmentation_by_connectivity(
    outputs, num_classes=num_classes, min_region_size=90, neighbor_radius=9
)

post_output_path = os.path.join(output_dir, f'Cir_result_post_fusion_{fusion_acc}.mat')
sio.savemat(post_output_path, {'output': outputs_smoothed})
print(f"Post-processed map saved to: {post_output_path}")

total_pixels = np.sum(data_gt.numpy() != 0)
classified_pixels = np.sum(outputs_smoothed != 0)
print("\n" + "=" * 60)
print("Classification summary")
print("=" * 60)
print(f"Labeled pixels: {total_pixels}")
print(f"Classified pixels: {classified_pixels}")
if total_pixels > 0:
    print(f"Coverage: {100 * classified_pixels / total_pixels:.2f}%")
print("Architecture: ViT_Cir + PGCL-guided fusion")
print("=" * 60)

gt_np = data_gt.numpy().astype(np.int32)
mask_labeled = (gt_np > 0) & (gt_np <= num_classes)
if np.sum(mask_labeled) > 0:
    y_true = gt_np[mask_labeled].ravel()
    y_pred_raw = outputs_smoothed[mask_labeled].ravel().astype(np.int32)
    y_pred = np.clip(y_pred_raw, 1, num_classes)
    acc = accuracy_score(y_true, y_pred)
    kappa = cohen_kappa_score(y_true, y_pred)
    print("\nMetrics on labeled pixels:")
    print(f"Accuracy: {acc * 100:.2f}%")
    print(f"Kappa: {kappa:.4f}")
    print("\nConfusion matrix:")
    print(confusion_matrix(y_true, y_pred, labels=np.arange(1, num_classes + 1)))
    print("\nClassification report:")
    print(classification_report(y_true, y_pred, labels=np.arange(1, num_classes + 1), digits=4))

print('\nDone.')
end_time = time()
run_time = end_time - begin_time
print(f'Running time: {run_time / 3600:.2f} hours ({run_time:.2f} seconds)')
