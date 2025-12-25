#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DINOv2 (ViT-B/14) 追加事前学習 (自己教師あり) + 年齢推定 linear probe 評価
- データ: /home/user/Desktop/yyama/ES_PNG_ROI_processed/<ResearchID>/<ResearchID>_<ExamNumber>/*.png
        かつ ファイル名: <ResearchID>_<ExamNumber>_<idx>.png (例: aabbccdd_1_1.png)
- ラベル: /home/user/Desktop/yyama/label.csv (ResearchID, ExamNumber, PatientAge)
- 分割: ResearchID単位 (train/val/test)
- サンプリング: 1 step で患者を選ぶ -> その患者から 8枚 (不足は復元抽出)
- マルチクロップ: global2(224) + local4(96)
- 保存先: /home/user/Desktop/yyama/ssl_dinov2/Runs/<run_name>/
- 目安 epoch: 20

使い方例:
python train_ssl_dinov2.py \
  --data_root /home/user/Desktop/yyama/ES_PNG_ROI_processed \
  --label_csv /home/user/Desktop/yyama/label.csv \
  --runs_root /home/user/Desktop/yyama/ssl_dinov2/Runs \
  --epochs 20 --batch_size 256 --num_workers 12

注意:
- torch.hub で DINOv2 事前学習重みを取得できる環境を想定（初回のみDL）。
  オフラインなら --pretrained_ckpt に .pth を指定してください。
"""

import argparse
import csv
import json
import math
import os
import random
import re
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Sampler
from torchvision import transforms
from PIL import Image

from tqdm import tqdm

# -----------------------------
# Utilities
# -----------------------------
def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def now_run_name(prefix: str = "dinov2_ssl"):
    return f"{prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"


def save_json(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def is_bf16_supported() -> bool:
    if not torch.cuda.is_available():
        return False
    # bf16 is generally supported on A100/H100 etc. Torch provides a flag:
    try:
        return torch.cuda.is_bf16_supported()
    except Exception:
        return False


# -----------------------------
# Label loading & parsing
# -----------------------------
def load_label_csv(label_csv: Path) -> Dict[Tuple[str, str], float]:
    """
    Returns: {(ResearchID, ExamNumber): PatientAge}
    """
    mapping: Dict[Tuple[str, str], float] = {}
    with open(label_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required = {"ResearchID", "ExamNumber", "PatientAge"}
        if not required.issubset(set(reader.fieldnames or [])):
            raise ValueError(f"label.csv must contain columns {required}, got {reader.fieldnames}")
        for row in reader:
            rid = str(row["ResearchID"]).strip()
            ex = str(row["ExamNumber"]).strip()
            age = float(row["PatientAge"])
            mapping[(rid, ex)] = age
    return mapping


_FILENAME_RE = re.compile(r"^(?P<rid>[^_]+)_(?P<exam>[^_]+)_(?P<idx>\d+)\.png$", re.IGNORECASE)


def parse_rid_exam_from_path(p: Path) -> Tuple[Optional[str], Optional[str]]:
    """
    Try extracting ResearchID and ExamNumber from file name first.
    Fallback: parent dir name like <ResearchID>_<ExamNumber>
    """
    m = _FILENAME_RE.match(p.name)
    if m:
        return m.group("rid"), m.group("exam")

    # fallback: parent folder like aabbccdd_1
    parent = p.parent.name
    if "_" in parent:
        rid, exam = parent.split("_", 1)
        return rid, exam

    return None, None


# -----------------------------
# Dataset indexing
# -----------------------------
@dataclass
class Sample:
    path: str
    rid: str
    exam: str
    age: float


def build_index(
    data_root: Path,
    label_map: Dict[Tuple[str, str], float],
    exts=(".png", ".PNG"),
) -> Tuple[List[Sample], Dict[str, List[int]]]:
    """
    Returns:
      - samples: list of Sample
      - patient_to_indices: {ResearchID: [sample_idx, ...]}
    Only keeps images whose (rid, exam) exist in label_map.
    """
    samples: List[Sample] = []
    patient_to_indices: Dict[str, List[int]] = {}

    for p in data_root.rglob("*"):
        if not p.is_file():
            continue
        if p.suffix not in exts:
            continue

        rid, exam = parse_rid_exam_from_path(p)
        if rid is None or exam is None:
            continue

        key = (rid, str(exam))
        if key not in label_map:
            # label missing => skip
            continue

        age = label_map[key]
        idx = len(samples)
        samples.append(Sample(path=str(p), rid=rid, exam=str(exam), age=age))
        patient_to_indices.setdefault(rid, []).append(idx)

    if len(samples) == 0:
        raise RuntimeError("No samples found. Check data_root, filename pattern, and label.csv keys.")
    return samples, patient_to_indices


def split_patients(
    patient_ids: List[str],
    seed: int,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
) -> Tuple[List[str], List[str], List[str]]:
    """
    Split by ResearchID into train/val/test
    """
    rng = random.Random(seed)
    ids = patient_ids[:]
    rng.shuffle(ids)
    n = len(ids)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    train = ids[:n_train]
    val = ids[n_train:n_train + n_val]
    test = ids[n_train + n_val:]
    return train, val, test


# -----------------------------
# Patient-balanced sampler
# -----------------------------
class PatientBatchSampler(Sampler[List[int]]):
    """
    Yields batches of indices where:
      - pick patients uniformly
      - for each patient, draw k images (with replacement if needed)
    Batch size must be multiple of k.

    Example: batch_size=256, k=8 -> 32 patients per batch.
    """
    def __init__(
        self,
        patient_to_indices: Dict[str, List[int]],
        patient_ids: List[str],
        batch_size: int,
        k_per_patient: int = 8,
        steps_per_epoch: int = 1000,
        seed: int = 0,
    ):
        assert batch_size % k_per_patient == 0, "batch_size must be multiple of k_per_patient"
        self.patient_to_indices = patient_to_indices
        self.patient_ids = patient_ids
        self.batch_size = batch_size
        self.k = k_per_patient
        self.patients_per_batch = batch_size // k_per_patient
        self.steps_per_epoch = steps_per_epoch
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch: int):
        self.epoch = epoch

    def __len__(self):
        return self.steps_per_epoch

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch * 10007)
        for _ in range(self.steps_per_epoch):
            batch: List[int] = []
            # sample patients uniformly (with replacement)
            for _p in range(self.patients_per_batch):
                rid = rng.choice(self.patient_ids)
                idxs = self.patient_to_indices[rid]
                # draw k images, with replacement if needed
                if len(idxs) >= self.k:
                    chosen = rng.sample(idxs, self.k)
                else:
                    chosen = [rng.choice(idxs) for _ in range(self.k)]
                batch.extend(chosen)
            rng.shuffle(batch)
            yield batch


# -----------------------------
# Image dataset
# -----------------------------
class EndoscopyDataset(Dataset):
    def __init__(self, samples: List[Sample], transform=None):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i: int):
        s = self.samples[i]
        img = Image.open(s.path).convert("RGB")
        if self.transform is not None:
            out = self.transform(img)
        else:
            out = img
        return out, float(s.age), s.rid


# -----------------------------
# Multi-crop augmentation (DINO-style)
# -----------------------------
class MultiCropTransform:
    def __init__(
        self,
        global_size=224,
        local_size=96,
        n_global=2,
        n_local=4,
    ):
        # For endoscopy, keep augmentation moderate (not too aggressive)
        flip_and_color = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply(
                [transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.02)],
                p=0.8,
            ),
            transforms.RandomGrayscale(p=0.05),
        ])

        normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                 std=(0.229, 0.224, 0.225)),
        ])

        # Gaussian blur & solarization (DINO-like, but slightly weaker)
        def gaussian_blur(p=1.0):
            return transforms.RandomApply([transforms.GaussianBlur(kernel_size=23, sigma=(0.1, 2.0))], p=p)

        def solarize(p=0.0):
            return transforms.RandomApply([transforms.RandomSolarize(threshold=128, p=1.0)], p=p)

        self.global_1 = transforms.Compose([
            transforms.RandomResizedCrop(global_size, scale=(0.4, 1.0), interpolation=transforms.InterpolationMode.BICUBIC),
            flip_and_color,
            gaussian_blur(p=1.0),
            normalize,
        ])
        self.global_2 = transforms.Compose([
            transforms.RandomResizedCrop(global_size, scale=(0.4, 1.0), interpolation=transforms.InterpolationMode.BICUBIC),
            flip_and_color,
            gaussian_blur(p=0.1),
            solarize(p=0.2),
            normalize,
        ])
        self.local = transforms.Compose([
            transforms.RandomResizedCrop(local_size, scale=(0.08, 0.4), interpolation=transforms.InterpolationMode.BICUBIC),
            flip_and_color,
            gaussian_blur(p=0.3),
            normalize,
        ])

        self.n_global = n_global
        self.n_local = n_local

    def __call__(self, img):
        crops = []
        # exactly 2 global (DINO standard)
        crops.append(self.global_1(img))
        crops.append(self.global_2(img))
        for _ in range(self.n_local):
            crops.append(self.local(img))
        return crops


# -----------------------------
# DINO head & wrapper
# -----------------------------
class DINOHead(nn.Module):
    def __init__(self, in_dim: int, out_dim: int = 65536, hidden_dim: int = 2048, bottleneck_dim: int = 256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, bottleneck_dim),
            nn.GELU(),
        )
        self.last = nn.utils.weight_norm(nn.Linear(bottleneck_dim, out_dim, bias=False))
        # Initialize weight_g to 1 (as common)
        self.last.weight_g.data.fill_(1.0)

    def forward(self, x):
        x = self.mlp(x)
        x = F.normalize(x, dim=-1)
        x = self.last(x)
        return x


class MultiCropWrapper(nn.Module):
    """
    Runs backbone on list of crops and returns list of logits.
    backbone must return a single embedding per image.
    """
    def __init__(self, backbone: nn.Module, head: nn.Module):
        super().__init__()
        self.backbone = backbone
        self.head = head

    def forward(self, x_list: List[torch.Tensor]) -> List[torch.Tensor]:
        outs: List[torch.Tensor] = []
        for x in x_list:
            feat = self.backbone(x)
            if isinstance(feat, (tuple, list)):
                feat = feat[0]
            out = self.head(feat)
            outs.append(out)
        return outs


# -----------------------------
# DINO loss (teacher-student with centering)
# -----------------------------
class DINOLoss(nn.Module):
    def __init__(
        self,
        out_dim: int,
        teacher_temp: float = 0.04,
        student_temp: float = 0.1,
        center_momentum: float = 0.9,
    ):
        super().__init__()
        self.teacher_temp = teacher_temp
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.register_buffer("center", torch.zeros(1, out_dim))

    @torch.no_grad()
    def update_center(self, teacher_outputs: torch.Tensor):
        # teacher_outputs: [B, D]
        batch_center = torch.mean(teacher_outputs, dim=0, keepdim=True)
        self.center = self.center * self.center_momentum + batch_center * (1 - self.center_momentum)

    def forward(self, student_logits: List[torch.Tensor], teacher_logits: List[torch.Tensor]) -> torch.Tensor:
        """
        DINO cross-entropy between teacher (only global views) and student (all views)
        Typical: teacher uses 2 globals, student uses all crops.
        """
        # teacher outputs: only first 2 (globals)
        teacher_probs = []
        for t in teacher_logits[:2]:
            t = (t - self.center) / self.teacher_temp
            teacher_probs.append(F.softmax(t, dim=-1))
        # student logits for all crops
        student_logp = [F.log_softmax(s / self.student_temp, dim=-1) for s in student_logits]

        total_loss = 0.0
        n_terms = 0
        for iq, q in enumerate(teacher_probs):  # teacher global views
            for iv, v in enumerate(student_logp):  # student all views
                if iv == iq:
                    # skip matching view index for globals (common practice)
                    continue
                # cross entropy: - sum q * log p
                loss = torch.sum(-q * v, dim=-1).mean()
                total_loss += loss
                n_terms += 1

        total_loss = total_loss / max(n_terms, 1)

        # update center with concatenated teacher outputs (globals)
        with torch.no_grad():
            t_cat = torch.cat([t.detach() for t in teacher_logits[:2]], dim=0)
            self.update_center(t_cat)

        return total_loss


# -----------------------------
# Linear probe (age regression)
# -----------------------------
class LinearRegressor(nn.Module):
    def __init__(self, in_dim: int):
        super().__init__()
        self.fc = nn.Linear(in_dim, 1)

    def forward(self, x):
        return self.fc(x).squeeze(-1)


@torch.no_grad()
def extract_embeddings(
    backbone: nn.Module,
    loader: DataLoader,
    device: torch.device,
    max_batches: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Extract embeddings and ages for linear probe.
    Uses the first global crop (224) only if dataset returns multi-crop.
    """
    backbone.eval()
    feats = []
    ys = []
    for bi, batch in enumerate(loader):
        x, y, _rid = batch  # x may be list of crops or tensor
        if isinstance(x, list):
            x = x[0]  # first global crop
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        f = backbone(x)
        if isinstance(f, (tuple, list)):
            f = f[0]
        feats.append(f.detach().float().cpu())
        ys.append(y.detach().float().cpu())
        if max_batches is not None and (bi + 1) >= max_batches:
            break
    return torch.cat(feats, dim=0), torch.cat(ys, dim=0)


def train_linear_probe(
    backbone: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    in_dim: int,
    epochs: int = 5,
    lr: float = 1e-2,
    weight_decay: float = 1e-4,
    max_train_batches: Optional[int] = None,
    max_val_batches: Optional[int] = None,
) -> Tuple[float, float]:
    """
    Train a tiny linear regressor on frozen embeddings.
    Returns: (val_mae, val_rmse)
    """
    backbone.eval()
    for p in backbone.parameters():
        p.requires_grad = False

    # Extract embeddings (CPU tensors)
    Xtr, ytr = extract_embeddings(backbone, train_loader, device, max_batches=max_train_batches)
    Xva, yva = extract_embeddings(backbone, val_loader, device, max_batches=max_val_batches)

    model = LinearRegressor(in_dim).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    # simple standardization (optional but helps regression)
    mu = Xtr.mean(dim=0, keepdim=True)
    sig = Xtr.std(dim=0, keepdim=True).clamp_min(1e-6)
    Xtr_n = (Xtr - mu) / sig
    Xva_n = (Xva - mu) / sig

    Xtr_n = Xtr_n.to(device)
    ytr = ytr.to(device)
    Xva_n = Xva_n.to(device)
    yva = yva.to(device)

    model.train()
    bs = 2048 if Xtr_n.shape[0] >= 2048 else Xtr_n.shape[0]
    for _ in range(epochs):
        # minibatch training on embeddings
        idx = torch.randperm(Xtr_n.shape[0], device=device)
        for i in range(0, Xtr_n.shape[0], bs):
            j = idx[i:i + bs]
            pred = model(Xtr_n[j])
            loss = F.smooth_l1_loss(pred, ytr[j])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

    model.eval()
    with torch.no_grad():
        pred = model(Xva_n)
        mae = torch.mean(torch.abs(pred - yva)).item()
        rmse = torch.sqrt(torch.mean((pred - yva) ** 2)).item()
    return mae, rmse


# -----------------------------
# Backbone loader (DINOv2 ViT-B/14)
# -----------------------------
def load_dinov2_vitb14(pretrained_ckpt: Optional[Path] = None) -> nn.Module:
    """
    Returns a backbone that outputs an embedding per image.
    Default: torch.hub load from facebookresearch/dinov2.
    If offline, pass a local ckpt and load_state_dict yourself (adjust as needed).
    """
    # backbone: returns CLS token embedding of dim 768 (for ViT-B/14)
    if pretrained_ckpt is None:
        model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14")
        return model
    else:
        # Fallback: load architecture from hub, then load weights.
        model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14")
        ckpt = torch.load(str(pretrained_ckpt), map_location="cpu")
        state = ckpt.get("model", ckpt)
        missing, unexpected = model.load_state_dict(state, strict=False)
        print(f"[load ckpt] missing={len(missing)} unexpected={len(unexpected)}")
        return model


# -----------------------------
# Freeze policy
# -----------------------------
def apply_freeze_policy(backbone: nn.Module, freeze_patch_embed=True, freeze_blocks_upto: int = 5):
    """
    Freeze patch embed and blocks[0..freeze_blocks_upto] for ViT.
    This assumes the model has attributes similar to ViT: patch_embed, blocks (ModuleList), norm, etc.
    """
    # patch embed
    if freeze_patch_embed and hasattr(backbone, "patch_embed"):
        for p in backbone.patch_embed.parameters():
            p.requires_grad = False

    # blocks
    if hasattr(backbone, "blocks"):
        blocks = backbone.blocks
        for i, blk in enumerate(blocks):
            if i <= freeze_blocks_upto:
                for p in blk.parameters():
                    p.requires_grad = False


# -----------------------------
# EMA update for teacher
# -----------------------------
@torch.no_grad()
def ema_update(teacher: nn.Module, student: nn.Module, m: float):
    t_params = dict(teacher.named_parameters())
    for name, s_p in student.named_parameters():
        if name in t_params:
            t_p = t_params[name]
            if t_p.requires_grad is False:
                # even if frozen, keep synced via EMA (harmless)
                pass
            t_p.data.mul_(m).add_(s_p.data, alpha=(1.0 - m))


# -----------------------------
# Training loop
# -----------------------------
def train(args):
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("CUDA is required for practical training.")

    runs_root = Path(args.runs_root)
    run_dir = runs_root / (args.run_name or now_run_name())
    ckpt_dir = run_dir / "checkpoints"
    log_dir = run_dir / "logs"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    # Load labels and index data
    label_map = load_label_csv(Path(args.label_csv))
    samples, patient_to_indices = build_index(Path(args.data_root), label_map)

    patient_ids = sorted(list(patient_to_indices.keys()))
    train_p, val_p, test_p = split_patients(patient_ids, seed=args.seed, train_ratio=args.train_ratio, val_ratio=args.val_ratio)

    # Create subset indices for each split
    def subset_indices(patients: List[str]) -> List[int]:
        out = []
        for rid in patients:
            out.extend(patient_to_indices[rid])
        return out

    train_indices = subset_indices(train_p)
    val_indices = subset_indices(val_p)
    test_indices = subset_indices(test_p)

    # Datasets
    ssl_transform = MultiCropTransform(global_size=224, local_size=args.local_size, n_global=2, n_local=4)
    # For linear probe we can reuse same dataset but only take first crop in extract_embeddings()
    train_ds = EndoscopyDataset([samples[i] for i in train_indices], transform=ssl_transform)
    val_ds = EndoscopyDataset([samples[i] for i in val_indices], transform=ssl_transform)
    test_ds = EndoscopyDataset([samples[i] for i in test_indices], transform=ssl_transform)

    # Steps per epoch: define as "see ~all images once" style
    steps_per_epoch = max(1, int(math.ceil(len(samples) / args.batch_size)))

    # Sampler: patient-balanced for training
    train_sampler = PatientBatchSampler(
        patient_to_indices={rid: [train_indices.index(i) for i in patient_to_indices[rid] if i in train_indices]
                            for rid in train_p},
        patient_ids=train_p,
        batch_size=args.batch_size,
        k_per_patient=args.k_per_patient,
        steps_per_epoch=steps_per_epoch,
        seed=args.seed,
    )
    # The above mapping is a bit tricky because train_ds is already subset.
    # Rebuild a correct map for train_ds indices:
    rid_to_train_ds_indices: Dict[str, List[int]] = {}
    for j, s in enumerate(train_ds.samples):
        rid_to_train_ds_indices.setdefault(s.rid, []).append(j)
    train_sampler = PatientBatchSampler(
        patient_to_indices=rid_to_train_ds_indices,
        patient_ids=train_p,
        batch_size=args.batch_size,
        k_per_patient=args.k_per_patient,
        steps_per_epoch=steps_per_epoch,
        seed=args.seed,
    )

    # DataLoaders
    # Train uses batch_sampler (provides already a list of indices)
    train_loader = DataLoader(
        train_ds,
        batch_sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=(args.num_workers > 0),
        collate_fn=lambda batch: (
            [torch.stack([b[0][i] for b in batch], dim=0) for i in range(len(batch[0][0]))],  # list of crops
            torch.tensor([b[1] for b in batch], dtype=torch.float32),
            [b[2] for b in batch],
        ),
    )

    # Val/Test for embeddings: normal batching
    eval_loader_kwargs = dict(
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=(args.num_workers > 0),
        collate_fn=lambda batch: (
            [torch.stack([b[0][i] for b in batch], dim=0) for i in range(len(batch[0][0]))],
            torch.tensor([b[1] for b in batch], dtype=torch.float32),
            [b[2] for b in batch],
        ),
    )
    val_loader = DataLoader(val_ds, **eval_loader_kwargs)
    test_loader = DataLoader(test_ds, **eval_loader_kwargs)

    # Load backbone
    backbone_s = load_dinov2_vitb14(pretrained_ckpt=Path(args.pretrained_ckpt) if args.pretrained_ckpt else None)
    backbone_t = load_dinov2_vitb14(pretrained_ckpt=Path(args.pretrained_ckpt) if args.pretrained_ckpt else None)

    # Infer embedding dim: ViT-B/14 typically 768
    with torch.no_grad():
        dummy = torch.randn(2, 3, 224, 224)
        emb_dim = backbone_s(dummy).shape[-1]

    # Wrap with DINO heads
    student = MultiCropWrapper(backbone_s, DINOHead(in_dim=emb_dim, out_dim=args.dino_out_dim)).to(device)
    teacher = MultiCropWrapper(backbone_t, DINOHead(in_dim=emb_dim, out_dim=args.dino_out_dim)).to(device)

    # Freeze policy on student backbone only (teacher is EMA)
    apply_freeze_policy(student.backbone, freeze_patch_embed=True, freeze_blocks_upto=args.freeze_blocks_upto)
    # teacher not trained by gradients
    for p in teacher.parameters():
        p.requires_grad = False

    # Optimizer on trainable params
    trainable = [p for p in student.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay)

    # LR schedule: warmup + cosine
    total_steps = args.epochs * steps_per_epoch
    warmup_steps = int(total_steps * args.warmup_ratio)

    def lr_at(step: int) -> float:
        if step < warmup_steps:
            return args.lr * (step + 1) / max(1, warmup_steps)
        # cosine to min_lr
        t = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return args.min_lr + 0.5 * (args.lr - args.min_lr) * (1 + math.cos(math.pi * t))

    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda" and not is_bf16_supported()))
    use_bf16 = (device.type == "cuda" and is_bf16_supported() and args.amp == "bf16")
    amp_dtype = torch.bfloat16 if use_bf16 else torch.float16

    dino_loss = DINOLoss(out_dim=args.dino_out_dim, teacher_temp=args.teacher_temp).to(device)

    # Save config
    cfg = vars(args).copy()
    cfg.update({
        "run_dir": str(run_dir),
        "steps_per_epoch": steps_per_epoch,
        "total_steps": total_steps,
        "warmup_steps": warmup_steps,
        "train_patients": len(train_p),
        "val_patients": len(val_p),
        "test_patients": len(test_p),
        "train_images": len(train_ds),
        "val_images": len(val_ds),
        "test_images": len(test_ds),
        "emb_dim": emb_dim,
        "bf16_supported": is_bf16_supported(),
    })
    save_json(cfg, run_dir / "config.json")

    # Training
    global_step = 0
    best_val_mae = float("inf")

    # teacher init = student (copy weights)
    teacher.load_state_dict(student.state_dict(), strict=False)

    for epoch in range(args.epochs):
        train_sampler.set_epoch(epoch)
        student.train()
        teacher.eval()

        t0 = time.time()
        loss_meter = 0.0

        pbar = tqdm(enumerate(train_loader), total=steps_per_epoch, dynamic_ncols=True)
        for it, (crops, _ages, _rids) in pbar:
            crops = [c.to(device, non_blocking=True) for c in crops]

            lr = lr_at(global_step)
            for pg in opt.param_groups:
                pg["lr"] = lr

            with torch.cuda.amp.autocast(enabled=True, dtype=amp_dtype):
                student_out = student(crops)
                with torch.no_grad():
                    teacher_out = teacher(crops[:2])
                loss = dino_loss(student_out, teacher_out)

            opt.zero_grad(set_to_none=True)
            if scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(trainable, args.grad_clip)
                scaler.step(opt)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(trainable, args.grad_clip)
                opt.step()

            m = 1.0 - (1.0 - args.ema_momentum) * (math.cos(math.pi * global_step / max(1, total_steps)) + 1) / 2
            ema_update(teacher, student, m)

            loss_meter += loss.item()
            global_step += 1

            # tqdm表示更新（表示頻度はここで調整可能）
            if (it + 1) % args.print_every == 0 or (it + 1) == steps_per_epoch:
                avg = loss_meter / (it + 1)
                pbar.set_postfix({
                    "loss": f"{avg:.4f}",
                    "lr": f"{lr:.2e}",
                    "m": f"{m:.4f}",
                })

        epoch_time = time.time() - t0
        train_loss = loss_meter / max(1, steps_per_epoch)

        # --- Linear probe evaluation (val) ---
        # Use backbone only (student.backbone)
        # To keep evaluation light, you can cap batches via args.lp_max_* if needed.
        val_mae, val_rmse = train_linear_probe(
            backbone=student.backbone,
            train_loader=DataLoader(train_ds, batch_size=args.eval_batch_size, shuffle=True,
                                    num_workers=args.num_workers, pin_memory=True,
                                    collate_fn=eval_loader_kwargs["collate_fn"]),
            val_loader=DataLoader(val_ds, batch_size=args.eval_batch_size, shuffle=False,
                                  num_workers=args.num_workers, pin_memory=True,
                                  collate_fn=eval_loader_kwargs["collate_fn"]),
            device=device,
            in_dim=emb_dim,
            epochs=args.lp_epochs,
            lr=args.lp_lr,
            weight_decay=args.lp_wd,
            max_train_batches=args.lp_max_train_batches,
            max_val_batches=args.lp_max_val_batches,
        )

        # logging
        log = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_mae": val_mae,
            "val_rmse": val_rmse,
            "epoch_time_sec": epoch_time,
            "global_step": global_step,
        }
        with open(log_dir / "metrics.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(log, ensure_ascii=False) + "\n")

        print(f"[epoch {epoch+1}] train_loss={train_loss:.4f} val_mae={val_mae:.3f} val_rmse={val_rmse:.3f} "
              f"time={epoch_time/60:.1f}min")

        # --- Checkpointing ---
        ckpt = {
            "epoch": epoch + 1,
            "global_step": global_step,
            "student": student.state_dict(),
            "teacher": teacher.state_dict(),
            "optimizer": opt.state_dict(),
            "scaler": scaler.state_dict() if scaler.is_enabled() else None,
            "best_val_mae": best_val_mae,
            "args": vars(args),
        }
        torch.save(ckpt, ckpt_dir / "last.pt")

        # rotate epoch checkpoints (keep last N)
        if args.keep_last_k > 0:
            ep_path = ckpt_dir / f"epoch_{epoch+1:03d}.pt"
            torch.save(ckpt, ep_path)
            # remove older
            old = sorted(ckpt_dir.glob("epoch_*.pt"))
            if len(old) > args.keep_last_k:
                for p in old[:-args.keep_last_k]:
                    try:
                        p.unlink()
                    except Exception:
                        pass

        # best
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            torch.save(ckpt, ckpt_dir / "best.pt")

    # Final test (optional): run linear probe and evaluate on test set
    test_mae, test_rmse = train_linear_probe(
        backbone=student.backbone,
        train_loader=DataLoader(train_ds, batch_size=args.eval_batch_size, shuffle=True,
                                num_workers=args.num_workers, pin_memory=True,
                                collate_fn=eval_loader_kwargs["collate_fn"]),
        val_loader=DataLoader(test_ds, batch_size=args.eval_batch_size, shuffle=False,
                              num_workers=args.num_workers, pin_memory=True,
                              collate_fn=eval_loader_kwargs["collate_fn"]),
        device=device,
        in_dim=emb_dim,
        epochs=args.lp_epochs,
        lr=args.lp_lr,
        weight_decay=args.lp_wd,
        max_train_batches=args.lp_max_train_batches,
        max_val_batches=args.lp_max_val_batches,
    )
    print(f"[final] test_mae={test_mae:.3f} test_rmse={test_rmse:.3f}")
    save_json({"best_val_mae": best_val_mae, "final_test_mae": test_mae, "final_test_rmse": test_rmse},
              run_dir / "summary.json")


def get_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=str, required=True)
    ap.add_argument("--label_csv", type=str, required=True)
    ap.add_argument("--runs_root", type=str, required=True)
    ap.add_argument("--run_name", type=str, default="")
    ap.add_argument("--pretrained_ckpt", type=str, default="", help="offline用: dinov2_vitb14の重みパス（任意）")

    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch_size", type=int, default=256, help="総画像枚数/step。8枚/患者なので患者数=bs/8")
    ap.add_argument("--k_per_patient", type=int, default=8)
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--eval_batch_size", type=int, default=128)

    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--train_ratio", type=float, default=0.8)
    ap.add_argument("--val_ratio", type=float, default=0.1)

    # Multi-crop
    ap.add_argument("--local_size", type=int, default=96)

    # DINO params
    ap.add_argument("--dino_out_dim", type=int, default=65536)
    ap.add_argument("--teacher_temp", type=float, default=0.04)
    ap.add_argument("--ema_momentum", type=float, default=0.996)

    # Optim
    ap.add_argument("--lr", type=float, default=2e-5, help="max effective LR (追加事前学習向けに小さめ)")
    ap.add_argument("--min_lr", type=float, default=1e-6)
    ap.add_argument("--warmup_ratio", type=float, default=0.05)
    ap.add_argument("--weight_decay", type=float, default=0.04)
    ap.add_argument("--grad_clip", type=float, default=1.0)

    # Freeze: freeze patch_embed and blocks 0..freeze_blocks_upto
    ap.add_argument("--freeze_blocks_upto", type=int, default=5)

    # AMP
    ap.add_argument("--amp", type=str, default="bf16", choices=["bf16", "fp16"])

    # Linear probe eval controls
    ap.add_argument("--lp_epochs", type=int, default=5)
    ap.add_argument("--lp_lr", type=float, default=1e-2)
    ap.add_argument("--lp_wd", type=float, default=1e-4)
    ap.add_argument("--lp_max_train_batches", type=int, default=0, help="0なら全て。軽量化なら例:50")
    ap.add_argument("--lp_max_val_batches", type=int, default=0, help="0なら全て。軽量化なら例:50")

    # Checkpoints
    ap.add_argument("--keep_last_k", type=int, default=5)
    ap.add_argument("--print_every", type=int, default=20)
    return ap.parse_args()


if __name__ == "__main__":
    args = get_args()
    # convert 0 to None
    if args.lp_max_train_batches == 0:
        args.lp_max_train_batches = None
    if args.lp_max_val_batches == 0:
        args.lp_max_val_batches = None
    train(args)