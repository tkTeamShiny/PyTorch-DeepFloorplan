# ============================================
# main.py — PyTorch-DeepFloorplan（フラット配置 & 分割テキスト対応 版）
#  - フォルダ直下に: <ID>.jpg, <ID>_rooms.png, <ID>_wall.png がある前提
#  - r3d_train.txt / r3d_test.txt があれば分割に利用（無ければ全体から8:2で分割）
#  - Conv2dにuint8を渡す問題を根絶: model(im)直前で float32化, /255, 1ch→3ch, ImageNet正規化
#  - "dataset" というフォルダ名がPythonモジュール名と衝突するため、外部の dataset.py は使わず、
#    本ファイル内のデータセット実装のみで動作します
# ============================================

import os
import sys
import gc
import json
import time
import argparse
from pathlib import Path
from typing import Tuple, Optional, List, Dict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import functional as TF

# ------------------------------
# 乱数固定（再現性）
# ------------------------------
def set_seed(seed: int = 42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# ------------------------------
# 画像前処理（VGG16 事前学習重みと整合）
# ------------------------------
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

def prepare_inputs(im: torch.Tensor) -> torch.Tensor:
    """
    im: (B, C, H, W) 期待, Cは1 or 3, dtypeはuint8/floatどちらでもOK
      1) float32化 + /255
      2) C==1なら3chに複製
      3) ImageNet正規化
    """
    if im.dtype != torch.float32:
        im = im.float().div_(255.0)
    if im.dim() == 4 and im.size(1) == 1:
        im = im.repeat(1, 3, 1, 1)
    im = TF.normalize(im, IMAGENET_MEAN, IMAGENET_STD)
    return im

# ------------------------------
# データセット（フラット配置 & 分割テキスト対応）
# ------------------------------
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

def _read_id_list(txt_path: Path) -> List[str]:
    """r3d_train.txt / r3d_test.txt を読み、stem（拡張子無しID）リストを返す"""
    stems = []
    with open(txt_path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            # 例: "31857804.jpg" or "31857804"
            stem = Path(s).stem
            stems.append(stem)
    return stems

class FlatRoomWallDataset(Dataset):
    """
    ディレクトリ直下に以下の3種ファイルが混在する構成を想定:
      - 入力画像: <ID>.jpg / .png ...
      - 部屋マスク: <ID>_rooms.png
      - 壁/境界マスク: <ID>_wall.png
    分割:
      - data_root/r3d_train.txt / r3d_test.txt があればそれに従う
      - 無ければ全体IDから8:2で分割（train側は8、val側は2）
    """
    def __init__(self,
                 data_root: Path,
                 split: str,
                 rooms_suffix: str = "_rooms.png",
                 wall_suffix: str = "_wall.png",
                 explicit_ids: Optional[List[str]] = None):
        from PIL import Image  # 遅延import
        self.Image = Image
        self.root = Path(data_root)

        if not self.root.exists():
            raise FileNotFoundError(f"data_root not found: {self.root}")

        # 1) 画像IDを収集（<ID>.* が存在し、かつ _rooms / _wall が揃うもの）
        #    → まず画像候補を列挙
        images = {}
        for p in self.root.iterdir():
            if p.is_file() and p.suffix.lower() in IMG_EXTS:
                stem = p.stem
                images.setdefault(stem, []).append(p)

        # 2) ペア成立（rooms + wall）
        samples_all = {}
        for stem, img_paths in images.items():
            rooms_p = self.root / f"{stem}{rooms_suffix}"
            wall_p  = self.root / f"{stem}{wall_suffix}"
            if rooms_p.exists() and wall_p.exists():
                # 画像は複数拡張子の候補がありうるが、まずは最初のものを使う
                img_p = sorted(img_paths, key=lambda x: x.suffix.lower() != ".jpg")[0]
                samples_all[stem] = {"img": img_p, "rooms": rooms_p, "wall": wall_p}

        if len(samples_all) == 0:
            raise FileNotFoundError(
                f"No <ID>.jpg + <ID>{rooms_suffix} + <ID>{wall_suffix} triplets found under: {self.root}"
            )

        # 3) 分割（explicit_ids 優先 → r3d_train/test.txt → 80/20 split）
        stems_all = sorted(samples_all.keys())
        train_txt = self.root / "r3d_train.txt"
        test_txt  = self.root / "r3d_test.txt"

        if explicit_ids is not None:
            use_ids = explicit_ids
        elif train_txt.exists() or test_txt.exists():
            if split == "train" and train_txt.exists():
                use_ids = _read_id_list(train_txt)
            elif split in ("val", "test") and test_txt.exists():
                use_ids = _read_id_list(test_txt)
            else:
                # 片方しか無い等の中途半端な場合は、テキストのある側に合わせ、残りは逆側へ
                if train_txt.exists():
                    train_ids = set(_read_id_list(train_txt))
                    if split == "train":
                        use_ids = sorted(train_ids)
                    else:
                        use_ids = sorted([s for s in stems_all if s not in train_ids])
                else:
                    test_ids = set(_read_id_list(test_txt))
                    if split in ("val", "test"):
                        use_ids = sorted(test_ids)
                    else:
                        use_ids = sorted([s for s in stems_all if s not in test_ids])
        else:
            # 80/20 固定分割（安定した順序のためseed固定のshuffleは避け、単純にソート後スライス）
            n = len(stems_all)
            n_val = max(1, int(round(n * 0.2)))
            if split == "train":
                use_ids = stems_all[:-n_val] if n_val < n else stems_all
            else:
                use_ids = stems_all[-n_val:] if n_val < n else stems_all

        # 4) 実サンプルを構築
        self.samples = []
        miss = 0
        for sid in use_ids:
            rec = samples_all.get(sid)
            if rec is None:
                miss += 1
                continue
            self.samples.append({
                "name": sid,
                "img": rec["img"],
                "rooms": rec["rooms"],
                "wall": rec["wall"],
            })

        if len(self.samples) == 0:
            raise FileNotFoundError(f"{split}: no samples matched under {self.root} (after split resolution).")

        msg_split = f"[data] split={split:5s} total={len(self.samples)}"
        if miss > 0:
            msg_split += f"  (skipped {miss} ids not present as complete triplets)"
        print(msg_split)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        import numpy as np
        rec = self.samples[i]

        # 入力画像 (RGB 化)
        img = self.Image.open(rec["img"])
        if img.mode != "RGB":
            img = img.convert("RGB")
        img = np.array(img, dtype=np.uint8)  # (H,W,3)

        # マスク（8bitのクラスID）
        rooms = np.array(self.Image.open(rec["rooms"]), dtype=np.uint8)
        wall  = np.array(self.Image.open(rec["wall"]),  dtype=np.uint8)

        im_t   = torch.from_numpy(img).permute(2, 0, 1).contiguous()  # uint8 (3,H,W)
        rooms_t = torch.from_numpy(rooms.astype(np.int64))  # (H,W)
        wall_t  = torch.from_numpy(wall.astype(np.int64))   # (H,W)

        # このリポの従来タプル順に合わせる: (im, cw, r, meta) として返す
        meta = {"name": rec["name"], "im_path": str(rec["img"])}
        return im_t, wall_t, rooms_t, meta

def build_dataloaders(args) -> Tuple[DataLoader, DataLoader]:
    root = Path(args.data_root)
    # 明示ID（カンマ区切り文字列）指定があれば使う
    train_ids = args.train_ids.split(",") if args.train_ids else None
    val_ids   = args.val_ids.split(",") if args.val_ids else None

    train_ds = FlatRoomWallDataset(root, split="train",
                                   rooms_suffix=args.rooms_suffix,
                                   wall_suffix=args.wall_suffix,
                                   explicit_ids=train_ids)
    val_ds   = FlatRoomWallDataset(root, split="val",
                                   rooms_suffix=args.rooms_suffix,
                                   wall_suffix=args.wall_suffix,
                                   explicit_ids=val_ids)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.val_batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=False,
    )
    return train_loader, val_loader

# ------------------------------
# モデル
# ------------------------------
from net import DFPmodel  # ← 事前に差し替えた新API対応 net.py を使用

# ------------------------------
# ロス・メトリクス
# ------------------------------
def compute_losses(logits_r, logits_cw, r_gt, cw_gt, w_r=1.0, w_cw=1.0):
    ce_r  = nn.CrossEntropyLoss()
    ce_cw = nn.CrossEntropyLoss()
    loss_r  = ce_r(logits_r,  r_gt.long())
    loss_cw = ce_cw(logits_cw, cw_gt.long())
    loss = w_r * loss_r + w_cw * loss_cw
    return loss, {"loss_r": loss_r.item(), "loss_cw": loss_cw.item()}

@torch.no_grad()
def pixel_accuracy(logits, target):
    pred = logits.argmax(dim=1)  # (B,H,W)
    correct = (pred == target)
    return correct.float().mean().item()

# ------------------------------
# 学習・評価ループ
# ------------------------------
def train_one_epoch(model, loader, optimizer, scaler, device, args):
    model.train()
    accum_loss = 0.0
    n_samples = 0
    t0 = time.time()

    for idx, (im, cw, r, _) in enumerate(loader):
        im = prepare_inputs(im).to(device, non_blocking=True)
        cw = cw.to(device, non_blocking=True)
        r  = r.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        if args.amp:
            dtype = torch.float16 if device.type == 'cuda' else torch.bfloat16
            with torch.autocast(device_type=device.type, dtype=dtype):
                logits_r, logits_cw = model(im)
                loss, parts = compute_losses(logits_r, logits_cw, r, cw, args.w_r, args.w_cw)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits_r, logits_cw = model(im)
            loss, parts = compute_losses(logits_r, logits_cw, r, cw, args.w_r, args.w_cw)
            loss.backward()
            optimizer.step()

        bs = im.size(0)
        accum_loss += loss.item() * bs
        n_samples += bs

        if (idx + 1) % args.log_interval == 0:
            print(f"[train] iter {idx+1}/{len(loader)}  loss={loss.item():.4f}  "
                  f"(r={parts['loss_r']:.4f}, cw={parts['loss_cw']:.4f})")

    avg_loss = accum_loss / max(1, n_samples)
    dt = time.time() - t0
    print(f"[train] epoch done: loss={avg_loss:.4f}  time={dt:.1f}s")
    return avg_loss

@torch.no_grad()
def validate(model, loader, device, args):
    model.eval()
    accum_loss = 0.0
    n_samples = 0
    acc_r_sum = 0.0
    acc_cw_sum = 0.0

    for im, cw, r, _ in loader:
        im = prepare_inputs(im).to(device, non_blocking=True)
        cw = cw.to(device, non_blocking=True)
        r  = r.to(device, non_blocking=True)

        logits_r, logits_cw = model(im)
        loss, parts = compute_losses(logits_r, logits_cw, r, cw, args.w_r, args.w_cw)

        bs = im.size(0)
        accum_loss += loss.item() * bs
        n_samples += bs

        acc_r_sum  += pixel_accuracy(logits_r,  r)  * bs
        acc_cw_sum += pixel_accuracy(logits_cw, cw) * bs

    avg_loss = accum_loss / max(1, n_samples)
    acc_r    = acc_r_sum  / max(1, n_samples)
    acc_cw   = acc_cw_sum / max(1, n_samples)
    print(f"[valid] loss={avg_loss:.4f}  acc_r={acc_r:.4f}  acc_cw={acc_cw:.4f}")
    return avg_loss, {"acc_r": acc_r, "acc_cw": acc_cw}

# ------------------------------
# 重みの保存/読込
# ------------------------------
def save_checkpoint(model, optimizer, epoch, path: str):
    ckpt = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
    }
    torch.save(ckpt, path)
    print(f"[ckpt] saved: {path}")

def load_checkpoint(model, optimizer, path: str, device):
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model"])
    if optimizer is not None and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    print(f"[ckpt] loaded: {path} (epoch={ckpt.get('epoch','?')})")

# ------------------------------
# 推論（サンプル可視化保存：画像/GT(r)/Pred(r)）
# ------------------------------
@torch.no_grad()
def run_inference_and_save_samples(model, loader, device, out_dir: str, limit: int = 10):
    from PIL import Image
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    for im, cw, r, meta in loader:
        im_raw = im.clone()  # 可視化用（uint8想定）
        im = prepare_inputs(im).to(device, non_blocking=True)
        logits_r, logits_cw = model(im)
        pred_r  = logits_r.argmax(dim=1).cpu().numpy()   # (B,H,W)

        im_np = im_raw.permute(0,2,3,1).cpu().numpy()  # (B,H,W,C) uint8
        r_np  = r.cpu().numpy()

        B = im_np.shape[0]
        for b in range(B):
            name = meta.get("name", f"sample_{count}")
            canvas = compose_triptych(im_np[b], r_np[b], pred_r[b])
            Image.fromarray(canvas).save(out_dir / f"{name}_triptych.png")
            count += 1
            if count >= limit:
                return

def colorize_mask(mask: np.ndarray, num_classes: int = 16) -> np.ndarray:
    import matplotlib
    cmap = matplotlib.cm.get_cmap("tab20", num_classes)
    out = (cmap(mask % num_classes)[:, :, :3] * 255).astype(np.uint8)
    return out

def compose_triptych(rgb_uint8: np.ndarray, gt_mask: np.ndarray, pred_mask: np.ndarray) -> np.ndarray:
    H, W, _ = rgb_uint8.shape
    gt_color   = colorize_mask(gt_mask)
    pred_color = colorize_mask(pred_mask)
    pad = 5
    canvas = np.ones((H, W*3 + pad*2, 3), dtype=np.uint8) * 255
    canvas[:, 0:W] = rgb_uint8
    canvas[:, W+pad:W+pad+W] = gt_color
    canvas[:, W*2+pad*2:W*3+pad*2] = pred_color
    return canvas

# ------------------------------
# 引数
# ------------------------------
def build_argparser():
    p = argparse.ArgumentParser()
    # 既定は /content/PyTorch-DeepFloorplan/dataset
    p.add_argument("--data_root", type=str, default="./dataset",
                   help="フラット配置のディレクトリ（<ID>.jpg, <ID>_rooms.png, <ID>_wall.png が混在）")
    p.add_argument("--rooms_suffix", type=str, default="_rooms.png",
                   help="部屋マスクファイルのサフィックス（既定: _rooms.png）")
    p.add_argument("--wall_suffix", type=str, default="_wall.png",
                   help="壁/境界マスクファイルのサフィックス（既定: _wall.png）")
    p.add_argument("--train_ids", type=str, default="",
                   help="学習に使うIDをカンマ区切りで明示（例: '1,2,3'）。未指定なら自動/テキスト使用")
    p.add_argument("--val_ids", type=str, default="",
                   help="評価に使うIDをカンマ区切りで明示（例: '4,5'）。未指定なら自動/テキスト使用")

    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--val_batch_size", type=int, default=4)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--w_r", type=float, default=1.0, help="部屋タイプ loss 重み")
    p.add_argument("--w_cw", type=float, default=1.0, help="壁/境界 loss 重み")
    p.add_argument("--amp", action="store_true", help="自動混合精度を有効化")
    p.add_argument("--resume", type=str, default="", help="checkpoint パス（再開用）")
    p.add_argument("--save_dir", type=str, default="./checkpoints")
    p.add_argument("--save_every", type=int, default=5)
    p.add_argument("--log_interval", type=int, default=50)
    p.add_argument("--inference_dir", type=str, default="./inference_samples")
    p.add_argument("--inference_limit", type=int, default=10)
    return p

# ------------------------------
# main
# ------------------------------
def main(args):
    set_seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[env] device={device}")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    # DataLoader（フラット構成に最適化）
    train_loader, val_loader = build_dataloaders(args)

    # Model
    model = DFPmodel(pretrained=True, freeze=True).to(device)

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Resume
    start_epoch = 1
    if args.resume and os.path.isfile(args.resume):
        load_checkpoint(model, optimizer, args.resume, device)
        try:
            ckpt = torch.load(args.resume, map_location=device)
            start_epoch = int(ckpt.get("epoch", 0)) + 1
        except Exception:
            pass

    # AMP scaler
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)

    # Train
    best_val = float("inf")
    Path(args.save_dir).mkdir(parents=True, exist_ok=True)

    for epoch in range(start_epoch, args.epochs + 1):
        print(f"\n===== Epoch {epoch}/{args.epochs} =====")
        tr_loss = train_one_epoch(model, train_loader, optimizer, scaler, device, args)
        val_loss, metrics = validate(model, val_loader, device, args)

        if val_loss < best_val - 1e-6:
            best_val = val_loss
            best_path = str(Path(args.save_dir) / "best.pth")
            save_checkpoint(model, optimizer, epoch, best_path)

        if (epoch % args.save_every) == 0:
            path = str(Path(args.save_dir) / f"epoch{epoch}.pth")
            save_checkpoint(model, optimizer, epoch, path)

    # Inference samples
    print("\n[infer] generate sample triptychs (image / GT(r) / Pred(r))")
    run_inference_and_save_samples(model.eval(), val_loader, device, args.inference_dir, args.inference_limit)
    print("[done]")

# ------------------------------
# エントリポイント
# ------------------------------
if __name__ == "__main__":
    parser = build_argparser()
    args = parser.parse_args()
    main(args)
