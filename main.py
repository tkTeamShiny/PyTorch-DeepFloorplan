# ============================================
# main.py — PyTorch-DeepFloorplan（全面改修版）
#  - 重要修正点:
#    * 画像テンソルが uint8 のまま Conv2d に入って落ちる問題を根絶
#      → モデルに渡す直前で float32 化 + /255 + ImageNet正規化 + 1ch→3chを強制
#    * VGG16 の weights は net.py 側ですでに新API化（VGG16_Weights.DEFAULT）
#  - 既存の Dataset/Loader を優先利用（dataset.py に build_dataloaders があれば使う）
#    無ければ簡易フォールバックのデータセットで動作
# ============================================

import os
import sys
import gc
import math
import json
import time
import random
import argparse
from pathlib import Path
from typing import Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.transforms import functional as TF
from torch.utils.data import Dataset, DataLoader

# ------------------------------
# 乱数固定（再現性）
# ------------------------------
def set_seed(seed: int = 42):
    import random, numpy as np, torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# ------------------------------
# 画像前処理（VGG想定）
# ------------------------------
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

def prepare_inputs(im: torch.Tensor) -> torch.Tensor:
    """
    im: (B, C, H, W) 期待、C は 1 or 3、dtype は uint8/float どちらでもOK。
    1) float32化 + /255（uint8由来の 0〜255 を 0〜1 に）
    2) C==1 の場合は 3ch に複製
    3) ImageNet 正規化（VGG16の事前学習重みに整合）
    """
    if im.dtype != torch.float32:
        im = im.float().div_(255.0)
    if im.dim() == 4 and im.size(1) == 1:
        im = im.repeat(1, 3, 1, 1)
    im = TF.normalize(im, IMAGENET_MEAN, IMAGENET_STD)
    return im

# ------------------------------
# データローダ（既存が無い場合のフォールバック）
#  ※あなたの環境に dataset.py / build_dataloaders(args) がある場合は
#    そちらを優先的に使います（下で try-import）。
# ------------------------------
class SimpleMaskDataset(Dataset):
    """
    期待するディレクトリ構造（例）:
        root/
          images/*.png(jpg)
          masks_cw/*.png  (クラスIDが画素値)
          masks_r/*.png   (クラスIDが画素値)
    - 画像: HxW または HxWx3 の8bit
    - マスク: HxW の8bit（クラスID）
    """
    def __init__(self, root: str, split: str = "train"):
        from PIL import Image
        self.root = Path(root)
        # ここでは train/val をフォルダでは分けず、簡易に 8:2 split
        img_dir = self.root / "images"
        cw_dir  = self.root / "masks_cw"
        r_dir   = self.root / "masks_r"
        paths = sorted(list(img_dir.glob("*.png")) + list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.jpeg")))
        if len(paths) == 0:
            raise FileNotFoundError(f"no images under: {img_dir}")

        # 固定シードで分割
        rng = np.random.RandomState(123)
        idx = np.arange(len(paths))
        rng.shuffle(idx)
        n_val = max(1, len(paths) // 5)
        val_set = set(idx[:n_val].tolist())
        if split == "train":
            self.paths = [paths[i] for i in range(len(paths)) if i not in val_set]
        else:
            self.paths = [paths[i] for i in range(len(paths)) if i in val_set]

        self.Image = Image
        self.cw_dir = cw_dir
        self.r_dir  = r_dir

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        import numpy as np
        from PIL import Image

        p = self.paths[i]
        # 画像（RGBで読む）
        img = Image.open(p).convert("RGB")
        img = np.array(img, dtype=np.uint8)  # (H,W,3) uint8

        # 対応するマスク（ファイル名ベース）
        base = p.stem
        cw_p = self.cw_dir / f"{base}.png"
        r_p  = self.r_dir  / f"{base}.png"
        if not cw_p.exists() or not r_p.exists():
            raise FileNotFoundError(f"missing mask(s) for {p.name}")

        cw = np.array(Image.open(cw_p), dtype=np.uint8)  # (H,W) uint8: class id
        r  = np.array(Image.open(r_p),  dtype=np.uint8)  # (H,W) uint8: class id

        # To Tensor
        # 画像: (H,W,3) -> (3,H,W) の uint8 tensor
        im_t = torch.from_numpy(img).permute(2, 0, 1).contiguous()  # uint8
        # マスク: long tensor (H,W)
        cw_t = torch.from_numpy(cw.astype(np.int64))
        r_t  = torch.from_numpy(r.astype(np.int64))

        # 既存 main と互換のタプル (im, cw, r, meta) を返す
        meta = {"name": base, "im_path": str(p)}
        return im_t, cw_t, r_t, meta


def build_dataloaders_fallback(args) -> Tuple[DataLoader, DataLoader]:
    root = args.data_root
    train_ds = SimpleMaskDataset(root, split="train")
    val_ds   = SimpleMaskDataset(root, split="val")

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
from net import DFPmodel  # ← あなたが置き換えた新API対応の net.py

# ------------------------------
# ロス・メトリクス
# ------------------------------
def compute_losses(logits_r, logits_cw, r_gt, cw_gt, w_r=1.0, w_cw=1.0):
    """
    - 部屋タイプ（r）: 9クラス想定 → CrossEntropyLoss
    - 壁/境界（cw）   : 3クラス想定 → CrossEntropyLoss
    ※クラス数は net.py の最終チャネル数に合わせる
    """
    ce_r  = nn.CrossEntropyLoss()
    ce_cw = nn.CrossEntropyLoss()

    # logits: (B, C, H, W), target: (B, H, W)
    loss_r  = ce_r(logits_r,  r_gt.long())
    loss_cw = ce_cw(logits_cw, cw_gt.long())
    loss = w_r * loss_r + w_cw * loss_cw
    return loss, {"loss_r": loss_r.item(), "loss_cw": loss_cw.item()}

@torch.no_grad()
def pixel_accuracy(logits, target):
    """
    単純な pixel acc
    """
    pred = logits.argmax(dim=1)  # (B,H,W)
    valid = (target >= 0)  # 欠損なし想定なので全部 True
    correct = (pred == target) & valid
    return correct.float().sum().item() / valid.float().sum().item()

# ------------------------------
# 学習・評価ループ
# ------------------------------
def train_one_epoch(model, loader, optimizer, scaler, device, args):
    model.train()
    accum_loss = 0.0
    n_samples = 0
    t0 = time.time()

    for idx, (im, cw, r, _) in enumerate(loader):
        # ---- 入力の前処理（ここが今回の重要ポイント）----
        im = prepare_inputs(im).to(device, non_blocking=True)
        cw = cw.to(device, non_blocking=True)
        r  = r.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        if args.amp:
            with torch.autocast(device_type=device.type, dtype=torch.float16 if device.type=='cuda' else torch.bfloat16):
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
            print(f"[train] iter {idx+1}/{len(loader)}  "
                  f"loss={loss.item():.4f}  "
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

    for idx, (im, cw, r, _) in enumerate(loader):
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
# 推論（可視化サンプル保存）
# ------------------------------
@torch.no_grad()
def run_inference_and_save_samples(model, loader, device, out_dir: str, limit: int = 10):
    from PIL import Image
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    for im, cw, r, meta in loader:
        im_n = im.clone()  # 可視化用にオリジナル保持（uint8想定）
        im = prepare_inputs(im).to(device, non_blocking=True)
        logits_r, logits_cw = model(im)
        pred_r  = logits_r.argmax(dim=1).cpu().numpy()   # (B,H,W)
        pred_cw = logits_cw.argmax(dim=1).cpu().numpy()  # (B,H,W)

        im_np = im_n.permute(0,2,3,1).cpu().numpy()  # (B,H,W,C) uint8
        cw_np = cw.cpu().numpy()
        r_np  = r.cpu().numpy()

        B = im_np.shape[0]
        for b in range(B):
            name = meta.get("name", f"sample_{count}")
            canvas = compose_triptych(im_np[b], r_np[b], pred_r[b])
            Image.fromarray(canvas).save(out_dir / f"{name}_triptych.png")
            count += 1
            if count >= limit:
                return

def colorize_mask(mask: np.ndarray, num_classes: int) -> np.ndarray:
    """
    単純なカラーマップで可視化（0..K-1）
    """
    import matplotlib
    cmap = matplotlib.cm.get_cmap("tab20", num_classes)
    h, w = mask.shape
    out = (cmap(mask % num_classes)[:, :, :3] * 255).astype(np.uint8)
    return out

def compose_triptych(rgb_uint8: np.ndarray, gt_mask: np.ndarray, pred_mask: np.ndarray) -> np.ndarray:
    """
    問題（画像）・正解（r）・予測（r_pred）の3枚を横並びに。
    """
    H, W, _ = rgb_uint8.shape
    # クラス数は一応 16 色で表示
    gt_color   = colorize_mask(gt_mask, 16)
    pred_color = colorize_mask(pred_mask, 16)

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
    p.add_argument("--data_root", type=str, default="./data",
                   help="画像とマスクのルート（images/, masks_cw/, masks_r/）")
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

    # cuDNN 高速化
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    # ===== DataLoader =====
    # 既存の dataset.py があるなら優先利用
    used_external = False
    try:
        from dataset import build_dataloaders  # ユーザ実装を想定
        train_loader, val_loader = build_dataloaders(args)
        used_external = True
        print("[data] using dataset.build_dataloaders from your repo.")
    except Exception as e:
        print(f"[data] fallback dataloaders (reason: {e})")
        train_loader, val_loader = build_dataloaders_fallback(args)

    # ===== Model =====
    model = DFPmodel(pretrained=True, freeze=True)
    model = model.to(device)

    # ===== Optimizer =====
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # ===== Resume =====
    start_epoch = 1
    if args.resume and os.path.isfile(args.resume):
        load_checkpoint(model, optimizer, args.resume, device)
        # 再開エポックは読み出し済みだが、省略せず:
        try:
            ckpt = torch.load(args.resume, map_location=device)
            start_epoch = int(ckpt.get("epoch", 0)) + 1
        except Exception:
            pass

    # ===== AMP scaler =====
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)

    # ===== Train =====
    best_val = float("inf")
    os.makedirs(args.save_dir, exist_ok=True)

    for epoch in range(start_epoch, args.epochs + 1):
        print(f"\n===== Epoch {epoch}/{args.epochs} =====")
        tr_loss = train_one_epoch(model, train_loader, optimizer, scaler, device, args)
        val_loss, metrics = validate(model, val_loader, device, args)

        # ベスト更新で保存
        if val_loss < best_val - 1e-6:
            best_val = val_loss
            best_path = os.path.join(args.save_dir, "best.pth")
            save_checkpoint(model, optimizer, epoch, best_path)

        # 定期保存
        if (epoch % args.save_every) == 0:
            path = os.path.join(args.save_dir, f"epoch{epoch}.pth")
            save_checkpoint(model, optimizer, epoch, path)

    # ===== Inference sample =====
    print("\n[infer] generate sample triptychs (image / GT(r) / Pred(r))")
    run_inference_and_save_samples(model.eval(), val_loader, device, args.inference_dir, args.inference_limit)
    print("[done]")

# ------------------------------
# エントリポイント
# ------------------------------
if __name__ == "__main__":
    parser = build_argparser()
    args = parser.parse_args() if sys.argv[0].endswith(".py") else parser.parse_args([])
    main(args)
