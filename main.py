# ============================================
# main.py — PyTorch-DeepFloorplan
#   newyork/{train,test} + r3d_{train,test}.txt 構成に対応
#   画像/マスクを Dataset 内で 512x512 にリサイズ（可変: --input_size）
#   マスクは必ず 1ch(2D) に統一（RGBマスクも安全に処理）
#   Conv2d に uint8 が入って落ちる問題は model(im) 直前の prepare_inputs() で根絶
#   AMP は CUDA 環境でのみ自動使用（バージョン差も吸収）
# ============================================

import os
import sys
import time
import argparse
from pathlib import Path
from typing import Tuple, Optional, List, Dict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import functional as TF

# --------------------------------------------
# 再現性
# --------------------------------------------
def set_seed(seed: int = 42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# --------------------------------------------
# 前処理（VGG16 の事前学習に整合）
# --------------------------------------------
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

def prepare_inputs(im: torch.Tensor) -> torch.Tensor:
    """
    im: (B,C,H,W) 期待（C=1/3, dtypeはuint8/floatどちらでもOK）
      1) float32化 + /255
      2) C==1 → 3chに複製
      3) ImageNet 正規化
    """
    if im.dtype != torch.float32:
        im = im.float().div_(255.0)
    if im.dim() == 4 and im.size(1) == 1:
        im = im.repeat(1, 3, 1, 1)
    im = TF.normalize(im, IMAGENET_MEAN, IMAGENET_STD)
    return im

# --------------------------------------------
# データ探索ユーティリティ
# --------------------------------------------
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

def _read_id_list(txt_path: Path) -> List[str]:
    """r3d_train.txt / r3d_test.txt から stem(ID) リストを読む"""
    ids = []
    with open(txt_path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            ids.append(Path(s).stem)  # "train/318.jpg" -> "318"
    return ids

def _scan_triplets(bases: List[Path], rooms_suffix: str, wall_suffix: str) -> Dict[str, Dict[str, Path]]:
    """
    bases 配下（再帰）から、<ID>.jpg|png 画像と <ID>{rooms_suffix}, <ID>{wall_suffix} を収集。
    戻り値: stem -> {"img": Path, "rooms": Path, "wall": Path}
    """
    img_by_stem: Dict[str, Path] = {}
    rooms_by_stem: Dict[str, Path] = {}
    wall_by_stem: Dict[str, Path] = {}

    # 画像
    for base in bases:
        if not base.exists():
            continue
        for p in base.rglob("*"):
            if p.is_file() and p.suffix.lower() in IMG_EXTS:
                stem = p.stem
                prev = img_by_stem.get(stem)
                # .jpg を優先
                if prev is None or (prev.suffix.lower() != ".jpg" and p.suffix.lower() == ".jpg"):
                    img_by_stem[stem] = p

    # rooms マスク
    rlen = len(rooms_suffix)
    for base in bases:
        if not base.exists():
            continue
        for p in base.rglob(f"*{rooms_suffix}"):
            if p.is_file():
                rooms_by_stem[p.name[:-rlen]] = p

    # wall マスク
    wlen = len(wall_suffix)
    for base in bases:
        if not base.exists():
            continue
        for p in base.rglob(f"*{wall_suffix}"):
            if p.is_file():
                wall_by_stem[p.name[:-wlen]] = p

    stems = sorted(set(img_by_stem.keys()) & set(rooms_by_stem.keys()) & set(wall_by_stem.keys()))
    triplets = {s: {"img": img_by_stem[s], "rooms": rooms_by_stem[s], "wall": wall_by_stem[s]} for s in stems}
    return triplets

# --------------------------------------------
# Dataset（train/test サブフォルダ & r3d_*.txt 対応 + リサイズ/1ch化）
# --------------------------------------------
class NYRoomWallDataset(Dataset):
    """
    data_root: ./dataset/newyork を想定
      - 優先: data_root/r3d_train.txt, r3d_test.txt による分割
      - 無ければ:
          * data_root/train, data_root/test があれば split に応じて収集
          * 無ければ data_root を単一ベースとして 80/20 に分割
    ファイル命名:
      画像: <ID>.jpg|png
      部屋: <ID>_rooms.png  （rooms_suffix）
      壁:   <ID>_wall.png   （wall_suffix）
    __getitem__ 内で (W,H) = (input_size, input_size) に**リサイズ**し、
    マスクは必ず 1ch(2D) に統一します。
    返却タプル: (im, cw, r, meta)  ※cw=wall, r=rooms（元コード互換）
    """
    _warned_color_mask_once = False  # 3chマスクを見つけたら1回だけ警告

    def __init__(self, data_root: Path, split: str,
                 rooms_suffix: str = "_rooms.png",
                 wall_suffix: str = "_wall.png",
                 explicit_ids: Optional[List[str]] = None,
                 input_size: int = 512):
        from PIL import Image
        self.Image = Image
        self.root = Path(data_root)
        self.input_size = int(input_size)
        if not self.root.exists():
            raise FileNotFoundError(f"data_root not found: {self.root}")

        train_dir = self.root / "train"
        test_dir  = self.root / "test"
        train_txt = self.root / "r3d_train.txt"
        test_txt  = self.root / "r3d_test.txt"

        # ベースディレクトリ
        if train_dir.is_dir() or test_dir.is_dir():
            if split == "train":
                bases = [train_dir] if train_dir.is_dir() else [self.root]
            else:
                bases = [test_dir] if test_dir.is_dir() else [self.root]
        else:
            bases = [self.root]

        # 三つ組収集
        trip_all = _scan_triplets(bases, rooms_suffix, wall_suffix)
        if len(trip_all) == 0 and bases != [self.root]:
            trip_all = _scan_triplets([self.root], rooms_suffix, wall_suffix)
        if len(trip_all) == 0:
            raise FileNotFoundError(
                f"No (<ID>.jpg|png, <ID>{rooms_suffix}, <ID>{wall_suffix}) triplets found under: {bases}"
            )

        # 使用ID
        stems_all = sorted(trip_all.keys())
        if explicit_ids is not None:
            use_ids = [sid for sid in explicit_ids if sid in trip_all]
        elif train_txt.exists() or test_txt.exists():
            if split == "train" and train_txt.exists():
                ids = _read_id_list(train_txt)
                use_ids = [sid for sid in ids if sid in trip_all]
            elif split in ("val", "test") and test_txt.exists():
                ids = _read_id_list(test_txt)
                use_ids = [sid for sid in ids if sid in trip_all]
            else:
                if train_txt.exists():
                    tids = set(_read_id_list(train_txt))
                    use_ids = [sid for sid in stems_all if (sid in tids) == (split == "train")]
                else:
                    vids = set(_read_id_list(test_txt))
                    use_ids = [sid for sid in stems_all if (sid in vids) == (split in ("val", "test"))]
        else:
            n = len(stems_all)
            n_val = max(1, int(round(n * 0.2)))
            use_ids = stems_all[:-n_val] if (split == "train" and n_val < n) else \
                      (stems_all[-n_val:] if n_val < n else stems_all)

        if len(use_ids) == 0:
            raise FileNotFoundError(f"{split}: no usable ids after split resolution under {self.root}")

        self.samples = []
        for sid in use_ids:
            rec = trip_all.get(sid)
            if rec is None:
                continue
            self.samples.append({
                "name": sid,
                "img": rec["img"],
                "rooms": rec["rooms"],
                "wall": rec["wall"],
            })

        print(f"[data] base={[str(b) for b in bases]}  split={split:5s}  samples={len(self.samples)}")

    @staticmethod
    def _to_single_channel(arr: np.ndarray, name: str) -> np.ndarray:
        """
        入力: arr (H,W) or (H,W,3)
        出力: (H,W) への安全な変換
        - 3ch かつ R=G=B の場合 → その1chを採用
        - 3ch でチャネルが異なる場合 → 先頭チャネルを採用（1回だけ警告）
        """
        if arr.ndim == 2:
            return arr
        if arr.ndim == 3 and arr.shape[2] == 3:
            if np.all(arr[..., 0] == arr[..., 1]) and np.all(arr[..., 0] == arr[..., 2]):
                return arr[..., 0]
            else:
                if not NYRoomWallDataset._warned_color_mask_once:
                    print(f"[warn] color mask detected on '{name}'. Using channel-0 as labels. "
                          f"(Ensure your masks are index images to avoid label drift.)")
                    NYRoomWallDataset._warned_color_mask_once = True
                return arr[..., 0]
        raise ValueError(f"Unexpected mask shape for '{name}': {arr.shape}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        import numpy as np
        rec = self.samples[i]
        # 画像（RGB化）
        img = self.Image.open(rec["img"])
        if img.mode != "RGB":
            img = img.convert("RGB")
        # マスク（8bitクラスID）。P/L以外（RGB等）は L に変換して2D化を優先
        rooms_img = self.Image.open(rec["rooms"])
        if rooms_img.mode not in ("L", "P"):
            rooms_img = rooms_img.convert("L")
        wall_img  = self.Image.open(rec["wall"])
        if wall_img.mode not in ("L", "P"):
            wall_img = wall_img.convert("L")

        # ---- 固定解像度へリサイズ（画像=双線形 / マスク=最近傍）----
        size = (self.input_size, self.input_size)
        img = img.resize(size, self.Image.BILINEAR)
        rooms_img = rooms_img.resize(size, self.Image.NEAREST)
        wall_img  = wall_img.resize(size,  self.Image.NEAREST)

        img_np   = np.array(img,        dtype=np.uint8)  # (H,W,3)
        rooms_np = np.array(rooms_img,  dtype=np.uint8)  # (H,W) or (H,W,3)
        wall_np  = np.array(wall_img,   dtype=np.uint8)  # (H,W) or (H,W,3)

        # 必ず 2D 化
        rooms_np = self._to_single_channel(rooms_np, rec["rooms"].name)
        wall_np  = self._to_single_channel(wall_np,  rec["wall"].name)

        im_t = torch.from_numpy(img_np).permute(2, 0, 1).contiguous()   # uint8 (3,H,W)
        cw_t = torch.from_numpy(wall_np.astype(np.int64))               # cw = wall (H,W)
        r_t  = torch.from_numpy(rooms_np.astype(np.int64))              # r  = rooms (H,W)

        meta = {"name": rec["name"], "im_path": str(rec["img"])}
        return im_t, cw_t, r_t, meta

def build_dataloaders(args) -> Tuple[DataLoader, DataLoader]:
    root = Path(args.data_root)

    train_ids = args.train_ids.split(",") if args.train_ids else None
    val_ids   = args.val_ids.split(",") if args.val_ids else None

    train_ds = NYRoomWallDataset(root, split="train",
                                 rooms_suffix=args.rooms_suffix,
                                 wall_suffix=args.wall_suffix,
                                 explicit_ids=train_ids,
                                 input_size=args.input_size)
    val_ds   = NYRoomWallDataset(root, split="test" if args.use_test_as_val else "val",
                                 rooms_suffix=args.rooms_suffix,
                                 wall_suffix=args.wall_suffix,
                                 explicit_ids=val_ids,
                                 input_size=args.input_size)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True, drop_last=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.val_batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True, drop_last=False
    )
    return train_loader, val_loader

# --------------------------------------------
# モデル
# --------------------------------------------
from net import DFPmodel  # ← 新API対応 net.py を使用

# --------------------------------------------
# ロス・メトリクス
# --------------------------------------------
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
    return (pred == target).float().mean().item()

# --------------------------------------------
# 学習・評価
# --------------------------------------------
def train_one_epoch(model, loader, optimizer, scaler, device, args):
    model.train()
    accum_loss, n_samples = 0.0, 0
    t0 = time.time()

    for idx, (im, cw, r, _) in enumerate(loader):
        im = prepare_inputs(im).to(device, non_blocking=True)
        cw = cw.to(device, non_blocking=True)
        r  = r.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        if args.amp and device.type == "cuda":
            # 互換性の高い autocast（cuda 環境のみ）
            from torch.cuda.amp import autocast
            with autocast(dtype=torch.float16):
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
            print(f"[train] iter {idx+1}/{len(loader)} loss={loss.item():.4f} "
                  f"(r={parts['loss_r']:.4f}, cw={parts['loss_cw']:.4f})")

    avg_loss = accum_loss / max(1, n_samples)
    print(f"[train] epoch done: loss={avg_loss:.4f}  time={time.time()-t0:.1f}s")
    return avg_loss

@torch.no_grad()
def validate(model, loader, device, args):
    model.eval()
    accum_loss, n_samples = 0.0, 0
    acc_r_sum, acc_cw_sum = 0.0, 0.0

    for im, cw, r, _ in loader:
        im = prepare_inputs(im).to(device, non_blocking=True)
        cw = cw.to(device, non_blocking=True)
        r  = r.to(device, non_blocking=True)

        if args.amp and device.type == "cuda":
            from torch.cuda.amp import autocast
            with autocast(dtype=torch.float16):
                logits_r, logits_cw = model(im)
        else:
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
    print(f"[valid] loss={avg_loss:.4f} acc_r={acc_r:.4f} acc_cw={acc_cw:.4f}")
    return avg_loss, {"acc_r": acc_r, "acc_cw": acc_cw}

# --------------------------------------------
# チェックポイント
# --------------------------------------------
def save_checkpoint(model, optimizer, epoch, path: str):
    ckpt = {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "epoch": epoch}
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(ckpt, path)
    print(f"[ckpt] saved: {path}")

def load_checkpoint(model, optimizer, path: str, device):
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model"])
    if optimizer is not None and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    print(f"[ckpt] loaded: {path} (epoch={ckpt.get('epoch','?')})")

# --------------------------------------------
# 簡易推論（画像/GT(rooms)/Pred(rooms) を横並び保存）
# --------------------------------------------
@torch.no_grad()
def run_inference_and_save_samples(model, loader, device, out_dir: str, limit: int = 10):
    from PIL import Image
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    count = 0
    for im, cw, r, meta in loader:
        im_raw = im.clone()  # 可視化用に uint8 を保持（リサイズ後の形）
        im = prepare_inputs(im).to(device, non_blocking=True)
        logits_r, logits_cw = model(im)
        pred_r = logits_r.argmax(dim=1).cpu().numpy()  # (B,H,W)

        im_np = im_raw.permute(0,2,3,1).cpu().numpy()
        r_np  = r.cpu().numpy()

        B = im_np.shape[0]
        for b in range(B):
            name = meta.get("name", f"sample_{count}")
            canvas = compose_triptych(im_np[b], r_np[b], pred_r[b])
            Image.fromarray(canvas).save(out / f"{name}_triptych.png")
            count += 1
            if count >= limit:
                return

def colorize_mask(mask: np.ndarray, num_classes: int = 16) -> np.ndarray:
    import matplotlib
    cmap = matplotlib.cm.get_cmap("tab20", num_classes)
    return (cmap(mask % num_classes)[:, :, :3] * 255).astype(np.uint8)

def compose_triptych(rgb_uint8: np.ndarray, gt_mask: np.ndarray, pred_mask: np.ndarray) -> np.ndarray:
    H, W, _ = rgb_uint8.shape
    gt_color   = colorize_mask(gt_mask)
    pred_color = colorize_mask(pred_mask)
    pad = 5
    canvas = np.ones((H, W*3 + pad*2, 3), dtype=np.uint8) * 255
    canvas[:, 0:W]                 = rgb_uint8
    canvas[:, W+pad:W+pad+W]       = gt_color
    canvas[:, W*2+pad*2:W*3+pad*2] = pred_color
    return canvas

# --------------------------------------------
# 引数
# --------------------------------------------
def build_argparser():
    p = argparse.ArgumentParser()
    # 既定を newyork 配下に
    p.add_argument("--data_root", type=str, default="./dataset/newyork",
                   help="データルート（例: ./dataset/newyork）")
    p.add_argument("--rooms_suffix", type=str, default="_rooms.png",
                   help="部屋マスクのサフィックス（既定: _rooms.png）")
    p.add_argument("--wall_suffix", type=str, default="_wall.png",
                   help="壁/境界マスクのサフィックス（既定: _wall.png）")
    p.add_argument("--use_test_as_val", action="store_true",
                   help="val を test ディレクトリ/リストで代用する（既定: False=val）")

    p.add_argument("--input_size", type=int, default=512,
                   help="Dataset 内でのリサイズ解像度（正方形）。バッチ安定化のため固定値推奨")
    p.add_argument("--train_ids", type=str, default="",
                   help="学習IDをカンマ区切りで明示（未指定なら r3d_train.txt or 自動分割）")
    p.add_argument("--val_ids", type=str, default="",
                   help="評価IDをカンマ区切りで明示（未指定なら r3d_test.txt or 自動分割）")

    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--val_batch_size", type=int, default=4)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--w_r", type=float, default=1.0)
    p.add_argument("--w_cw", type=float, default=1.0)
    p.add_argument("--amp", action="store_true")
    p.add_argument("--resume", type=str, default="")
    p.add_argument("--save_dir", type=str, default="./checkpoints")
    p.add_argument("--save_every", type=int, default=5)
    p.add_argument("--log_interval", type=int, default=50)
    p.add_argument("--inference_dir", type=str, default="./inference_samples")
    p.add_argument("--inference_limit", type=int, default=10)
    return p

# --------------------------------------------
# main
# --------------------------------------------
def main(args):
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[env] device={device}")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    # Data
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

    # AMP（バージョン互換の GradScaler 準備）
    if args.amp and device.type == "cuda":
        try:
            from torch.amp import GradScaler  # 新APIがあれば使用
            scaler = GradScaler(enabled=True)
        except Exception:
            scaler = torch.cuda.amp.GradScaler(enabled=True)  # 旧APIにフォールバック
    else:
        class _NoScaler:
            def scale(self, x): return x
            def step(self, opt): opt.step()
            def update(self): pass
        scaler = _NoScaler()

    # Train
    best_val = float("inf")
    Path(args.save_dir).mkdir(parents=True, exist_ok=True)

    for epoch in range(start_epoch, args.epochs + 1):
        print(f"\n===== Epoch {epoch}/{args.epochs} =====")
        train_one_epoch(model, train_loader, optimizer, scaler, device, args)
        val_loss, _ = validate(model, val_loader, device, args)

        if val_loss < best_val - 1e-6:
            best_val = val_loss
            save_checkpoint(model, optimizer, epoch, str(Path(args.save_dir) / "best.pth"))
        if (epoch % args.save_every) == 0:
            save_checkpoint(model, optimizer, epoch, str(Path(args.save_dir) / f"epoch{epoch}.pth"))

    # Inference samples
    print("\n[infer] generate sample triptychs (image / GT(rooms) / Pred(rooms))")
    run_inference_and_save_samples(model.eval(), val_loader, device, args.inference_dir, args.inference_limit)
    print("[done]")

# --------------------------------------------
# エントリポイント
# --------------------------------------------
if __name__ == "__main__":
    parser = build_argparser()
    args = parser.parse_args()
    main(args)
