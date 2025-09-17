# ============================================
# deploy.py — PyTorch-DeepFloorplan (inference, Colab-friendly)
# - 学習チェックポイント(best.pth等) / 純state_dict の両方に対応
# - main.py と同じ前処理（float化→3ch→ImageNet正規化）
# - 画像は --image_path 未指定でも --search_root から自動選択
# - Colab で !python 実行でも見られるよう PNG を保存（--out_dir）
# - 保存時に画像dtypeをuint8へ正規化（OpenCV対応）
# ============================================

import os
import sys
import argparse
from pathlib import Path
from typing import Tuple

import numpy as np
import torch

# ---- Matplotlib は保存用バックエンドに ----
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import cv2

# utils
sys.path.append('./utils/')
from rgb_ind_convertor import ind2rgb, floorplan_fuse_map
from util import fill_break_line, flood_fill, refine_room_region

# model
from net import DFPmodel


# -------------------------
# main.py と合わせた前処理
# -------------------------
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

def prepare_inputs(im: torch.Tensor) -> torch.Tensor:
    """
    im: (B,C,H,W) 期待（C=1/3, dtypeはuint8/floatどちらでも可）
      1) float化 + /255
      2) 1ch → 3ch複製
      3) ImageNet 正規化
    """
    if im.dtype != torch.float32:
        im = im.float().div_(255.0)
    if im.dim() == 4 and im.size(1) == 1:
        im = im.repeat(1, 3, 1, 1)
    mean = torch.tensor(IMAGENET_MEAN, dtype=im.dtype, device=im.device).view(1,3,1,1)
    std  = torch.tensor(IMAGENET_STD,  dtype=im.dtype, device=im.device).view(1,3,1,1)
    im = (im - mean) / std
    return im


# -------------------------
# ログits → (H,W) 予測ID
# -------------------------
def BCHW2argmax(tensor: torch.Tensor) -> np.ndarray:
    """
    tensor: (B,C,H,W) ログits
    return: (H,W) のクラスID（argmax）
    """
    if tensor.size(0) != 1:
        tensor = tensor[0].unsqueeze(0)
    arr = tensor.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()  # (H,W,C)
    return np.argmax(arr, axis=2)  # (H,W)


# -------------------------
# チェックポイント読込
# -------------------------
def _load_state_dict_flex(path: str, device: torch.device):
    """
    学習時保存した dict(checkpoint) / 純state_dict の両方に対応
    """
    ckpt = torch.load(path, map_location=device)
    if isinstance(ckpt, dict):
        if "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict):
            return ckpt["state_dict"]
        if "model" in ckpt and isinstance(ckpt["model"], dict):
            return ckpt["model"]
    return ckpt  # 純 state_dict とみなす


# -------------------------
# 推論初期化
# -------------------------
def _pick_image(image_path: str, search_root: str) -> Tuple[np.ndarray, str]:
    """
    画像パスが未指定/存在しない場合、search_root から最初の画像を選択。
    戻り値: (RGB画像[numpy], 使用パス)
    """
    if image_path and os.path.isfile(image_path):
        use_path = image_path
    else:
        patterns = ["*.jpg", "*.png", "*.jpeg", "*.bmp", "*.tif", "*.tiff"]
        files = []
        root = Path(search_root)
        for pat in patterns:
            files.extend(root.rglob(pat))
        if not files:
            raise FileNotFoundError(
                f"No images found. Specify --image_path or put images under: {search_root}"
            )
        use_path = str(sorted(files)[0])
    bgr = cv2.imread(use_path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"Failed to read image: {use_path}")
    # モデル入力は 512x512（main.py と合わせる）
    bgr = cv2.resize(bgr, (512, 512), interpolation=cv2.INTER_LINEAR)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return rgb, use_path


def initialize(args):
    # device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # image
    orig_rgb, used_path = _pick_image(args.image_path, args.search_root)
    # (H,W,3) uint8 -> (1,3,H,W) uint8 tensor
    im_t = torch.from_numpy(orig_rgb).permute(2, 0, 1).unsqueeze(0)  # (1,3,512,512), uint8
    im_t = prepare_inputs(im_t).to(device, non_blocking=True)

    # model（pretrained=False で余計なDL回避。state_dict で上書き）
    model = DFPmodel(pretrained=False, freeze=False).to(device)
    state = _load_state_dict_flex(args.loadmodel, device)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"[warn] missing keys: {len(missing)} e.g. {missing[:8]}")
    if unexpected:
        print(f"[warn] unexpected keys: {len(unexpected)} e.g. {unexpected[:8]}")
    model.eval()

    return device, orig_rgb, im_t, model, used_path


# -------------------------
# 既存の後処理そのまま
# -------------------------
def post_process(rm_ind: np.ndarray, bd_ind: np.ndarray) -> np.ndarray:
    hard_c = (bd_ind > 0).astype(np.uint8)
    # region from room prediction
    rm_mask = np.zeros(rm_ind.shape, dtype=np.uint8)
    rm_mask[rm_ind > 0] = 1
    # region from close wall line
    cw_mask = hard_c
    # close wall mask の切れ目を補完
    cw_mask = fill_break_line(cw_mask)

    fuse_mask = cw_mask + rm_mask
    fuse_mask[fuse_mask >= 1] = 255

    # 穴埋め
    fuse_mask = flood_fill(fuse_mask)
    fuse_mask = fuse_mask // 255

    # 一領域一ラベル化
    new_rm_ind = refine_room_region(cw_mask, rm_ind)

    # 背景の誤ラベルを無視
    new_rm_ind = fuse_mask * new_rm_ind
    return new_rm_ind


# -------------------------
# OpenCV 保存のための dtype 正規化
# -------------------------
def to_uint8(arr: np.ndarray) -> np.ndarray:
    """OpenCV が扱えるように uint8 へ変換"""
    if arr.dtype == np.uint8:
        return arr
    if np.issubdtype(arr.dtype, np.floating):
        m = float(np.nanmax(arr)) if arr.size else 0.0
        if m <= 1.0 + 1e-6:  # 0..1 を想定
            arr = np.clip(arr, 0.0, 1.0) * 255.0
        else:
            arr = np.clip(arr, 0.0, 255.0)
        return arr.round().astype(np.uint8)
    # 整数系（int32等）は 0..255 にクリップして uint8 化
    return np.clip(arr, 0, 255).astype(np.uint8)


# -------------------------
# 可視化の保存（Colab対応）
# -------------------------
def save_visualizations(orig_rgb: np.ndarray,
                        pred_rooms_ind: np.ndarray,
                        pred_boundary_ind: np.ndarray,
                        out_dir: Path,
                        stem: str):
    out_dir.mkdir(parents=True, exist_ok=True)

    # Rooms を RGB に着色（→ uint8 化）
    rooms_rgb = ind2rgb(pred_rooms_ind, color_map=floorplan_fuse_map)
    rooms_rgb = to_uint8(rooms_rgb)

    # Triptych 図を保存（matplotlib）
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1); plt.title("Input");    plt.imshow(orig_rgb);          plt.axis("off")
    plt.subplot(1, 3, 2); plt.title("Rooms");    plt.imshow(rooms_rgb);         plt.axis("off")
    plt.subplot(1, 3, 3); plt.title("Boundary"); plt.imshow(pred_boundary_ind); plt.axis("off")
    plt.tight_layout()
    trip_path = out_dir / f"{stem}_triptych.png"
    plt.savefig(trip_path, dpi=150, bbox_inches="tight")
    plt.close()

    # 単体画像も保存（OpenCV は BGR なので変換）
    cv2.imwrite(str(out_dir / f"{stem}_input.jpg"),
                cv2.cvtColor(to_uint8(orig_rgb), cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(out_dir / f"{stem}_rooms_rgb.png"),
                cv2.cvtColor(rooms_rgb, cv2.COLOR_RGB2BGR))
    # Boundary はインデックス（モノクロ）
    maxv = int(pred_boundary_ind.max()) if pred_boundary_ind.size else 0
    scale = 255 // max(1, maxv)
    norm_boundary = (pred_boundary_ind.astype(np.int32) * scale)
    cv2.imwrite(str(out_dir / f"{stem}_boundary.png"),
                to_uint8(norm_boundary))

    print(f"[save] {trip_path}")
    print(f"[save] {out_dir / f'{stem}_rooms_rgb.png'}")
    print(f"[save] {out_dir / f'{stem}_boundary.png'}")


# -------------------------
# メイン
# -------------------------
def main(args):
    device, orig_rgb, image_t, model, used_path = initialize(args)
    print(f"[info] device={device}, image='{used_path}'")

    with torch.no_grad():
        logits_r, logits_cw = model(image_t)          # (1,Cr,H,W), (1,Cw,H,W)
        predroom = BCHW2argmax(logits_r)              # (H,W)
        predboundary = BCHW2argmax(logits_cw)         # (H,W)

    if args.postprocess:
        predroom = post_process(predroom, predboundary)

    stem = Path(used_path).stem
    out_dir = Path(args.out_dir)
    save_visualizations(orig_rgb, predroom, predboundary, out_dir, stem)

    # （ノートブックの Python 実行時のみ）希望があれば表示
    if args.show:
        img_path = out_dir / f"{stem}_triptych.png"
        try:
            from IPython.display import Image, display
            display(Image(filename=str(img_path)))
        except Exception:
            pass


# -------------------------
# 入口
# -------------------------
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--loadmodel", type=str, default="./checkpoints/best.pth",
                   help="学習済みチェックポイント or 純state_dict のパス")
    p.add_argument("--image_path", type=str, default="",
                   help="推論する画像ファイル。未指定なら --search_root から自動選択")
    p.add_argument("--search_root", type=str, default="./dataset/newyork/test",
                   help="--image_path 未指定時に画像を探すルートディレクトリ")
    p.add_argument("--postprocess", action="store_true",
                   help="後処理を有効化 (fill/flood/refine)")
    p.add_argument("--out_dir", type=str, default="./vis",
                   help="可視化の保存先ディレクトリ")
    p.add_argument("--show", action="store_true",
                   help="ノートブックから import 実行した場合に inline 表示する")

    args = p.parse_args()
    main(args)
