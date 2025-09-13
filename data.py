# data.py —— 自動インデックス生成（柔軟トークン対応）＋ 変換適用の“順に試す”最小パッチ

import os, re, glob, random
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from skimage.transform import rotate
import cv2
from PIL import Image  # torchvision対応（PIL<->np変換用）

# ===== 設定 =====
ALLOW_IMAGE_INPUT = True                         # True: .png/.jpg も許可
VALID_EXTS = ('.csv', '.png', '.jpg', '.jpeg')   # 対応拡張子
# 役割トークン（lower）
TOKENS = {
    'boundary': ['wall', 'boundary', 'cw', 'contour', 'edge'],
    'room':     ['rooms', 'room', 'r'],
    'door':     ['close', 'door', 'doors', 'd', 'opening'],
}
# ベース（画像）側に含まれていたら “画像ではない” とみなす否定語
NOT_IMAGE_HINTS = sum(TOKENS.values(), []) + ['mask', 'seg', 'label']

# ===== ユーティリティ =====
def _role_from_name(name_lower: str):
    for role, toks in TOKENS.items():
        for t in toks:
            if t in name_lower:
                return role
    return 'image'

def _strip_role_tokens(stem_lower: str):
    s = stem_lower
    for toks in TOKENS.values():
        for t in toks:
            s = re.sub(rf'[_\-\.]?{re.escape(t)}(?=$|[_\-\.])', '', s)
    # 連続した区切りの整理
    s = re.sub(r'[_\-\.]+$', '', s)
    return s

def _is_image_candidate(stem_lower: str):
    return not any(h in stem_lower for h in NOT_IMAGE_HINTS)

def _read_csv(path: str, size: int):
    exp3 = size * size * 3
    exp1 = size * size
    try:
        arr = np.loadtxt(path, dtype=np.uint8, delimiter=',')
    except Exception:
        arr = np.genfromtxt(path, dtype=np.uint8, delimiter=',')
    arr = np.asarray(arr).reshape(-1,)
    if arr.size == exp3:
        return arr.reshape(size, size, 3)
    if arr.size == exp1:
        return arr.reshape(size, size)
    raise ValueError(f'[CSV size mismatch] {path} -> {arr.size} (expected {exp3} or {exp1})')

def _imread_any(path: str, role: str, size: int):
    """画像/CSVどちらでも読み込む。必要ならリサイズ。role in {'image','boundary','room','door'}"""
    ext = os.path.splitext(path)[1].lower()
    if ext == '.csv':
        img = _read_csv(path, size)
        return img
    if not ALLOW_IMAGE_INPUT:
        raise FileNotFoundError(f'[Only-CSV mode] {path}')
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f'[IMG missing] {path}')
    # BGR -> RGB if 3ch
    if img.ndim == 3 and img.shape[2] == 3:
        img = img[:, :, ::-1]
    # リサイズ（サイズ違いなら）
    if img.ndim == 2:
        h, w = img.shape
    else:
        h, w, _ = img.shape
    if (h, w) != (size, size):
        if role == 'image':
            img = cv2.resize(img, (size, size), interpolation=cv2.INTER_LINEAR)
        else:
            img = cv2.resize(img, (size, size), interpolation=cv2.INTER_NEAREST)
    # マスクを3chで読んでしまった場合は 1chに落とす（簡便）
    if role != 'image' and img.ndim == 3 and img.shape[2] == 3:
        # 既に RGB にしているため COLOR_RGB2GRAY を用いる
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    return img

def _rotate_img(x, angle):
    y = rotate(x, angle, preserve_range=True, order=1, mode='edge')
    return y.astype(x.dtype)

def _rotate_mask(x, angle):
    y = rotate(x, angle, preserve_range=True, order=0, mode='edge')
    return np.rint(y).astype(x.dtype)

class MyRotationTransform:
    def __init__(self, angles=(0, 90, -90, 180)):
        self.angles = angles
    def __call__(self, image, boundary, room, door):
        a = random.choice(self.angles)
        return (_rotate_img(image, a),
                _rotate_mask(boundary, a),
                _rotate_mask(room, a),
                _rotate_mask(door, a))

# ===== 変換の“順に試す”適用（最小パッチの要） =====
def _apply_transform(transform, image, boundary, room, door):
    """transform を安全に適用するために順に試す:
       ① Albumentations 形式: transform(image=..., masks=[...])  -> dict返り値想定
       ② torchvision / 単一引数: transform(PIL/array) -> 画像のみ
       ③ 自作4引数: transform(image, boundary, room, door)
       ④ それでもダメならそのまま返す
    """
    if transform is None:
        return image, boundary, room, door

    # ① Albumentations 形式を試す（kwargs対応）
    try:
        b_i = boundary.astype(np.int32, copy=False)
        r_i = room.astype(np.int32, copy=False)
        d_i = door.astype(np.int32, copy=False)
        out = transform(image=image, masks=[b_i, r_i, d_i])
        if isinstance(out, dict) and 'image' in out and 'masks' in out:
            img = out['image']
            m1, m2, m3 = out['masks']
            return (img,
                    m1.astype(boundary.dtype, copy=False),
                    m2.astype(room.dtype, copy=False),
                    m3.astype(door.dtype, copy=False))
    except TypeError:
        # キーワードを受け付けない＝Albumentationsではない
        pass
    except Exception:
        # Albumentations と見なしても失敗するケースは次へ
        pass

    # ② torchvision / 単一引数（画像のみ変換）を試す
    try:
        # PIL 経由で最大互換
        pil = Image.fromarray(image.astype(np.uint8, copy=False)) if image.ndim == 3 else Image.fromarray(image.astype(np.uint8, copy=False))
        out = transform(pil)
        # Tensor or PIL -> np.ndarray(HWC)
        try:
            import torch as _torch
            if isinstance(out, _torch.Tensor):
                arr = out.detach().cpu().numpy()
                if arr.ndim == 3 and arr.shape[0] in (1, 3):
                    arr = np.moveaxis(arr, 0, 2)
                if arr.dtype.kind == 'f' and arr.max() <= 1.0:
                    arr = (arr * 255.0).clip(0, 255)
                img_np = arr.astype(np.uint8)
            else:
                img_np = np.array(out)
        except Exception:
            img_np = np.array(out)
        return img_np, boundary, room, door
    except TypeError:
        # torchvision ではない / imgのみの1引数ではない
        pass
    except Exception:
        pass

    # ③ 自作4引数 Transform を試す
    try:
        return transform(image, boundary, room, door)
    except TypeError:
        pass
    except Exception:
        pass

    # ④ 何も適用できなければ素通し
    return image, boundary, room, door

# ===== インデックス生成 =====
def _scan_and_build_index(root='dataset', out_path='dataset/r3d_index.csv'):
    files = [p for p in glob.glob(os.path.join(root, '**', '*'), recursive=True)
             if os.path.splitext(p)[1].lower() in VALID_EXTS and os.path.isfile(p)]
    if not files:
        raise RuntimeError(f'No files with {VALID_EXTS} under ./{root}')

    # 同一ディレクトリ内でベース名（役割トークン除去後）ごとにグルーピング
    buckets = {}  # key=(dirpath, basekey) -> dict(role->path)
    for p in files:
        d = os.path.dirname(p)
        stem = os.path.splitext(os.path.basename(p))[0]
        name_l = stem.lower()
        role = _role_from_name(name_l)
        # 画像候補の定義：役割トークンが含まれていない or 最もシンプル
        if role != 'image' and _is_image_candidate(name_l):
            role = 'image'
        basekey = _strip_role_tokens(name_l)
        key = (d, basekey)
        if key not in buckets:
            buckets[key] = {}
        # 既に同役割があれば、より“役割トークンに近い方/CSV優先”を採用
        prev = buckets[key].get(role)
        if prev is None:
            buckets[key][role] = p
        else:
            prio = {'.csv': 3, '.png': 2, '.jpg': 1, '.jpeg': 1}
            e_new = os.path.splitext(p)[1].lower()
            e_old = os.path.splitext(prev)[1].lower()
            if prio.get(e_new, 0) >= prio.get(e_old, 0):
                buckets[key][role] = p

    rows, skipped = [], []
    for (d, base), group in buckets.items():
        if {'image','boundary','room','door'}.issubset(group.keys()):
            rows.append({
                'image_csv': group['image'],
                'boundary_csv': group['boundary'],
                'room_csv': group['room'],
                'door_csv': group['door'],
            })
        else:
            skipped.append((d, base, group.keys()))

    if not rows:
        # デバッグ出力（最大10件）
        msg = '[Index build] No pairs found. Examples of incomplete groups:\n'
        for i, (d, base, ks) in enumerate(skipped[:10]):
            msg += f'  - dir={d}, base={base}, roles={list(ks)}\n'
        raise RuntimeError(msg + 'Rename files to include tokens like '
                           '"*_wall", "*_rooms", "*_close" (or boundary/room/door synonyms).')

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    pd.DataFrame(rows).to_csv(out_path, index=False)
    return out_path, len(rows), len(skipped)

def _is_valid_index(df: pd.DataFrame):
    need = {'image_csv','boundary_csv','room_csv','door_csv'}
    return need.issubset(set(c.lower() for c in df.columns)) and len(df) > 0

# ===== データセット =====
class r3dDataset(Dataset):
    """
    - 既に dataset/r3d_index.csv があればそれを使う
    - 無ければ ./dataset を走査し、命名トークンからペアを自動生成
    - .csv / .png / .jpg に対応
    """
    def __init__(self, index_csv='dataset/r3d_index.csv', size=512, transform=None):
        self.size = int(size)
        self.transform = transform

        need_build = True
        if os.path.exists(index_csv):
            try:
                df = pd.read_csv(index_csv)
                if _is_valid_index(df):
                    need_build = False
            except Exception:
                need_build = True

        if need_build:
            print(f'[data.py] building index -> {index_csv}')
            path, n_ok, n_skip = _scan_and_build_index('dataset', index_csv)
            print(f'[data.py] index built: {n_ok} pairs (skipped groups: {n_skip})')

        self.df = pd.read_csv(index_csv)
        # 列名 normalize
        cols = {c.lower(): c for c in self.df.columns}
        self.k_img  = cols['image_csv']
        self.k_wall = cols['boundary_csv']
        self.k_room = cols['room_csv']
        self.k_door = cols['door_csv']

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        p_img  = str(row[self.k_img]).strip()
        p_wall = str(row[self.k_wall]).strip()
        p_room = str(row[self.k_room]).strip()
        p_door = str(row[self.k_door]).strip()

        image   = _imread_any(p_img,  'image',   self.size)
        boundary= _imread_any(p_wall, 'boundary',self.size)
        room    = _imread_any(p_room, 'room',    self.size)
        door    = _imread_any(p_door, 'door',    self.size)

        # ★ 最小パッチ：変換を順に試して適用（Albumentations/torchvision/自作）
        image, boundary, room, door = _apply_transform(self.transform, image, boundary, room, door)
        return image, boundary, room, door
