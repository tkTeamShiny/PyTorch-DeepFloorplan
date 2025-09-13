from importmod import *
import os, random, gc
import numpy as np
import pandas as pd
from skimage.transform import rotate
from torch.utils.data import Dataset, DataLoader

# ---------- CSV読み取り（チャンネル自動判定：1ch or 3ch） ----------
def _read_csv_auto(path: str, size: int):
    exp3 = size * size * 3
    exp1 = size * size
    if not os.path.exists(path):
        raise FileNotFoundError(f'[CSV missing] {path}')
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

# ---------- 行からパス列を安全に取得 ----------
def _pick(row: pd.Series, candidates):
    for c in candidates:
        if c in row and isinstance(row[c], str) and row[c].strip():
            return row[c].strip()
    raise KeyError(f'Missing required columns. Tried: {candidates} in row keys={list(row.index)}')

# ---------- 画像は双一次, マスクは最近傍で回転 ----------
def _rotate_img(x, angle):
    return rotate(x, angle, preserve_range=True, order=1, mode='edge').astype(x.dtype)

def _rotate_mask(x, angle):
    # マスク（整数ラベル）を壊さない
    y = rotate(x, angle, preserve_range=True, order=0, mode='edge')
    return np.rint(y).astype(x.dtype)

class MyRotationTransform:
    def __init__(self, angles=(0, 90, -90, 180)):
        self.angles = angles
    def __call__(self, image, boundary, room, door):
        a = random.choice(self.angles)
        image_r = _rotate_img(image, a)
        boundary_r = _rotate_mask(boundary, a)
        room_r = _rotate_mask(room, a)
        door_r = _rotate_mask(door, a)
        return image_r, boundary_r, room_r, door_r

class r3dDataset(Dataset):
    """
    r3d.csv / r3d2.csv は、以下いずれかの列名で各CSVへのパスを持っている前提：
      - 画像:  ['image_csv','image','img_csv','img']
      - 壁境界: ['boundary_csv','wall_csv','boundary','wall']
      - 部屋  : ['room_csv','rooms_csv','room','rooms']
      - ドア  : ['door_csv','close_csv','door','close']
    パスは相対でも絶対でもOK（存在チェックあり）
    """
    IMG_KEYS  = ['image_csv','image','img_csv','img']
    WALL_KEYS = ['boundary_csv','wall_csv','boundary','wall','cw_csv','cw']
    ROOM_KEYS = ['room_csv','rooms_csv','room','rooms','r_csv','r']
    DOOR_KEYS = ['door_csv','close_csv','door','close','d_csv','d']

    def __init__(self, csv_file='r3d.csv', size=512, transform=None, csv_file2='r3d2.csv'):
        self.size = int(size)
        self.df  = pd.read_csv(csv_file)
        self.df2 = pd.read_csv(csv_file2) if os.path.exists(csv_file2) else pd.DataFrame(columns=self.df.columns)
        self.transform = transform  # 例: MyRotationTransform()
        # 早期に列存在チェック
        if self.df.empty and self.df2.empty:
            raise RuntimeError('Both r3d.csv and r3d2.csv are empty.')
        # optional: 先頭行で列名ヒントを表示
        sample = (self.df if not self.df.empty else self.df2).iloc[0]
        _ = (_pick(sample, self.IMG_KEYS), _pick(sample, self.WALL_KEYS),
             _pick(sample, self.ROOM_KEYS), _pick(sample, self.DOOR_KEYS))

    def __len__(self):
        return len(self.df) + len(self.df2)

    def _row(self, idx):
        if idx < len(self.df):
            return self.df.iloc[idx]
        return self.df2.iloc[idx - len(self.df)]

    def _get_paths(self, idx):
        row = self._row(idx)
        image_csv_path   = _pick(row, self.IMG_KEYS)
        boundary_csv_path= _pick(row, self.WALL_KEYS)
        room_csv_path    = _pick(row, self.ROOM_KEYS)
        door_csv_path    = _pick(row, self.DOOR_KEYS)
        return image_csv_path, boundary_csv_path, room_csv_path, door_csv_path

    def _getset(self, idx):
        image_csv_path, boundary_csv_path, room_csv_path, door_csv_path = self._get_paths(idx)
        image   = _read_csv_auto(image_csv_path,   self.size)
        boundary= _read_csv_auto(boundary_csv_path, self.size)
        room    = _read_csv_auto(room_csv_path,     self.size)
        door    = _read_csv_auto(door_csv_path,     self.size)
        return image, boundary, room, door

    def __getitem__(self, idx):
        image, boundary, room, door = self._getset(idx)
        if self.transform is not None:
            image, boundary, room, door = self.transform(image, boundary, room, door)
        # そのまま numpy を返す（元 main.py が内部でテンソル化する前提）
        return image, boundary, room, door

# ===== 単体テスト =====
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    ds = r3dDataset(csv_file='r3d.csv', csv_file2='r3d2.csv',
                    size=512, transform=MyRotationTransform())
    img, wall, room, door = ds[min(200, len(ds)-1)]
    plt.subplot(2,2,1); plt.title('image'); plt.imshow(img if img.ndim==3 else img, cmap=None)
    plt.subplot(2,2,2); plt.title('boundary'); plt.imshow(wall, cmap='gray')
    plt.subplot(2,2,3); plt.title('room'); plt.imshow(room, cmap='gray')
    plt.subplot(2,2,4); plt.title('door'); plt.imshow(door, cmap='gray')
    plt.tight_layout(); plt.show()
    breakpoint()
    gc.collect()
