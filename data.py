# data.py ーーー ドロップイン置換用（自動 index 生成つき）

import os, glob, random
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from skimage.transform import rotate

# ---------- ユーティリティ ----------
def _read_csv_auto(path: str, size: int):
    """1ch/3ch を自動判定して ndarray を返す"""
    exp3 = size * size * 3
    exp1 = size * size
    if not isinstance(path, str):
        raise TypeError(f'[Index error] path cell must be str, got {type(path)}: {path!r}')
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

def _rotate_img(x, angle):
    # 画像は双一次補間
    y = rotate(x, angle, preserve_range=True, order=1, mode='edge')
    return y.astype(x.dtype)

def _rotate_mask(x, angle):
    # マスクは最近傍（整数ラベル破壊しない）
    y = rotate(x, angle, preserve_range=True, order=0, mode='edge')
    return np.rint(y).astype(x.dtype)

def _is_index_df(df: pd.DataFrame):
    """index CSV として妥当か簡易チェック"""
    need_cols = {'image_csv','boundary_csv','room_csv','door_csv'}
    if not need_cols.issubset(set(c.lower() for c in df.columns)):
        return False
    # 先頭行に .csv を含む文字列があるか
    row = df.iloc[0]
    keys = ['image_csv','boundary_csv','room_csv','door_csv']
    for k in keys:
        k0 = next(c for c in df.columns if c.lower()==k)
        v = row[k0]
        if not (isinstance(v, str) and v.strip().endswith('.csv')):
            return False
    return True

def _build_index_csv(root='dataset', out_path='dataset/r3d_index.csv'):
    """
    例：
      ./dataset/newyork/train/21.csv
      ./dataset/newyork/train/21_wall.csv
      ./dataset/newyork/train/21_rooms.csv
      ./dataset/newyork/train/21_close.csv
    をペアにして index CSV を作る
    """
    # ベース画像候補（*_wall/_rooms/_close を除く）
    all_csv = glob.glob(os.path.join(root, '**', '*.csv'), recursive=True)
    def is_base(p):
        b = os.path.basename(p)
        lb = b.lower()
        return (not lb.endswith('_wall.csv')
                and not lb.endswith('_rooms.csv')
                and not lb.endswith('_close.csv'))
    bases = [p for p in all_csv if is_base(p)]
    rows = []
    miss = 0
    for img in sorted(bases):
        base = img[:-4]  # drop .csv
        wall = base + '_wall.csv'
        rooms = base + '_rooms.csv'
        close = base + '_close.csv'
        ok = True
        for q in (wall, rooms, close):
            if not os.path.exists(q):
                ok = False
                break
        if ok:
            rows.append({
                'image_csv': img,
                'boundary_csv': wall,
                'room_csv': rooms,
                'door_csv': close,
            })
        else:
            miss += 1
    if not rows:
        raise RuntimeError(f'[Index build] No pairs found under ./{root}. '
                           f'画像/マスクの命名（*_wall/_rooms/_close）と配置を確認してください。')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    pd.DataFrame(rows).to_csv(out_path, index=False)
    return out_path, len(rows), miss

# ---------- 回転 Transform ----------
class MyRotationTransform:
    def __init__(self, angles=(0, 90, -90, 180)):
        self.angles = angles
    def __call__(self, image, boundary, room, door):
        a = random.choice(self.angles)
        return (_rotate_img(image, a),
                _rotate_mask(boundary, a),
                _rotate_mask(room, a),
                _rotate_mask(door, a))

# ---------- データセット本体 ----------
class r3dDataset(Dataset):
    """
    使い方：
      - 既に dataset/r3d_index.csv があればそのまま使用（列: image_csv,boundary_csv,room_csv,door_csv）
      - 無ければ ./dataset 以下を走査して自動生成（*_wall/_rooms/_close をペアリング）
    """
    def __init__(self, index_csv='dataset/r3d_index.csv', size=512, transform=None):
        self.size = int(size)
        self.transform = transform  # 例: MyRotationTransform()

        # index 読み込み or 自動生成
        need_build = (not os.path.exists(index_csv))
        if not need_build:
            df = pd.read_csv(index_csv)
            if not len(df):
                need_build = True
            elif not _is_index_df(df):
                need_build = True

        if need_build:
            print(f'[data.py] building index -> {index_csv}')
            index_csv, n_ok, n_miss = _build_index_csv(root='dataset', out_path=index_csv)
            print(f'[data.py] index built: {n_ok} pairs (skipped {n_miss})')

        self.df = pd.read_csv(index_csv)
        # 列名 normalize
        cols = {c.lower(): c for c in self.df.columns}
        self.k_img  = cols.get('image_csv')
        self.k_wall = cols.get('boundary_csv')
        self.k_room = cols.get('room_csv')
        self.k_door = cols.get('door_csv')
        for k in [self.k_img, self.k_wall, self.k_room, self.k_door]:
            if k is None:
                raise KeyError(f'[index csv] columns must include image_csv,boundary_csv,room_csv,door_csv; got {self.df.columns.tolist()}')

    def __len__(self):
        return len(self.df)

    def _get_paths(self, idx):
        row = self.df.iloc[idx]
        return (str(row[self.k_img]).strip(),
                str(row[self.k_wall]).strip(),
                str(row[self.k_room]).strip(),
                str(row[self.k_door]).strip())

    def __getitem__(self, idx):
        p_img, p_wall, p_room, p_door = self._get_paths(idx)
        image   = _read_csv_auto(p_img,   self.size)
        boundary= _read_csv_auto(p_wall,  self.size)
        room    = _read_csv_auto(p_room,  self.size)
        door    = _read_csv_auto(p_door,  self.size)
        if self.transform is not None:
            image, boundary, room, door = self.transform(image, boundary, room, door)
        return image, boundary, room, door
