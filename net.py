# ============================================
# net.py — PyTorch-DeepFloorplan (改修版: torchvision 新API対応)
# - VGG16 weights: 旧 pretrained 引数を新APIへ自動マッピング
# - それ以外の構造/挙動は元コードを踏襲
# ============================================

# 依存パッケージ（importmod を使う環境でも単体動作できるよう明示）
import gc
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision.models import VGG16_Weights

# 既存環境で使っている共通importがあれば維持（無ければスルーされます）
try:
    from importmod import *  # noqa: F401,F403
except Exception:
    pass


class DFPmodel(torch.nn.Module):
    def __init__(self, pretrained: bool = True, freeze: bool = True):
        super(DFPmodel, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._initializeVGG(pretrained, freeze)

        # === Room Boundary Prediction (壁/境界) ===
        rblist = [512, 256, 128, 64, 32, 3]
        self.rbtrans = nn.ModuleList([
            self._transconv2d(rblist[i], rblist[i + 1], 4, 2, 1)
            for i in range(len(rblist) - 2)
        ])
        self.rbconvs = nn.ModuleList([
            self._conv2d(rblist[i], rblist[i + 1], 3, 1, 1)
            for i in range(len(rblist) - 1)
        ])
        self.rbgrs = nn.ModuleList([
            self._conv2d(rblist[i], rblist[i], 3, 1, 1)
            for i in range(1, len(rblist) - 1)
        ])

        # === Room Type Prediction (部屋カテゴリ) ===
        rtlist = [512, 256, 128, 64, 32]
        self.rttrans = nn.ModuleList([
            self._transconv2d(rtlist[i], rtlist[i + 1], 4, 2, 1)
            for i in range(len(rtlist) - 1)
        ])
        self.rtconvs = nn.ModuleList([
            self._conv2d(rtlist[i], rtlist[i + 1], 3, 1, 1)
            for i in range(len(rtlist) - 1)
        ])
        self.rtgrs = nn.ModuleList([
            self._conv2d(rtlist[i], rtlist[i], 3, 1, 1)
            for i in range(1, len(rtlist))
        ])

        # === Attention Non-local context ===
        clist = [256, 128, 64, 32]
        self.ac1s = nn.ModuleList(self._conv2d(clist[i], clist[i], 3, 1, 1) for i in range(len(clist)))
        self.ac2s = nn.ModuleList(self._conv2d(clist[i], clist[i], 3, 1, 1) for i in range(len(clist)))
        self.ac3s = nn.ModuleList(self._conv2d(clist[i], 1, 1, 1) for i in range(len(clist)))
        self.xc1s = nn.ModuleList(self._conv2d(clist[i], clist[i], 3, 1, 1) for i in range(len(clist)))
        self.xc2s = nn.ModuleList(self._conv2d(clist[i], 1, 1, 1) for i in range(len(clist)))
        self.ecs  = nn.ModuleList(self._conv2d(1, clist[i], 1, 1) for i in range(len(clist)))
        self.rcs  = nn.ModuleList(self._conv2d(2 * clist[i], clist[i], 1, 1) for i in range(len(clist)))

        # === Direction-aware kernel ===
        dak = [9, 17, 33, 65]
        # horizontal
        self.hs = nn.ModuleList(self._dirawareLayer([1, 1, dim, 1]) for dim in dak)
        # vertical
        self.vs = nn.ModuleList(self._dirawareLayer([1, 1, 1, dim]) for dim in dak)
        # diagonal
        self.ds = nn.ModuleList(self._dirawareLayer([1, 1, dim, dim], diag=True) for dim in dak)
        # diagonal flip
        self.dfs = nn.ModuleList(self._dirawareLayer([1, 1, dim, dim], diag=True, flip=True) for dim in dak)

        # 最終層（部屋タイプのロジット出力）
        self.last = self._conv2d(clist[-1], 9, 1, 1)

    # ----- internal builders -----

    def _dirawareLayer(self, shape, diag=False, flip=False, trainable=False):
        w = self.constant_kernel(shape, diag=diag, flip=flip, trainable=trainable)
        pad = ((np.array(shape[2:]) - 1) / 2).astype(int)
        conv = nn.Conv2d(1, 1, shape[2:], 1, list(pad), bias=False)
        conv.weight = w
        return conv

    def _initializeVGG(self, pretrained: bool, freeze: bool):
        """
        旧API: models.vgg16(pretrained=True/False)  -> 非推奨
        新API: models.vgg16(weights=VGG16_Weights.DEFAULT or None)
        """
        weights = VGG16_Weights.DEFAULT if pretrained else None
        encmodel = models.vgg16(weights=weights)

        if freeze:
            # features を凍結（classifier は使わないが念のため）
            for p in encmodel.parameters():
                p.requires_grad = False

        # VGG16 features の前段を使う（元実装に合わせて31層まで）
        features = list(encmodel.features)[:31]
        self.features = nn.ModuleList(features)

    def _conv2d(self, in_, out, kernel, stride=1, padding=0):
        conv2d = nn.Conv2d(in_, out, kernel, stride, padding)
        nn.init.kaiming_uniform_(conv2d.weight)
        nn.init.zeros_(conv2d.bias)
        return conv2d

    def _transconv2d(self, in_, out, kernel, stride=1, padding=0):
        transconv2d = nn.ConvTranspose2d(in_, out, kernel, stride, padding)
        nn.init.kaiming_uniform_(transconv2d.weight)
        nn.init.zeros_(transconv2d.bias)
        return transconv2d

    def constant_kernel(self, shape, value=1, diag=False, flip=False, trainable=False):
        if not diag:
            k = nn.Parameter(torch.ones(shape) * value, requires_grad=trainable)
        else:
            w = torch.eye(shape[2], shape[3])
            if flip:
                w = torch.reshape(w, (1, shape[2], shape[3]))
                w = w.flip(0, 1)
            w = torch.reshape(w, shape)
            k = nn.Parameter(w, requires_grad=trainable)
        return k

    def context_conv2d(self, t, dim=1, size=7, diag=False, flip=False, stride=1, trainable=False):
        # 注意: この関数は self.device に conv を移してから適用
        N, C, H, W = t.size(0), t.size(1), t.size(2), t.size(3)
        size = size if isinstance(size, (tuple, list)) else [size, size]
        stride = stride if isinstance(stride, (tuple, list)) else [1, stride, stride, 1]
        shape = [dim, C, size[0], size[1]]
        w = self.constant_kernel(shape, diag=diag, flip=flip, trainable=trainable)
        pad = ((np.array(shape[2:]) - 1) / 2).astype(int)
        conv = nn.Conv2d(1, 1, shape[2:], 1, list(pad), bias=False)
        conv.weight = w
        conv.to(self.device)
        return conv(t)

    def non_local_context(self, t1, t2, idx, stride=4):
        # t1: RB features, t2: RT features
        N, C, H, W = t1.size(0), t1.size(1), t1.size(2), t1.size(3)
        hs = H // stride if (H // stride) > 1 else (stride - 1)
        vs = W // stride if (W // stride) > 1 else (stride - 1)
        hs = hs if (hs % 2 != 0) else hs + 1
        vs = hs if (vs % 2 != 0) else vs + 1

        a = F.relu(self.ac1s[idx](t1))
        a = F.relu(self.ac2s[idx](a))
        a = torch.sigmoid(self.ac3s[idx](a))

        x = F.relu(self.xc1s[idx](t2))
        x = torch.sigmoid(self.xc2s[idx](x))
        x = a * x

        # direction-aware kernels
        h = self.hs[idx](x)
        v = self.vs[idx](x)
        d1 = self.ds[idx](x)
        d2 = self.dfs[idx](x)

        # double attention
        c1 = a * (h + v + d1 + d2)

        # expand channel
        c1 = self.ecs[idx](c1)

        # concatenation + upsample
        features = torch.cat((t2, c1), dim=1)

        out = F.relu(self.rcs[idx](features))
        return out

    # ----- forward -----

    def forward(self, x):
        # VGG features を通し、所定の層の出力を拾う
        results = []
        for ii, model in enumerate(self.features):
            x = model(x)
            if ii in {4, 9, 16, 23, 30}:
                results.append(x)

        # Room Boundary ブランチ
        rbfeatures = []
        for i, rbtran in enumerate(self.rbtrans):
            x = rbtran(x) + self.rbconvs[i](results[3 - i])
            x = F.relu(self.rbgrs[i](x))
            rbfeatures.append(x)
        # 明示的に size=(512,512) とする
        logits_cw = F.interpolate(self.rbconvs[-1](x), size=(512, 512), mode="bilinear", align_corners=False)

        # Room Type ブランチ
        rtfeatures = []
        x = results[-1]
        for j, rttran in enumerate(self.rttrans):
            x = rttran(x) + self.rtconvs[j](results[3 - j])
            x = F.relu(self.rtgrs[j](x))
            x = self.non_local_context(rbfeatures[j], x, j)

        logits_r = F.interpolate(self.last(x), size=(512, 512), mode="bilinear", align_corners=False)

        return logits_r, logits_cw


if __name__ == "__main__":
    # 簡易チェック（weightsのロード部は手元の環境に合わせて）
    with torch.no_grad():
        testin = torch.randn(1, 3, 512, 512)
        model = DFPmodel(pretrained=True, freeze=True).eval()
        # もし学習済み重みを使う場合はアンコメント:
        # model.load_state_dict(torch.load('weights650.pth', map_location='cpu'))
        logits_r, logits_cw = model.forward(testin)
        print("logits_cw:", tuple(logits_cw.size()), " | logits_r:", tuple(logits_r.size()))
        gc.collect()
