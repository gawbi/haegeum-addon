#!/usr/bin/env python3
"""v2 VAE 학습 (시스템 python3 + torch). 같은 정상 데이터로 12피처(비교용)와 16피처 모델을 학습해
numpy 가중치(vae_{tag}_np.npz)와 정규화 통계(vae_{tag}_norm.npz)를 저장한다. 구조와 학습 설정은 기존과 동일."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from advanced_attacks import VAE  # noqa: E402

OUT = HERE / 'out'
WIN, SEED = 20, 42
F12 = ['gps_vel_x', 'gps_vel_y', 'imu_ax', 'imu_ay', 'residual_x', 'residual_y', 'wheel_vel_l', 'wheel_vel_r',
       'cmd_vel_x', 'cmd_vel_w', 'pos_resid_x', 'pos_resid_y']
F16 = F12 + ['gyro_z', 'yaw_resid', 'w_resid', 'v_resid']

tr = pd.read_csv(OUT / 'normal_train_v2.csv')
for tag, feats in (('v2_f12', F12), ('v2', F16)):
    x = tr[feats].to_numpy(np.float32)
    mean, std = x.mean(0), x.std(0) + 1e-8
    wins = []
    for _, g in tr.groupby('run'):   # 주행 단위로 윈도우 (주행 경계를 넘지 않음)
        xn = (g[feats].to_numpy(np.float32) - mean) / std
        idx = np.arange(WIN)[None] + np.arange(len(xn) - WIN + 1)[:, None]
        wins.append(xn[idx].reshape(len(idx), -1))
    xt = torch.FloatTensor(np.concatenate(wins))
    torch.manual_seed(SEED)
    vae = VAE(input_dim=WIN * len(feats)); opt = torch.optim.Adam(vae.parameters(), lr=1e-3)
    for ep in range(60):
        perm = torch.randperm(len(xt)); tot = 0.0
        for i in range(0, len(xt), 128):
            xb = xt[perm[i:i + 128]]
            rec, mu, lv = vae(xb)
            loss = (nn.functional.mse_loss(rec, xb, reduction='sum') - 0.05 * torch.sum(1 + lv - mu.pow(2) - lv.exp())) / len(xb)
            opt.zero_grad(); loss.backward(); opt.step(); tot += loss.item() * len(xb)
        if ep % 20 == 0 or ep == 59:
            print(f'{tag} epoch {ep} loss {tot / len(xt):.3f}')
    np.savez(OUT / f'vae_{tag}_np.npz', **{k.replace('.', '_'): v.detach().numpy() for k, v in vae.state_dict().items()})
    np.savez(OUT / f'vae_{tag}_norm.npz', mean=mean, std=std, feats=np.array(feats))
    print('saved', tag, len(xt), 'windows')
