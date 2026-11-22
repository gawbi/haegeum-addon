#!/usr/bin/env python3
"""학습된 시뮬레이션 VAE(.pth)를 numpy 가중치(.npz)로 내보낸다 (시스템 python3 + torch 로 실행).
sim/.venv 에는 torch 가 없으므로, 폐루프 보정 실험(recovery_eval.py)은 이 npz 로 numpy 추론을 한다."""
from pathlib import Path
import numpy as np
import torch

OUT = Path(__file__).resolve().parent / 'out'
for f in (10, 12):
    sd = torch.load(OUT / f'vae_ugv_sim_f{f}.pth', map_location='cpu')
    np.savez(OUT / f'vae_ugv_sim_f{f}_np.npz', **{k.replace('.', '_'): v.numpy() for k, v in sd.items()})
    print(f'f{f}:', list(sd.keys()))
