#!/usr/bin/env python3
"""
MuJoCo 시뮬레이션 데이터로 UGV VAE를 재학습하고 공격 CSV를 평가한다.

입력 (sim/generate_datasets.py 산출물, sim/out/):
  normal_train.csv, normal_test.csv, gps_spoof_bias.csv, gps_spoof_gradual.csv, gps_freeze.csv
  — 공통 컬럼: t, 10개 피처(UGV_FEATS 순서), pos_resid_x/y(확장 피처), label(0/1)
옵션:
  --feats 10  (기본) 기존 10개 피처만 사용 — 베이스라인
  --feats 12  10개 + pos_resid_x, pos_resid_y (GPS 위치 − 바퀴 오도메트리 위치 잔차)
출력 (sim/out/, 접미사 _f10 / _f12):
  vae_ugv_sim_f{N}.pth, sim_vae_results_f{N}.json, sim_vae_scores_f{N}.png,
  sim_vae_scores_f{N}.npz (공격별 점수 시계열 — feature_comparison.png 용)

평가 방식은 advanced_attacks.run_experiment 와 동일: 공격 시작 후 WIN-1개 경계 윈도우 제외,
임계값은 정상 검증셋 재구성 오차의 95번째 백분위.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
from advanced_attacks import VAE  # noqa: E402  (동일 구조 재사용)

OUT = HERE / 'out'
UGV_FEATS_10 = ['gps_vel_x', 'gps_vel_y', 'imu_ax', 'imu_ay',
                'residual_x', 'residual_y', 'wheel_vel_l', 'wheel_vel_r',
                'cmd_vel_x', 'cmd_vel_w']
POS_RESID_FEATS = ['pos_resid_x', 'pos_resid_y']
UGV_FEATS = list(UGV_FEATS_10)      # main()에서 --feats 에 따라 교체
WIN = 20
N_FEAT = len(UGV_FEATS)
SEED = 42
ATTACK_FILES = ['gps_spoof_bias.csv', 'gps_spoof_gradual.csv', 'gps_freeze.csv']


def windows(x: np.ndarray) -> np.ndarray:
    """(T, F) -> (T-WIN+1, WIN*F), 스트라이드 1"""
    idx = np.arange(WIN)[None, :] + np.arange(len(x) - WIN + 1)[:, None]
    return x[idx].reshape(len(idx), -1)


def load(name: str) -> pd.DataFrame:
    p = OUT / name
    if not p.exists():
        sys.exit(f'[없음] {p} — 먼저 sim/generate_datasets.py 를 실행하세요')
    return pd.read_csv(p)


def train(x_train: np.ndarray, epochs: int = 60, lr: float = 1e-3, batch: int = 128) -> VAE:
    torch.manual_seed(SEED)
    vae = VAE(input_dim=WIN * N_FEAT)
    opt = torch.optim.Adam(vae.parameters(), lr=lr)
    xt = torch.FloatTensor(x_train)
    for ep in range(epochs):
        perm = torch.randperm(len(xt))
        tot = 0.0
        for i in range(0, len(xt), batch):
            xb = xt[perm[i:i + batch]]
            recon, mu, logvar = vae(xb)
            rec = nn.functional.mse_loss(recon, xb, reduction='sum')
            kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            loss = (rec + 0.1 * kld) / len(xb)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item() * len(xb)
        if ep % 10 == 0 or ep == epochs - 1:
            print(f'  epoch {ep:3d}  loss {tot / len(xt):.4f}')
    vae.eval()
    return vae


def score_series(vae: VAE, data: np.ndarray, mean, std) -> np.ndarray:
    normed = (data - mean) / std
    w = torch.FloatTensor(windows(normed))
    torch.manual_seed(SEED)
    with torch.no_grad():
        recon, _, _ = vae(w)
        s = torch.mean((recon - w) ** 2, dim=1).numpy()
    scores = np.zeros(len(data))
    scores[WIN - 1:] = s
    return scores


def evaluate(vae, df: pd.DataFrame, th, mean, std, name: str) -> dict:
    data = df[UGV_FEATS].to_numpy(np.float32)
    labels = df['label'].to_numpy(int)
    n = len(data)
    atk_start = int(np.argmax(labels == 1)) if labels.any() else n
    scores = score_series(vae, data, mean, std)
    preds = (scores > th).astype(int)

    mask = np.ones(n, bool)
    mask[:WIN - 1] = False                       # 윈도우가 차기 전
    mask[atk_start: atk_start + WIN - 1] = False  # 경계 구간
    yt, yp = labels[mask], preds[mask]
    TP = int(((yp == 1) & (yt == 1)).sum()); FP = int(((yp == 1) & (yt == 0)).sum())
    TN = int(((yp == 0) & (yt == 0)).sum()); FN = int(((yp == 0) & (yt == 1)).sum())
    prec = TP / (TP + FP + 1e-9); rec = TP / (TP + FN + 1e-9)
    f1 = 2 * prec * rec / (prec + rec + 1e-9)
    det = next((i for i in range(atk_start, n) if preds[i]), None)
    delay_ts = (det - atk_start) if det is not None else -1

    # 최초 탐지 윈도우의 피처별 기여도 (어느 센서가 울렸는가)
    top = []
    if det is not None:
        w = ((data[det - WIN + 1: det + 1] - mean) / std).reshape(1, -1)
        with torch.no_grad():
            r, _, _ = vae(torch.FloatTensor(w))
        contrib = ((r.numpy() - w) ** 2).reshape(WIN, N_FEAT).mean(0)
        top = [UGV_FEATS[i] for i in np.argsort(contrib)[::-1][:3]]

    return {'attack': name, 'atk_start_ts': atk_start, 'TP': TP, 'FP': FP, 'TN': TN, 'FN': FN,
            'Precision': round(prec, 4), 'Recall': round(rec, 4), 'F1': round(f1, 4),
            'delay_ts': delay_ts, 'delay_ms': delay_ts * 100 if delay_ts >= 0 else -1,
            'top_feats': top, '_scores': scores, '_labels': labels}


def main():
    global UGV_FEATS, N_FEAT
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--feats', type=int, choices=(10, 12), default=10,
                    help='10 = 기존 10개 피처(베이스라인), 12 = 10개 + pos_resid_x/y')
    args = ap.parse_args()
    UGV_FEATS = UGV_FEATS_10 + (POS_RESID_FEATS if args.feats == 12 else [])
    N_FEAT = len(UGV_FEATS)
    sfx = f'_f{args.feats}'

    print('=' * 60); print(f'시뮬레이션 데이터 기반 UGV VAE 재학습·평가  (피처 {N_FEAT}개)'); print('=' * 60)
    tr = load('normal_train.csv'); te = load('normal_test.csv')
    missing = [f for f in UGV_FEATS if f not in tr.columns]
    if missing:
        sys.exit(f'[없음] CSV에 {missing} 열이 없습니다 — sim/generate_datasets.py 를 다시 실행하세요')
    x_tr = tr[UGV_FEATS].to_numpy(np.float32); x_te = te[UGV_FEATS].to_numpy(np.float32)
    mean = x_tr.mean(0); std = x_tr.std(0) + 1e-8
    print(f'[데이터] train {len(x_tr)} / test {len(x_te)} 샘플, 피처 {N_FEAT}')

    vae = train(windows((x_tr - mean) / std))
    torch.save(vae.state_dict(), OUT / f'vae_ugv_sim{sfx}.pth')

    cal = score_series(vae, x_te, mean, std)[WIN - 1:]
    th = float(np.percentile(cal, 95))
    print(f'[임계값] 정상 검증셋 95p = {th:.4f}  (max={cal.max():.4f})')

    results = [evaluate(vae, load(f), th, mean, std, f.replace('.csv', '')) for f in ATTACK_FILES]

    fig, axes = plt.subplots(len(results), 1, figsize=(10, 2.6 * len(results)), sharex=False)
    for ax, r in zip(np.atleast_1d(axes), results):
        s, lab = r['_scores'], r['_labels']
        t = np.arange(len(s)) * 0.1
        ax.plot(t, s, lw=1.2, color='#2a78d6')
        ax.axhline(th, ls='--', color='#e34948', lw=1, label=f'threshold {th:.3f}')
        a0 = r['atk_start_ts'] * 0.1
        ax.axvspan(a0, t[-1], color='#e34948', alpha=0.08)
        d = f"{r['delay_ms']} ms" if r['delay_ms'] >= 0 else '미탐'
        ax.set_title(f"{r['attack']}  P={r['Precision']:.3f} R={r['Recall']:.3f} F1={r['F1']:.3f} delay={d}", fontsize=10, loc='left')
        ax.set_ylabel('MSE'); ax.set_yscale('log'); ax.legend(loc='upper left', fontsize=8, frameon=False)
    axes[-1].set_xlabel('time (s)')
    fig.tight_layout(); fig.savefig(OUT / f'sim_vae_scores{sfx}.png', dpi=150); plt.close(fig)

    # 점수 시계열 저장 (feature_comparison.png 에서 10/12 피처 비교용)
    np.savez(OUT / f'sim_vae_scores{sfx}.npz', threshold=th,
             **{f"{r['attack']}_scores": r['_scores'] for r in results},
             **{f"{r['attack']}_labels": r['_labels'] for r in results})

    summary = [{k: v for k, v in r.items() if not k.startswith('_')} for r in results]
    with open(OUT / f'sim_vae_results{sfx}.json', 'w', encoding='utf-8') as f:
        json.dump({'n_feats': N_FEAT, 'feats': UGV_FEATS, 'threshold': th,
                   'train_rows': int(len(x_tr)), 'test_rows': int(len(x_te)),
                   'results': summary}, f, ensure_ascii=False, indent=2)

    print(f'\n{"공격":<22}{"P":>7}{"R":>7}{"F1":>7}{"delay":>9}  top feats')
    print('-' * 70)
    for r in summary:
        d = f"{r['delay_ms']}ms" if r['delay_ms'] >= 0 else '미탐'
        print(f"{r['attack']:<22}{r['Precision']:>7.3f}{r['Recall']:>7.3f}{r['F1']:>7.3f}{d:>9}  {', '.join(r['top_feats'])}")
    print(f"\n[저장] {OUT / f'sim_vae_results{sfx}.json'}, sim_vae_scores{sfx}.png, sim_vae_scores{sfx}.npz, vae_ugv_sim{sfx}.pth")


if __name__ == '__main__':
    main()
