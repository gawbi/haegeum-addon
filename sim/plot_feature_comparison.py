#!/usr/bin/env python3
"""
10개 피처(베이스라인) vs 12개 피처(+pos_resid_x/y) VAE 이상 점수 비교 그림
    입력: out/sim_vae_scores_f10.npz, out/sim_vae_scores_f12.npz, out/sim_vae_results_f{10,12}.json
          (train_eval_sim_vae.py --feats 10 / --feats 12 산출물)
    출력: out/feature_comparison.png
행 = 공격 3종, 열 = 피처 변형(10 / 12). 각 패널: 이상 점수(윈도우 MSE, log) 시계열, 임계값, 공격 시작.
"""
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import font_manager
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
OUT = HERE / 'out'
FONT = '/System/Library/Fonts/Supplemental/AppleGothic.ttf'
ATTACKS = [('gps_spoof_bias', 'GPS 스푸핑 — 고정 편향 (+4 m)'),
           ('gps_spoof_gradual', 'GPS 스푸핑 — 점진 드리프트 (0.25 m/s)'),
           ('gps_freeze', 'GPS 동결 (재생)')]
VARIANTS = [(10, '10개 피처 (기존)', '#2a78d6'), (12, '12개 피처 (+pos_resid)', '#eb6834')]
C_ATTACK = '#e34948'
C_TEXT, C_MUTED, C_GRID = '#0b0b0b', '#52514e', '#e3e2dd'


def setup_font():
    try:
        font_manager.fontManager.addfont(FONT)
        name = font_manager.FontProperties(fname=FONT).get_name()
        plt.rcParams['font.family'] = name
    except Exception as e:  # pragma: no cover
        print(f'[경고] 한글 폰트 로드 실패 ({e}) — 기본 폰트 사용')
    plt.rcParams['axes.unicode_minus'] = False


def main():
    setup_font()
    data = {}
    for n, _, _ in VARIANTS:
        z = np.load(OUT / f'sim_vae_scores_f{n}.npz')
        res = json.load(open(OUT / f'sim_vae_results_f{n}.json', encoding='utf-8'))
        data[n] = (z, {r['attack']: r for r in res['results']})

    Y_MAX = 1.5 * max(float(z[f'{a}_scores'].max()) for z, _ in data.values() for a, _ in ATTACKS)
    fig, axes = plt.subplots(len(ATTACKS), len(VARIANTS), figsize=(12, 8.4), sharex=True)
    for i, (atk, atk_label) in enumerate(ATTACKS):
        for j, (n, var_label, color) in enumerate(VARIANTS):
            ax = axes[i, j]
            z, res = data[n]
            s = z[f'{atk}_scores']; th = float(z['threshold']); r = res[atk]
            t = np.arange(len(s)) * 0.1
            a0 = r['atk_start_ts'] * 0.1
            s_plot = np.where(s > 0, s, np.nan)   # 윈도우가 차기 전(0)은 생략
            ax.axvspan(a0, t[-1], color=C_ATTACK, alpha=0.06, lw=0)
            ax.axvline(a0, color=C_ATTACK, ls='--', lw=1.2)
            ax.axhline(th, color=C_MUTED, ls=':', lw=1.2)
            ax.plot(t, s_plot, color=color, lw=1.6)
            ax.set_yscale('log'); ax.set_ylim(0.03, Y_MAX)
            ax.set_yticks([0.1, 1, 10, 100]); ax.set_yticklabels(['0.1', '1', '10', '100'])   # mathtext(U+2212) 회피
            ax.yaxis.set_minor_formatter(plt.NullFormatter())
            ax.grid(color=C_GRID, lw=0.8); ax.set_axisbelow(True)
            for sp in ('top', 'right'):
                ax.spines[sp].set_visible(False)
            for sp in ('left', 'bottom'):
                ax.spines[sp].set_color(C_GRID)
            ax.tick_params(colors=C_MUTED, labelsize=9)
            d = f"{r['delay_ms'] / 1000:.1f} s" if r['delay_ms'] >= 0 and r['Recall'] > 0 else '미탐'
            ax.set_title(f"{var_label}   R={r['Recall']:.2f}  F1={r['F1']:.2f}  지연={d}",
                         fontsize=10, loc='left', color=C_TEXT)
            if j == 0:
                ax.set_ylabel(f'{atk_label}\n이상 점수 (MSE, log)', fontsize=9, color=C_TEXT)
            ax.text(a0 - 0.4, Y_MAX * 0.7, '공격 시작', color=C_ATTACK, fontsize=8.5, va='top', ha='right')
            ax.text(t[-1], th * 1.2, f'임계값 {th:.3f}', color=C_MUTED, fontsize=8, ha='right', va='bottom')
    for ax in axes[-1]:
        ax.set_xlabel('시간 [s]', fontsize=9, color=C_TEXT)
    fig.suptitle('UGV VAE 이상 점수: 10개 피처 vs 12개 피처(+ GPS - 오도메트리 위치 잔차)   '
                 '— 임계값 = 정상 검증셋 95 백분위, 음영 = 공격 구간', fontsize=11.5, color=C_TEXT, x=0.01, ha='left')
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    fig.savefig(OUT / 'feature_comparison.png', dpi=150, facecolor='white')
    print(f'[저장] {OUT / "feature_comparison.png"}')


if __name__ == '__main__':
    main()
