#!/usr/bin/env python3
"""
UGV 피처별 기여도 그림(docs/images/ugv_feature_contribution.png) 재생성 스크립트.

배경 — 초기 분석본(docs/images/legacy/ugv_feature_contribution.png)의 문제
  1) 정규화 기준 붕괴: 노트북 `UGV_anomaly_vae.ipynb` 는 정상 데이터를
     generate_normal(1000), 공격 데이터를 generate_*(300) 으로 만들어 섞어 썼습니다.
     generate_normal 이 t=linspace(0,10,n) 의 np.gradient 로 속도를 만들기 때문에
     n 이 달라지면 속도 스케일 자체가 달라지고, 정상(n=1000) 기준 mean/std 로
     공격(n=300)을 정규화하면 공격과 무관한 구간까지 분포를 벗어납니다.
     → 본 스크립트는 정상/공격 시계열 길이를 n=1000 으로 통일합니다.
  2) 극단 OOD 포화: GPS 스푸핑처럼 이상 점수가 임계값의 수십만 배로 튀는 구간에서는
     디코더 출력이 발산해 재구성 오차가 전 피처에 거의 균일하게 퍼집니다. 이때
     "피처별 재구성 오차 최댓값"(anomaly_detector._classify 의 휴리스틱)은 원인을
     식별하지 못합니다. → 본 스크립트는 그 사실을 그림에 드러냅니다.
        · 상단 행: 재구성 오차 기여도. 포화된 패널은 회색 빗금 + 경고 문구로 표시.
        · 하단 행: 같은 윈도우에 대한 SHAP 기여도(KernelExplainer). 포화 구간에서
          원인 센서를 식별하는 것은 이쪽입니다.

데이터/모델 출처
  - 합성 데이터 생성 로직: `UGV_anomaly_vae.ipynb` 의 generate_normal /
    generate_gps_spoofing / generate_wheel_slip / generate_command_anomaly 를
    utils/ugv_data.py 로 옮긴 것(길이 통일·시드 주입만 변경).
  - 재구성 오차 기여도 계산: `anomaly_detector.AnomalyDetector._classify()` 와 동일
    (VAE forward 후 (recon-x)^2 를 (WIN, N_FEAT) 로 접어 시간축 평균).
  - VAE 구조·가중치: anomaly_detector.VAE + vae_ugv.pth (utils/load_model.py).

실행:  python explainability/regen_ugv_feature_contribution.py
       (= make figures)
출력:  docs/images/ugv_feature_contribution.png
       explainability/ugv_feature_contribution_regen.json
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from utils.seed import set_seed, env_info      # noqa: E402
from utils.load_model import load_vae          # noqa: E402
from utils import ugv_data                     # noqa: E402

import matplotlib                              # noqa: E402
matplotlib.use('Agg')
import matplotlib.pyplot as plt                # noqa: E402
import matplotlib.font_manager as fm           # noqa: E402

import shap                                    # noqa: E402

WIN = ugv_data.WIN
N_FEAT = ugv_data.N_FEAT
SEED = 42

C_SURFACE = '#fcfcfb'
C_TEXT = '#0b0b0b'
C_MUTED = '#52514e'
C_OK = '#2a78d6'        # 신뢰 가능한 기여도
C_SHAP = '#1baf7a'      # SHAP 행
C_BAD = '#8a8a86'       # 포화되어 원인 식별 불가
C_WARN = '#e34948'

# 한글 폰트가 있으면 기존 그림과 같은 한글 라벨, 없으면(예: Docker) 영문 라벨
_KOR_FONTS = ['AppleGothic', 'Apple SD Gothic Neo', 'NanumGothic',
              'Malgun Gothic', 'Noto Sans CJK KR', 'Noto Sans KR']
_AVAILABLE = {f.name for f in fm.fontManager.ttflist}
KOR_FONT = next((f for f in _KOR_FONTS if f in _AVAILABLE), None)
if KOR_FONT:
    matplotlib.rcParams['font.family'] = KOR_FONT
matplotlib.rcParams['axes.unicode_minus'] = False

L = {
    'suptitle': ('UGV VAE 이상탐지 XAI — 피처별 기여도 (재생성본)',
                 'UGV VAE anomaly detection XAI — feature attribution (regenerated)'),
    'subtitle': ('정상·공격 시계열 길이 n=1000 통일 / 공격 구간 내 최고 점수 윈도우 1개 기준',
                 'normal & attack series unified to n=1000 / highest-scoring window inside the attack span'),
    'row1': ('재구성 오차 기여도 (기존 휴리스틱)',
             'reconstruction error per feature (existing heuristic)'),
    'row2': ('SHAP 기여도 |value| (KernelExplainer)',
             'SHAP |value| (KernelExplainer)'),
    'saturated': ('[!] 전 피처 균일 포화 — 원인 식별 불가',
                  '[!] uniformly saturated — cause not identifiable'),
    'uniformly': ('완전 균일 시', 'uniform would be'),
    'top1': ('1위 점유율', 'top-1 share'),
    'score': ('이상 점수', 'anomaly score'),
}


def t(key: str) -> str:
    return L[key][0] if KOR_FONT else L[key][1]


# ─────────────────────────────────────────────────────────────
def recon_contribution(vae, x_row: np.ndarray) -> np.ndarray:
    """anomaly_detector._classify() 와 동일한 피처별 재구성 오차."""
    torch.manual_seed(SEED)
    x = torch.from_numpy(x_row.astype(np.float32)).unsqueeze(0)
    with torch.no_grad():
        recon, _, _ = vae(x)
        per_elem = (recon - x) ** 2
        return per_elem.reshape(WIN, N_FEAT).mean(0).numpy()


def make_score_fn(vae):
    """윈도우 → 재구성 오차. SHAP 섭동 평가용이라 z=mu 로 고정(결정적)."""
    def f(X):
        x = torch.from_numpy(np.asarray(X, dtype=np.float32))
        with torch.no_grad():
            mu, _ = vae.encode(x)
            recon = vae.decoder(mu)
            return torch.mean((recon - x) ** 2, dim=1).numpy()
    return f


def saturation_stats(vals: np.ndarray) -> dict:
    """기여도가 전 피처에 균일하게 퍼졌는지(=원인 식별 불가) 정량화."""
    total = float(vals.sum())
    share = float(vals.max() / total) if total > 0 else 0.0
    uniform = 1.0 / len(vals)                      # 완전 균일 시 1위 점유율
    return {
        'top1_share': round(share, 4),
        'uniform_share': round(uniform, 4),
        'max_over_min': round(float(vals.max() / max(vals.min(), 1e-30)), 3),
        'cv': round(float(vals.std() / vals.mean()), 4) if vals.mean() > 0 else 0.0,
        # 1위 점유율이 균일분포의 2배에도 못 미치면 포화로 판정
        'saturated': bool(share < 2 * uniform),
    }


# ─────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=1000, help='시계열 길이 (정상/공격 동일)')
    ap.add_argument('--nsamples', type=int, default=8192,
                    help='KernelExplainer 섭동 샘플 수')
    ap.add_argument('--background', type=int, default=30,
                    help='배경 분포 kmeans 클러스터 수')
    args = ap.parse_args()

    set_seed(SEED)
    vae, cfg = load_vae('ugv', WIN)
    feat_names = cfg['names']

    n = args.n
    normal = ugv_data.generate_normal(n, seed=42)
    mean, std = normal.mean(0), normal.std(0) + 1e-8
    Xn = ugv_data.sliding_windows(normal, mean, std)

    f_det = make_score_fn(vae)
    s_norm = f_det(Xn)
    threshold = float(np.percentile(s_norm, 95))

    rng = np.random.default_rng(SEED)
    bg = shap.kmeans(Xn[rng.choice(len(Xn), 400, replace=False)], args.background)
    explainer = shap.KernelExplainer(f_det, bg)

    regions = ugv_data.attack_regions(n)
    attacks = {
        'GPS Spoofing': ugv_data.generate_gps_spoofing(n),
        'Wheel Slip': ugv_data.generate_wheel_slip(n),
        'Command Anomaly': ugv_data.generate_command_anomaly(n),
    }
    region_key = {'GPS Spoofing': 'GPS_Spoofing',
                  'Wheel Slip': 'Wheel_Slip',
                  'Command Anomaly': 'Command_Anomaly'}

    print('=' * 68)
    print('UGV 피처별 기여도 그림 재생성')
    print(f'  한글 폰트: {KOR_FONT or "없음 → 영문 라벨"}')
    print(f'  정상 윈도우 임계값(95th, 결정적 경로): {threshold:.4f}')
    print('=' * 68)

    payload = {
        'meta': {
            'generated_at': time.strftime('%Y-%m-%dT%H:%M:%S%z'),
            'environment': env_info(),
            'shap_version': shap.__version__,
            'seed': SEED,
            'series_length': n,
            'window': WIN,
            'threshold_p95_deterministic': round(threshold, 4),
            'window_selection': '공격 주입 구간에 완전히 포함된 윈도우 중 이상 점수 최고',
            'recon_contribution_path': 'anomaly_detector._classify() 와 동일 (샘플링 경로, 시드 고정)',
            'shap_path': 'z=mu 결정적 경로, KernelExplainer',
            'nsamples': args.nsamples,
            'background': f'shap.kmeans(normal_windows, {args.background})',
        },
        'attacks': {},
    }

    results = {}
    for name, data in attacks.items():
        lo, hi = regions[region_key[name]]
        X = ugv_data.sliding_windows(data, mean, std)
        scores = f_det(X)
        idx = np.arange(len(X))
        inside = np.where((idx >= lo) & (idx + WIN <= hi))[0]
        pick = int(inside[np.argmax(scores[inside])])

        rc = recon_contribution(vae, X[pick])
        sv = np.asarray(explainer.shap_values(X[pick:pick + 1],
                                              nsamples=args.nsamples,
                                              silent=True, l1_reg=0.0))
        sv = sv.reshape(WIN, N_FEAT)
        shap_abs = np.abs(sv).sum(axis=0)

        rc_stat = saturation_stats(rc)
        sh_stat = saturation_stats(shap_abs)
        rc_order = [feat_names[i] for i in np.argsort(rc)[::-1]]
        sh_order = [feat_names[i] for i in np.argsort(shap_abs)[::-1]]

        results[name] = {
            'window_index': pick, 'score': float(scores[pick]),
            'recon': rc, 'shap': shap_abs,
            'rc_stat': rc_stat, 'sh_stat': sh_stat,
        }
        payload['attacks'][name] = {
            'attack_span_samples': [int(lo), int(hi)],
            'window_index': pick,
            'anomaly_score': round(float(scores[pick]), 4),
            'score_over_threshold': round(float(scores[pick] / threshold), 1),
            'recon_error_by_feature': {k: float(v) for k, v in zip(feat_names, rc)},
            'recon_ranking': rc_order,
            'recon_saturation': rc_stat,
            'shap_abs_by_feature': {k: float(v) for k, v in zip(feat_names, shap_abs)},
            'shap_ranking': sh_order,
            'shap_saturation': sh_stat,
            'top1_agreement': rc_order[0] == sh_order[0],
        }
        print(f'  [{name}] window={pick} score={scores[pick]:.1f} '
              f'({scores[pick]/threshold:.0f}× 임계값)')
        print(f'      재구성오차 1위={rc_order[0]}  1위 점유율={rc_stat["top1_share"]:.3f} '
              f'(균일={rc_stat["uniform_share"]:.3f}) 포화={rc_stat["saturated"]}')
        print(f'      SHAP    1위={sh_order[0]}  1위 점유율={sh_stat["top1_share"]:.3f} '
              f'포화={sh_stat["saturated"]}')

    # ── 플롯 ────────────────────────────────────────────────
    cols = list(results.keys())
    fig, axes = plt.subplots(2, len(cols), figsize=(5.0 * len(cols), 8.4),
                             facecolor=C_SURFACE)

    for c, name in enumerate(cols):
        r = results[name]
        rows = [
            (0, t('row1'), r['recon'], r['rc_stat']),
            (1, t('row2'), r['shap'], r['sh_stat']),
        ]
        for row, xlabel, vals, stat in rows:
            ax = axes[row, c]
            ax.set_facecolor(C_SURFACE)
            saturated = stat['saturated']
            color = C_BAD if saturated else (C_OK if row == 0 else C_SHAP)
            order = np.argsort(vals)[::-1]
            # 지수 오프셋 텍스트가 축 라벨과 겹치지 않도록 스케일을 라벨로 흡수
            exp = int(np.floor(np.log10(vals.max()))) if vals.max() >= 1e4 else 0
            scale = 10.0 ** exp
            unit = f'  ×1e{exp}' if exp else ''
            vals = vals / scale
            ax.barh([feat_names_short(feat_names[i]) for i in order], vals[order],
                    color=color, alpha=0.9,
                    hatch='//' if saturated else None,
                    edgecolor=C_WARN if saturated else 'none',
                    linewidth=0.8 if saturated else 0)
            ax.invert_yaxis()
            ax.tick_params(colors=C_MUTED, labelsize=9)
            for sp in ('top', 'right'):
                ax.spines[sp].set_visible(False)
            for sp in ('left', 'bottom'):
                ax.spines[sp].set_color(C_MUTED)
            ax.grid(axis='x', color='#d8d8d4', lw=0.6)
            ax.set_axisbelow(True)
            share_txt = f'{t("top1")} {stat["top1_share"]*100:.0f}%'
            if saturated:
                share_txt += f' / {t("uniformly")} {stat["uniform_share"]*100:.0f}%'
            ax.set_xlabel(f'{xlabel}{unit}   ({share_txt})',
                          color=C_MUTED, fontsize=9)
            if saturated:
                ax.text(0.5, 0.5, t('saturated'),
                        transform=ax.transAxes, ha='center', va='center',
                        color=C_WARN, fontsize=12, fontweight='bold',
                        bbox=dict(facecolor=C_SURFACE, edgecolor=C_WARN,
                                  boxstyle='round,pad=0.45', alpha=0.95))
            if row == 0:
                ax.set_title(f'{name}\n{t("score")} {r["score"]:,.0f}',
                             color=C_TEXT, fontsize=11, fontweight='bold')

    fig.suptitle(f'{t("suptitle")}\n{t("subtitle")}',
                 color=C_TEXT, fontsize=12.5, fontweight='bold')
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    out_img = REPO_ROOT / 'docs' / 'images' / 'ugv_feature_contribution.png'
    fig.savefig(out_img, dpi=150, facecolor=C_SURFACE)
    plt.close(fig)

    out_json = Path(__file__).resolve().parent / 'ugv_feature_contribution_regen.json'
    with open(out_json, 'w') as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print()
    print(f'[OK] {out_img}')
    print(f'[OK] {out_json}')


def feat_names_short(name: str) -> str:
    return name


if __name__ == '__main__':
    main()
