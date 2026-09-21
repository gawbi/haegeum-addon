#!/usr/bin/env python3
"""
VAE 이상탐지 설명가능성 분석 — SHAP (KernelExplainer)
(DAH 2026 해금팀 / haegeum-addon 추가 연구)

무엇을 설명하는가
  이상탐지 판단의 근거는 "재구성 오차(anomaly score)"입니다. 따라서 설명 대상 함수는
  모델의 출력이 아니라 **윈도우 → 재구성 오차(스칼라)** 함수 f(x) 입니다.
  SHAP 은 이 f 에 대해 윈도우의 각 입력값(20 타임스텝 × N 피처)이 오차를 얼마나
  끌어올렸는지 분해합니다. 타임스텝 축으로 합산하면 "어느 센서 때문에 경보가
  울렸는가"가 나옵니다.

결정성(determinism)
  anomaly_detector 의 추론 경로는 VAE 재파라미터화 샘플링(z = mu + std*eps)을 쓰기
  때문에 같은 입력에도 점수가 미세하게 흔들립니다. SHAP 은 수만 번의 섭동 평가를
  하므로 이 잡음이 기여도 추정을 오염시킵니다. 그래서 설명용 f 는 z = mu (잠재
  평균)로 고정한 결정적 경로를 씁니다. 두 경로의 점수 상관계수를 함께 측정해
  결과(JSON/README)에 기록합니다.

실행:  python explainability/shap_analysis.py
출력:  explainability/shap_results.json
       docs/images/shap_summary_{uav,ugv}.png
       docs/images/shap_attack_contribution_{uav,ugv}.png
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

import shap                                    # noqa: E402

WIN = 20
SEED = 42

C_SURFACE = '#fcfcfb'
C_TEXT = '#0b0b0b'
C_MUTED = '#52514e'
ATTACK_COLORS = ['#2a78d6', '#eb6834', '#1baf7a', '#4a3aa7']


# ─────────────────────────────────────────────────────────────
# 설명 대상 함수
# ─────────────────────────────────────────────────────────────
def make_score_fn(vae, deterministic: bool = True):
    """윈도우(평탄화) → 재구성 오차. deterministic=True 면 z=mu 고정."""
    def f(X):
        x = torch.from_numpy(np.asarray(X, dtype=np.float32))
        with torch.no_grad():
            if deterministic:
                mu, _ = vae.encode(x)
                recon = vae.decoder(mu)
            else:
                recon, _, _ = vae(x)
            return torch.mean((recon - x) ** 2, dim=1).numpy()
    return f


def legacy_contribution(vae, x_row: np.ndarray, n_feat: int) -> np.ndarray:
    """
    기존 방식(anomaly_detector._classify / 노트북 XAI)과 동일한
    '피처별 재구성 오차' 기여도. 비교 기준선으로 사용.
    """
    torch.manual_seed(SEED)
    x = torch.from_numpy(x_row.astype(np.float32)).unsqueeze(0)
    with torch.no_grad():
        recon, _, _ = vae(x)
        per_elem = (recon - x) ** 2
        return per_elem.reshape(WIN, n_feat).mean(0).numpy()


# ─────────────────────────────────────────────────────────────
# 데이터셋
# ─────────────────────────────────────────────────────────────
def build_uav_dataset():
    import advanced_attacks as AA
    normal = AA.gen_normal(1000, seed=42)
    mean, std = normal.mean(0), normal.std(0) + 1e-8
    n_total, start = 300, 100
    base = AA.gen_normal(n_total, seed=123)
    attacks = {
        'GPS_Spoofing': (AA.atk_gps_sudden(base, start), (start, n_total)),
        'Altitude_Spoofing': (AA.atk_altitude_sudden(base, start), (start, n_total)),
        'Command_Injection': (AA.atk_cmd_injection(base, start), (start, n_total)),
    }
    return normal, mean, std, attacks, AA.UAV_FEATS, 8


def build_ugv_dataset():
    n = 1000
    normal = ugv_data.generate_normal(n, seed=42)
    mean, std = normal.mean(0), normal.std(0) + 1e-8
    regions = ugv_data.attack_regions(n)
    attacks = {
        'GPS_Spoofing': (ugv_data.generate_gps_spoofing(n), regions['GPS_Spoofing']),
        'Wheel_Slip': (ugv_data.generate_wheel_slip(n), regions['Wheel_Slip']),
        'Command_Anomaly': (ugv_data.generate_command_anomaly(n), regions['Command_Anomaly']),
    }
    return normal, mean, std, attacks, ugv_data.FEATURE_NAMES, 10


def windows(data, mean, std, n_feat):
    normed = (data - mean) / std
    idx = np.arange(len(normed) - WIN + 1)
    out = np.stack([normed[i:i + WIN] for i in idx])
    return out.reshape(len(idx), WIN * n_feat).astype(np.float32)


# ─────────────────────────────────────────────────────────────
def analyze(platform: str, n_explain: int, nsamples: int, n_background: int) -> dict:
    set_seed(SEED)
    vae, cfg = load_vae(platform, WIN)
    n_feat = cfg['n_feat']
    feat_names = cfg['names']

    if platform == 'uav':
        normal, mean, std, attacks, _, n_feat_chk = build_uav_dataset()
    else:
        normal, mean, std, attacks, _, n_feat_chk = build_ugv_dataset()
    assert n_feat == n_feat_chk

    f_det = make_score_fn(vae, deterministic=True)
    f_sto = make_score_fn(vae, deterministic=False)

    Xn = windows(normal, mean, std, n_feat)
    s_norm = f_det(Xn)
    threshold = float(np.percentile(s_norm, 95))

    # 결정적 경로 vs 실제 추론 경로(샘플링) 점수 상관
    torch.manual_seed(SEED)
    s_sto = f_sto(Xn)
    det_vs_sto_r = float(np.corrcoef(s_norm, s_sto)[0, 1])

    # 배경 분포: 정상 윈도우 kmeans 요약
    bg = shap.kmeans(Xn[np.random.default_rng(SEED).choice(len(Xn), 400, replace=False)],
                     n_background)
    explainer = shap.KernelExplainer(f_det, bg)

    conditions = {}
    # (1) 정상 기준선 — 점수 중앙값 부근 윈도우
    med_idx = np.argsort(np.abs(s_norm - np.median(s_norm)))[:n_explain]
    conditions['Normal'] = Xn[med_idx]

    # (2) 공격별 — 공격 구간 안에서 점수 상위 윈도우
    for name, (data, (lo, hi)) in attacks.items():
        X = windows(data, mean, std, n_feat)
        s = f_det(X)
        idx = np.arange(len(X))
        inside = np.where((idx >= lo) & (idx + WIN <= hi))[0]
        if len(inside) == 0:
            inside = np.where(idx >= lo)[0]
        top = inside[np.argsort(s[inside])[::-1][:n_explain]]
        conditions[name] = X[top]

    result = {
        'platform': platform,
        'feature_names': feat_names,
        'threshold_p95_deterministic': round(threshold, 4),
        'deterministic_vs_sampled_score_pearson_r': round(det_vs_sto_r, 4),
        'shap': {
            'explainer': 'shap.KernelExplainer',
            'background': f'shap.kmeans(normal_windows, {n_background})',
            'nsamples': nsamples,
            'n_explained_windows_per_condition': n_explain,
            'expected_value': round(float(explainer.expected_value), 4),
        },
        'conditions': {},
        'agreement_with_reconstruction_error': {},
    }

    agg_shap_all, agg_val_all, cond_labels = [], [], []

    for cond, X in conditions.items():
        t0 = time.time()
        sv = explainer.shap_values(X, nsamples=nsamples, silent=True, l1_reg=0.0)
        sv = np.asarray(sv).reshape(len(X), WIN, n_feat)
        per_feat_signed = sv.sum(axis=1)                  # (k, n_feat)
        per_feat_abs = np.abs(sv).sum(axis=1)

        legacy = np.stack([legacy_contribution(vae, row, n_feat) for row in X])

        mean_shap = per_feat_signed.mean(0)
        mean_abs = per_feat_abs.mean(0)
        mean_legacy = legacy.mean(0)

        from scipy.stats import spearmanr
        rho = float(spearmanr(mean_abs, mean_legacy).statistic)
        top3_shap = [feat_names[i] for i in np.argsort(mean_abs)[::-1][:3]]
        top3_legacy = [feat_names[i] for i in np.argsort(mean_legacy)[::-1][:3]]

        result['conditions'][cond] = {
            'mean_score': round(float(f_det(X).mean()), 4),
            'shap_signed_by_feature': {n: round(float(v), 6)
                                       for n, v in zip(feat_names, mean_shap)},
            'shap_abs_by_feature': {n: round(float(v), 6)
                                    for n, v in zip(feat_names, mean_abs)},
            'recon_error_by_feature': {n: round(float(v), 6)
                                       for n, v in zip(feat_names, mean_legacy)},
            'elapsed_sec': round(time.time() - t0, 1),
        }
        result['agreement_with_reconstruction_error'][cond] = {
            'spearman_rho': round(rho, 4),
            'top3_shap': top3_shap,
            'top3_recon_error': top3_legacy,
            'top1_match': top3_shap[0] == top3_legacy[0],
            'top3_set_overlap': len(set(top3_shap) & set(top3_legacy)),
        }

        # summary plot 용 누적 (윈도우별 피처 평균값을 feature value 로)
        agg_shap_all.append(per_feat_signed)
        agg_val_all.append(X.reshape(len(X), WIN, n_feat).mean(axis=1))
        cond_labels += [cond] * len(X)

        print(f'  [{platform}/{cond}] rho={rho:.3f} '
              f'top1 SHAP={top3_shap[0]} / recon={top3_legacy[0]} '
              f'({result["conditions"][cond]["elapsed_sec"]}s)')

    result['_agg_shap'] = np.concatenate(agg_shap_all)
    result['_agg_val'] = np.concatenate(agg_val_all)
    result['_cond_labels'] = cond_labels
    return result


# ─────────────────────────────────────────────────────────────
# 플롯
# ─────────────────────────────────────────────────────────────
def plot_summary(res: dict, out_path: Path):
    feat_names = res['feature_names']
    plt.figure(figsize=(8, 4.6), facecolor=C_SURFACE)
    shap.summary_plot(res['_agg_shap'], features=res['_agg_val'],
                      feature_names=feat_names, show=False, plot_size=None)
    fig = plt.gcf()
    fig.set_facecolor(C_SURFACE)
    ax = plt.gca()
    ax.set_facecolor(C_SURFACE)
    ax.set_xlabel('SHAP value (contribution to reconstruction error)',
                  color=C_MUTED)
    ax.tick_params(colors=C_MUTED)
    plt.title(f'{res["platform"].upper()} — SHAP summary '
              f'(normal + attack windows, time-axis aggregated)',
              color=C_TEXT, fontsize=11)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, facecolor=C_SURFACE)
    plt.close('all')


def plot_attack_contributions(res: dict, out_path: Path):
    """공격 유형별 SHAP 기여도 (기존 재구성 오차 기여도와 나란히 비교)."""
    feat_names = res['feature_names']
    conds = list(res['conditions'].keys())
    fig, axes = plt.subplots(2, len(conds), figsize=(4.1 * len(conds), 7.6),
                             facecolor=C_SURFACE)

    for col, cond in enumerate(conds):
        color = ATTACK_COLORS[col % len(ATTACK_COLORS)]
        c = res['conditions'][cond]
        rows = [
            ('SHAP |value| (time-aggregated)',
             np.array([c['shap_abs_by_feature'][n] for n in feat_names])),
            ('reconstruction error per feature',
             np.array([c['recon_error_by_feature'][n] for n in feat_names])),
        ]
        for row, (title, vals) in enumerate(rows):
            ax = axes[row, col]
            ax.set_facecolor(C_SURFACE)
            order = np.argsort(vals)[::-1]
            ax.barh([feat_names[i] for i in order], vals[order],
                    color=color, alpha=0.9 if row == 0 else 0.45)
            ax.invert_yaxis()
            ax.tick_params(colors=C_MUTED, labelsize=8.5)
            for spine in ('top', 'right'):
                ax.spines[spine].set_visible(False)
            for spine in ('left', 'bottom'):
                ax.spines[spine].set_color(C_MUTED)
            ax.grid(axis='x', color='#d8d8d4', lw=0.6)
            ax.set_axisbelow(True)
            if row == 0:
                agree = res['agreement_with_reconstruction_error'][cond]
                ax.set_title(f'{cond}\nSpearman ρ(SHAP, recon) = '
                             f'{agree["spearman_rho"]:.2f}',
                             color=C_TEXT, fontsize=10)
            ax.set_xlabel(title, color=C_MUTED, fontsize=8.5)

    fig.suptitle(f'{res["platform"].upper()} — feature attribution by condition '
                 f'(top: SHAP on reconstruction error, bottom: existing per-feature '
                 f'reconstruction error)',
                 color=C_TEXT, fontsize=12, fontweight='bold')
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out_path, dpi=150, facecolor=C_SURFACE)
    plt.close(fig)


# ─────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n-explain', type=int, default=5,
                    help='조건별 설명할 윈도우 수')
    ap.add_argument('--nsamples', type=int, default=1024,
                    help='KernelExplainer 섭동 샘플 수')
    ap.add_argument('--background', type=int, default=20,
                    help='배경 분포 kmeans 클러스터 수')
    args = ap.parse_args()

    out_dir = Path(__file__).resolve().parent
    img_dir = REPO_ROOT / 'docs' / 'images'
    img_dir.mkdir(parents=True, exist_ok=True)

    print('=' * 68)
    print('VAE 이상탐지 SHAP 설명가능성 분석')
    print('=' * 68)

    payload = {
        'meta': {
            'generated_at': time.strftime('%Y-%m-%dT%H:%M:%S%z'),
            'environment': env_info(),
            'shap_version': shap.__version__,
            'seed': SEED,
            'explained_function': 'window -> VAE reconstruction error (z = mu, 결정적)',
        },
        'platforms': {},
    }

    for platform in ('uav', 'ugv'):
        res = analyze(platform, args.n_explain, args.nsamples, args.background)
        plot_summary(res, img_dir / f'shap_summary_{platform}.png')
        plot_attack_contributions(
            res, img_dir / f'shap_attack_contribution_{platform}.png')
        for k in ('_agg_shap', '_agg_val', '_cond_labels'):
            res.pop(k)
        payload['platforms'][platform] = res

    with open(out_dir / 'shap_results.json', 'w') as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print()
    print(f'[OK] {out_dir / "shap_results.json"}')
    for platform in ('uav', 'ugv'):
        print(f'[OK] {img_dir / f"shap_summary_{platform}.png"}')
        print(f'[OK] {img_dir / f"shap_attack_contribution_{platform}.png"}')


if __name__ == '__main__':
    main()
