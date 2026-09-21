#!/usr/bin/env python3
"""
이상탐지 임계값 선택 근거 산출 — ROC/AUC 기반 재검토
(DAH 2026 해금팀 / haegeum-addon 추가 연구)

왜 필요한가
  현재 운용 임계값은 `anomaly_detector.py` 의 `threshold_percentile=95.0`,
  즉 **정상 점수의 95th percentile** 입니다. 이 값은 "정상 데이터의 5 %를 버린다"는
  규칙일 뿐, 공격 점수 분포를 한 번도 보지 않고 정해진 값입니다. 즉 사실상
  **목표 FPR 5 % 고정 규칙**이며, ROC 상에서 최적이라는 근거가 없습니다.
  이 스크립트는 정상/공격 점수 분포로 ROC·AUC 를 산출하고, 임계값 선택 기준
  6가지를 같은 평가셋에서 비교해 그 근거를 수치로 남깁니다.

무엇을 측정하는가
  1) 명목 평가셋(기존 실험과 동일한 공격 강도)의 ROC·AUC 와 분리 여유(margin)
  2) **탐지 한계 근방(marginal) 평가셋** — 주입 크기를 정상 피처 표준편차의
     m σ 로 줄여 가며 AUC 가 무너지기 시작하는 구간을 찾고, 그 구간에서
     기준별 임계값의 Precision / Recall / F1 / FPR 을 비교
       - current_normal_p95 : 현재 구현 (정상 95th percentile)
       - youden_j           : max(TPR - FPR)
       - target_fpr_1pct    : FPR ≤ 1 % 중 TPR 최대
       - target_fpr_5pct    : FPR ≤ 5 % 중 TPR 최대
       - max_f1             : F1 최대
       - min_cost_fn10      : 10·FN + 1·FP 최소 (미탐 비용 10배 가정)
  3) 미탐(FN)/오탐(FP) 비용 비대칭 민감도 — 비용비 스윕 + 오경보 부하 환산
  4) 경계 윈도우 오염 샘플 수(k)별 탐지 여부 → 탐지 지연의 하한
  5) README「실험 결과」1) 의 UAV Recall 0.714 원인 분해
     (라벨링 아티팩트 / 경계 윈도우 / 임계값 보수성 중 무엇인가)

점수 경로
  `anomaly_detector._infer()` 와 동일한 확률적 경로(z = mu + std·eps)를 씁니다.
  재파라미터화 샘플링 때문에 실행마다 점수가 미세하게 흔들리므로
  `utils/seed.set_seed(42)` + 스코어링 직전 `torch.manual_seed(42)` 로 고정합니다.
  판정 규칙도 `score > threshold` (strict) 로 동일합니다.

실행:  python analysis/threshold_selection.py   (= make threshold)
출력:  analysis/threshold_results.json
       docs/images/threshold_roc.png
       docs/images/threshold_tradeoff.png
       docs/images/threshold_sensitivity.png
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

WIN = 20
SEED = 42

# 센서 발행 주기 / 경보 쿨다운 (sensor_publisher.py, anomaly_detector.py 기본값)
SENSOR_RATE_HZ = 10.0
COOLDOWN_SEC = 2.0

# 주입 크기 스윕 (정상 피처 표준편차 배수)
SIGMA_LEVELS = (0.05, 0.1, 0.2, 0.5, 1.0, 2.0)

# "탐지 한계 근방" 판정 구간: 개별 AUC 가 이 범위인 강도만 모아 임계값 기준을 비교한다.
# 상한(0.999) 위는 어떤 임계값을 골라도 완벽 분리라 기준 간 차이가 없고,
# 하한(0.60) 아래는 점수에 신호가 거의 없어 임계값으로 살릴 수 없는 영역이다.
MARGINAL_AUC_RANGE = (0.60, 0.999)

C_SURFACE = '#fcfcfb'
C_TEXT = '#0b0b0b'
C_MUTED = '#52514e'
C_NORMAL = '#2a78d6'
C_ATTACK = '#eb6834'
C_WEAK = '#4a3aa7'
C_OK = '#1baf7a'

CRITERION_COLORS = {
    'current_normal_p95': '#e34948',
    'youden_j': '#1baf7a',
    'target_fpr_1pct': '#4a3aa7',
    'target_fpr_5pct': '#2a78d6',
    'max_f1': '#eb6834',
    'min_cost_fn10': '#0b0b0b',
}


# ─────────────────────────────────────────────────────────────
# 평가 지표 (외부 의존 없이 직접 구현 — tests/test_threshold_metrics.py 에서 검증)
# ─────────────────────────────────────────────────────────────
def confusion_at(y_true: np.ndarray, scores: np.ndarray, thr: float) -> dict:
    """판정 규칙은 anomaly_detector 와 동일한 strict `score > threshold`."""
    pred = np.asarray(scores) > thr
    y = np.asarray(y_true).astype(bool)
    return {'TP': int((pred & y).sum()),
            'FP': int((pred & ~y).sum()),
            'TN': int((~pred & ~y).sum()),
            'FN': int((~pred & y).sum())}


def metrics_at(y_true: np.ndarray, scores: np.ndarray, thr: float) -> dict:
    c = confusion_at(y_true, scores, thr)
    TP, FP, TN, FN = c['TP'], c['FP'], c['TN'], c['FN']
    prec = TP / (TP + FP) if (TP + FP) else 0.0
    rec = TP / (TP + FN) if (TP + FN) else 0.0
    fpr = FP / (FP + TN) if (FP + TN) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    c.update({'threshold': round(float(thr), 4),
              'precision': round(prec, 4), 'recall': round(rec, 4),
              'f1': round(f1, 4), 'fpr': round(fpr, 6)})
    return c


def roc_curve_strict(y_true: np.ndarray, scores: np.ndarray):
    """
    strict `score > thr` 판정에 맞춘 ROC.
    후보 임계값 = 관측 점수의 내림차순 유일값 (+ ±inf 양 끝점).
    Returns: (fpr, tpr, thresholds) — fpr 오름차순, thresholds 내림차순.
    """
    y = np.asarray(y_true).astype(bool)
    s = np.asarray(scores, dtype=np.float64)
    P, N = int(y.sum()), int((~y).sum())
    if P == 0 or N == 0:
        raise ValueError('ROC 를 그리려면 양성/음성 표본이 모두 필요합니다')

    order = np.argsort(-s, kind='mergesort')
    s_sorted, y_sorted = s[order], y[order]
    tp_cum = np.cumsum(y_sorted)
    fp_cum = np.cumsum(~y_sorted)

    # 후보 임계값 v 에서 strict `score > v` 로 잡히는 개수 =
    # 내림차순 정렬에서 v 동점 그룹이 시작되기 직전까지의 누적 개수.
    starts = np.where(np.r_[True, np.diff(s_sorted) != 0])[0]
    tp_gt = np.where(starts > 0, tp_cum[starts - 1], 0)
    fp_gt = np.where(starts > 0, fp_cum[starts - 1], 0)

    thr = np.r_[np.inf, s_sorted[starts], -np.inf]
    tpr = np.r_[0.0, tp_gt / P, 1.0]
    fpr = np.r_[0.0, fp_gt / N, 1.0]
    return fpr, tpr, thr


def auc_trapezoid(x: np.ndarray, y: np.ndarray) -> float:
    trapz = getattr(np, 'trapezoid', None) or np.trapz  # noqa: NPY201
    return float(trapz(y, x))


def roc_auc(y_true: np.ndarray, scores: np.ndarray) -> float:
    fpr, tpr, _ = roc_curve_strict(y_true, scores)
    return auc_trapezoid(fpr, tpr)


# ─────────────────────────────────────────────────────────────
# 임계값 선택 기준
# ─────────────────────────────────────────────────────────────
def pick_youden(fpr, tpr, thr) -> float:
    return float(thr[int(np.argmax(tpr - fpr))])


def pick_target_fpr(fpr, tpr, thr, target: float) -> float:
    """FPR ≤ target 을 만족하는 지점 중 TPR 최대인 임계값."""
    ok = np.where(fpr <= target + 1e-12)[0]
    return float(thr[ok[int(np.argmax(tpr[ok]))]])


def pick_max_f1(y_true, scores, thr_candidates) -> float:
    best_thr, best_f1 = float(thr_candidates[0]), -1.0
    for t in thr_candidates:
        f1 = metrics_at(y_true, scores, t)['f1']
        if f1 > best_f1:
            best_f1, best_thr = f1, float(t)
    return best_thr


def pick_min_cost(y_true, scores, thr_candidates, cost_fn: float,
                  cost_fp: float = 1.0) -> float:
    best_thr, best_cost = float(thr_candidates[0]), float('inf')
    for t in thr_candidates:
        c = confusion_at(y_true, scores, t)
        cost = cost_fn * c['FN'] + cost_fp * c['FP']
        if cost < best_cost:
            best_cost, best_thr = cost, float(t)
    return best_thr


def alarm_load(fpr: float) -> dict:
    """FPR → 운용자 체감 오경보 부하 (10 Hz 센서, 2 s 경보 쿨다운 기준)."""
    raw = fpr * SENSOR_RATE_HZ * 3600.0
    capped = min(raw, 3600.0 / COOLDOWN_SEC)
    return {'false_alarms_per_hour_raw': round(raw, 1),
            'false_alarms_per_hour_with_cooldown': round(capped, 1),
            'mean_minutes_between_false_alarms':
                round(60.0 / capped, 2) if capped > 0 else None}


# ─────────────────────────────────────────────────────────────
# 점수 산출
# ─────────────────────────────────────────────────────────────
def score_windows(vae, X: np.ndarray, seed: int = SEED) -> np.ndarray:
    """윈도우 배열 → 재구성 오차(MSE). anomaly_detector._infer 와 동일 경로."""
    torch.manual_seed(seed)
    out = np.empty(len(X), dtype=np.float64)
    with torch.no_grad():
        for i, row in enumerate(X):
            x = torch.from_numpy(np.ascontiguousarray(row)).unsqueeze(0)
            recon, _, _ = vae(x)
            out[i] = torch.mean((recon - x) ** 2).item()
    return out


def make_windows(data: np.ndarray, mean, std) -> np.ndarray:
    normed = (data - mean) / std
    idx = np.arange(len(normed) - WIN + 1)
    out = np.stack([normed[i:i + WIN] for i in idx])
    return out.reshape(len(idx), -1).astype(np.float32)


def inject_sigma(data: np.ndarray, start: int, cols, m: float,
                 std: np.ndarray, residual_cols=None) -> np.ndarray:
    """
    정상 피처 표준편차의 m 배만큼을 지정 컬럼에 상수 편향으로 주입.
    기존 공격 함수(advanced_attacks.atk_*, ugv_data.generate_*)와 주입 대상 컬럼은
    같고 크기만 σ 단위로 바꾼 것 — 탐지 한계(강도 대비 AUC)를 재기 위한 용도.
    """
    d = data.copy()
    for c in cols:
        d[start:, c] += m * std[c]
    if residual_cols:
        rx, ry = residual_cols
        d[:, rx] = d[:, 0] - d[:, 2]
        d[:, ry] = d[:, 1] - d[:, 3]
    return d


# ─────────────────────────────────────────────────────────────
# 데이터셋 구성
# ─────────────────────────────────────────────────────────────
def build_uav(vae) -> dict:
    """
    UAV: advanced_attacks.py 의 생성기·주입 함수를 그대로 재사용.
      보정셋 : gen_normal(1000, seed=42)   — 임계값 산출용 (기존 구현과 동일)
      음성셋 : gen_normal(1000, seed=123)  — 보정에 쓰지 않은 정상 (held-out)
      양성셋 : 공격 9종 × '공격 구간으로 완전히 채워진' 윈도우 181개
              (경계 윈도우는 라벨이 모호하므로 양성/음성 어디에도 넣지 않고 별도 분석)
    """
    import advanced_attacks as AA

    calib = AA.gen_normal(1000, seed=SEED)
    mean, std = calib.mean(0), calib.std(0) + 1e-8
    s_calib = score_windows(vae, make_windows(calib, mean, std))
    s_neg = score_windows(vae, make_windows(AA.gen_normal(1000, seed=123), mean, std))

    n_total, start = 300, 100
    base = AA.gen_normal(n_total, seed=123)
    attacks = {
        'GPS_Spoofing': AA.atk_gps_sudden(base, start),
        'Altitude_Spoofing': AA.atk_altitude_sudden(base, start),
        'Command_Injection': AA.atk_cmd_injection(base, start),
        'Composite_GPS+CmdInj': AA.atk_composite_gps_cmd(base, start),
        'Composite_GPS+Alt': AA.atk_composite_gps_alt(base, start),
        'Gradual_GPS_rate0.3': AA.atk_gradual_gps(base, start, rate=0.3),
        'Gradual_GPS_rate0.1': AA.atk_gradual_gps(base, start, rate=0.1),
        'Gradual_Alt_rate0.5': AA.atk_gradual_alt(base, start, rate=0.5),
        'Gradual_Alt_rate0.2': AA.atk_gradual_alt(base, start, rate=0.2),
    }

    pos, series = {}, {}
    for name, data in attacks.items():
        X = make_windows(data, mean, std)
        s = score_windows(vae, X)
        j = np.arange(len(X))
        series[name] = {'scores': s,
                        'full_normal': j + WIN <= start,
                        'boundary': (j > start - WIN) & (j < start),
                        'full_attack': j >= start,
                        'contamination': np.clip(j + WIN - start, 0, WIN)}
        pos[name] = s[j >= start]

    # 강도 스윕 (σ 단위)
    sweep = {}
    for family, cols, res in (('GPS', (0, 1), (4, 5)),
                              ('Altitude', (6,), None)):
        for m in SIGMA_LEVELS:
            d = inject_sigma(base, start, cols, m, std, res)
            X = make_windows(d, mean, std)
            s = score_windows(vae, X)
            sweep[(family, m)] = s[np.arange(len(X)) >= start]

    return {'platform': 'uav',
            'calib_scores': s_calib, 'neg_scores': s_neg,
            'pos_by_attack': pos, 'series': series, 'sweep': sweep,
            'basic_attacks': ['GPS_Spoofing', 'Altitude_Spoofing',
                              'Command_Injection'],
            'calib_desc': 'advanced_attacks.gen_normal(1000, seed=42) 전체 윈도우 981개',
            'neg_desc': 'advanced_attacks.gen_normal(1000, seed=123) — 보정에 미사용',
            'pos_desc': ('공격 9종: gen_normal(300, seed=123) 에 t=100 부터 주입, '
                         '윈도우가 공격 구간으로 완전히 채워진 181개만 양성'),
            'sweep_desc': ('GPS(gps_vel_x/y + residual 재계산) / Altitude(baro_alt) 에 '
                           '정상 피처 표준편차의 m 배를 상수 편향으로 주입')}


def build_ugv(vae) -> dict:
    """
    UGV: utils/ugv_data (노트북 생성기 이식본) 사용.
      보정셋 : generate_normal(1000, seed=42) 윈도우의 뒤 20 % (benchmarks 와 동일 분할)
      음성셋 : generate_normal(1000, seed=123) 전체 — 보정 미사용
      양성셋 : window_labels == 1 (공격 구간에 완전히 포함된 윈도우)
    """
    n = 1000
    calib_src = ugv_data.generate_normal(n, seed=SEED)
    mean, std = calib_src.mean(0), calib_src.std(0) + 1e-8
    s_all = score_windows(vae, ugv_data.sliding_windows(calib_src, mean, std))
    s_calib = s_all[int(len(s_all) * 0.8):]
    s_neg = score_windows(
        vae, ugv_data.sliding_windows(ugv_data.generate_normal(n, seed=123), mean, std))

    regions = ugv_data.attack_regions(n)
    attacks = {
        'GPS_Spoofing': ugv_data.generate_gps_spoofing(n),
        'Wheel_Slip': ugv_data.generate_wheel_slip(n),
        'Command_Anomaly': ugv_data.generate_command_anomaly(n),
    }

    pos, series = {}, {}
    for name, data in attacks.items():
        X = ugv_data.sliding_windows(data, mean, std)
        s = score_windows(vae, X)
        lab = ugv_data.window_labels(n, regions[name])
        lo = regions[name][0]
        j = np.arange(len(X))
        series[name] = {'scores': s,
                        'full_normal': lab == 0,
                        'boundary': lab == -1,
                        'full_attack': lab == 1,
                        'contamination': np.clip(j + WIN - lo, 0, WIN)}
        pos[name] = s[lab == 1]

    start = ugv_data.attack_start(n)
    base = ugv_data.generate_normal(n, seed=7)
    lab_cont = ugv_data.window_labels(n, (start, n))
    sweep = {}
    for family, cols in (('GPS', (0, 1, 4, 5)), ('Wheel', (6, 7))):
        for m in SIGMA_LEVELS:
            d = inject_sigma(base, start, cols, m, std)
            s = score_windows(vae, ugv_data.sliding_windows(d, mean, std))
            sweep[(family, m)] = s[lab_cont == 1]

    return {'platform': 'ugv',
            'calib_scores': s_calib, 'neg_scores': s_neg,
            'pos_by_attack': pos, 'series': series, 'sweep': sweep,
            'basic_attacks': list(attacks.keys()),
            'calib_desc': 'ugv_data.generate_normal(1000, seed=42) 윈도우의 뒤 20 %',
            'neg_desc': 'ugv_data.generate_normal(1000, seed=123) — 보정에 미사용',
            'pos_desc': 'window_labels == 1 (공격 주입 구간에 완전히 포함된 윈도우)',
            'sweep_desc': ('GPS(gps_vel + residual) / Wheel(wheel_vel_l/r) 에 '
                           '정상 피처 표준편차의 m 배를 상수 편향으로 주입')}


# ─────────────────────────────────────────────────────────────
# 기준별 비교
# ─────────────────────────────────────────────────────────────
def _describe(a: np.ndarray) -> dict:
    return {'n': int(len(a)),
            'mean': round(float(a.mean()), 4),
            'std': round(float(a.std()), 4),
            'min': round(float(a.min()), 4),
            'p50': round(float(np.percentile(a, 50)), 4),
            'p95': round(float(np.percentile(a, 95)), 4),
            'p99': round(float(np.percentile(a, 99)), 4),
            'max': round(float(a.max()), 4)}


def criteria_table(neg: np.ndarray, pos_groups: dict, calib: np.ndarray,
                   cost_ratios) -> dict:
    """음성/양성 점수로 ROC 를 만들고 기준별 임계값·지표 표를 산출."""
    pos = np.concatenate(list(pos_groups.values()))
    y = np.r_[np.zeros(len(neg)), np.ones(len(pos))].astype(int)
    s = np.r_[neg, pos]

    fpr, tpr, thr = roc_curve_strict(y, s)
    candidates = thr[np.isfinite(thr)]
    current = float(np.percentile(calib, 95))

    chosen = {
        'current_normal_p95': current,
        'youden_j': pick_youden(fpr, tpr, thr),
        'target_fpr_1pct': pick_target_fpr(fpr, tpr, thr, 0.01),
        'target_fpr_5pct': pick_target_fpr(fpr, tpr, thr, 0.05),
        'max_f1': pick_max_f1(y, s, candidates),
        'min_cost_fn10': pick_min_cost(y, s, candidates, cost_fn=10.0),
    }

    table = {}
    for name, t in chosen.items():
        m = metrics_at(y, s, t)
        m['alarm_load'] = alarm_load(m['fpr'])
        m['calib_percentile_equivalent'] = round(float((calib <= t).mean() * 100), 2)
        m['recall_by_group'] = {g: round(float((p > t).mean()), 4)
                                for g, p in pos_groups.items()}
        table[name] = m

    sweep = {}
    for k in cost_ratios:
        t = pick_min_cost(y, s, candidates, cost_fn=float(k))
        m = metrics_at(y, s, t)
        sweep[f'{k:g}:1'] = {
            'cost_fn_over_fp': k, 'threshold': m['threshold'],
            'recall': m['recall'], 'fpr': m['fpr'],
            'precision': m['precision'], 'f1': m['f1'],
            'FN': m['FN'], 'FP': m['FP'],
            'false_alarms_per_hour_with_cooldown':
                alarm_load(m['fpr'])['false_alarms_per_hour_with_cooldown'],
        }

    return {'auc': round(auc_trapezoid(fpr, tpr), 4),
            'n_negative': int(len(neg)), 'n_positive': int(len(pos)),
            'criteria': table, 'cost_ratio_sweep': sweep,
            '_roc': (fpr, tpr, thr), '_y': y, '_s': s, '_chosen': chosen}


def separation_margin(neg: np.ndarray, pos: np.ndarray, current: float) -> dict:
    """명목 강도에서 정상/공격 점수가 얼마나 떨어져 있는지 + 현재 임계값의 위치."""
    gap_lo, gap_hi = float(neg.max()), float(pos.min())
    separable = gap_hi > gap_lo
    m_cur_fpr = float((neg > current).mean())
    return {
        'max_normal_score': round(gap_lo, 4),
        'min_attack_score': round(gap_hi, 4),
        'perfectly_separable': bool(separable),
        'separating_threshold_interval': [round(gap_lo, 4), round(gap_hi, 4)]
        if separable else None,
        'gap_ratio_min_attack_over_max_normal': round(gap_hi / gap_lo, 2),
        'current_threshold': round(current, 4),
        'current_below_max_normal': bool(current < gap_lo),
        'current_fpr_on_heldout_normal': round(m_cur_fpr, 4),
        'verdict': ('현재 임계값이 정상 점수 최댓값보다 낮아, 같은 Recall 을 유지하면서도 '
                    'FPR 만 더 내는 파레토 열위 지점'
                    if separable and current < gap_lo else
                    '현재 임계값이 분리 구간 안에 있음'),
    }


# ─────────────────────────────────────────────────────────────
# 경계 윈도우 오염도 / Recall 0.714 분해
# ─────────────────────────────────────────────────────────────
def boundary_contamination(ds: dict, thresholds: dict) -> dict:
    """
    경계 윈도우를 '윈도우 20 샘플 중 공격 샘플 k개' 로 나눠 탐지 여부를 본다.
    최초로 탐지되는 k 가 곧 탐지 지연의 하한 (k 샘플 = k×100 ms @10 Hz).
    """
    out = {}
    for label, thr in thresholds.items():
        per_k, first_k = {}, {}
        for name, sv in ds['series'].items():
            s, cont = sv['scores'], sv['contamination']
            det_k = []
            for k in range(1, WIN):
                sel = cont == k
                if not sel.any():
                    continue
                sc = float(s[sel].mean())
                det = bool((s[sel] > thr).all())
                per_k.setdefault(str(k), {})[name] = {
                    'mean_score': round(sc, 4), 'detected': det}
                if det:
                    det_k.append(k)
            if det_k:
                first_k[name] = {
                    'first_detected_contamination_k': min(det_k),
                    'implied_detection_delay_ms': min(det_k) * int(1000 / SENSOR_RATE_HZ)}
            else:
                first_k[name] = {'first_detected_contamination_k': None,
                                 'implied_detection_delay_ms': None}
        out[label] = {'threshold': round(float(thr), 4),
                      'first_detection': first_k, 'by_k': per_k}
    return out


def recall_decomposition(ds: dict, thresholds: dict) -> dict:
    """
    README「실험 결과」1) 의 Recall ≈ 0.714 가 어디서 나오는지 분해.

    노트북 평가는 공격 시계열 전체를 '공격'으로 라벨링합니다. 그런데 주입은
    시계열의 1/3 지점에서 시작하므로, 그 시계열 앞부분 윈도우(주입 이전 구간만
    담는 윈도우)에는 **주입된 공격이 단 한 샘플도 없습니다.** 이 윈도우들이 전부
    FN 으로 집계되면 Recall 의 상한 자체가 (전체-앞부분)/전체 로 묶입니다.
    아래는 같은 임계값에서 세 원인을 분리 계산한 결과입니다.
      (a) 라벨링 아티팩트 : 공격 미주입 윈도우인데 양성으로 라벨 → 미탐 집계
      (b) 경계 윈도우     : 정상+공격이 섞인 전이 구간
      (c) 임계값 보수성   : 공격으로 완전히 채워진 윈도우인데 놓친 경우
    """
    out = {}
    for label, thr in thresholds.items():
        per_attack = {}
        agg = dict(a_label_artifact_FN=0, b_boundary_FN=0, c_true_missed_FN=0,
                   n_full_attack=0, n_boundary=0, n_full_normal=0,
                   n_windows_total=0, detected_full_attack=0, detected_total=0)
        for name, sv in ds['series'].items():
            s = sv['scores']
            det = s > thr
            fn_norm = int((~det[sv['full_normal']]).sum())
            fn_bnd = int((~det[sv['boundary']]).sum())
            fn_atk = int((~det[sv['full_attack']]).sum())
            per_attack[name] = {
                'n_windows_total': int(len(s)),
                'n_full_normal_content': int(sv['full_normal'].sum()),
                'n_boundary': int(sv['boundary'].sum()),
                'n_full_attack': int(sv['full_attack'].sum()),
                'recall_notebook_style_all_windows_positive':
                    round(float(det.mean()), 4),
                'recall_full_attack_windows_only':
                    round(float(det[sv['full_attack']].mean()), 4),
                'FN_from_label_artifact': fn_norm,
                'FN_from_boundary_windows': fn_bnd,
                'FN_from_true_miss': fn_atk,
            }
            agg['a_label_artifact_FN'] += fn_norm
            agg['b_boundary_FN'] += fn_bnd
            agg['c_true_missed_FN'] += fn_atk
            agg['n_full_attack'] += int(sv['full_attack'].sum())
            agg['n_boundary'] += int(sv['boundary'].sum())
            agg['n_full_normal'] += int(sv['full_normal'].sum())
            agg['n_windows_total'] += int(len(s))
            agg['detected_full_attack'] += int(det[sv['full_attack']].sum())
            agg['detected_total'] += int(det.sum())
        total_fn = (agg['a_label_artifact_FN'] + agg['b_boundary_FN']
                    + agg['c_true_missed_FN'])
        agg['total_FN_notebook_style'] = total_fn
        for key, val in (('share_label_artifact', agg['a_label_artifact_FN']),
                         ('share_boundary', agg['b_boundary_FN']),
                         ('share_true_miss', agg['c_true_missed_FN'])):
            agg[key] = round(val / total_fn, 4) if total_fn else None
        agg['recall_notebook_style'] = round(
            agg['detected_total'] / agg['n_windows_total'], 4)
        agg['recall_full_attack_windows_only'] = round(
            agg['detected_full_attack'] / agg['n_full_attack'], 4)
        out[label] = {'threshold': round(float(thr), 4),
                      'per_attack': per_attack, 'aggregate': agg}
    return out


def threshold_needed_for_notebook_recall(ds: dict, targets=(0.90, 0.95, 0.99)) -> dict:
    """
    노트북식 라벨링(공격 시계열 전체 = 양성)에서 목표 Recall 을 달성하려면
    임계값을 어디까지 내려야 하고, 그 대가(정상 데이터 FPR)는 얼마인지.
    """
    pooled = np.concatenate([sv['scores'] for sv in ds['series'].values()])
    neg = ds['neg_scores']
    out = {}
    for tgt in targets:
        thr = float(np.quantile(pooled, 1.0 - tgt))
        fpr = float((neg > thr).mean())
        out[f'recall_{tgt:.2f}'] = {
            'required_threshold': round(thr, 4),
            'achieved_recall_notebook_style': round(float((pooled > thr).mean()), 4),
            'heldout_normal_fpr': round(fpr, 4),
            'false_alarms_per_hour_with_cooldown':
                alarm_load(fpr)['false_alarms_per_hour_with_cooldown'],
        }
    return out


# ─────────────────────────────────────────────────────────────
# 플롯
# ─────────────────────────────────────────────────────────────
def _style(ax):
    ax.set_facecolor(C_SURFACE)
    ax.tick_params(colors=C_MUTED, labelsize=8.5)
    for spine in ('top', 'right'):
        ax.spines[spine].set_visible(False)
    for spine in ('left', 'bottom'):
        ax.spines[spine].set_color(C_MUTED)
    ax.grid(color='#d8d8d4', lw=0.6)
    ax.set_axisbelow(True)


def plot_roc(report: dict, out_path: Path):
    fig, axes = plt.subplots(1, 2, figsize=(12.6, 5.6), facecolor=C_SURFACE)
    for ax, platform in zip(axes, ('uav', 'ugv')):
        r = report[platform]
        _style(ax)
        f_n, t_n, _ = r['_nominal']['_roc']
        ax.plot(f_n, t_n, color=C_ATTACK, lw=2,
                label=f'nominal attacks (AUC = {r["_nominal"]["auc"]:.4f})')
        f_m, t_m, _ = r['_marginal']['_roc']
        ax.plot(f_m, t_m, color=C_WEAK, lw=2,
                label=f'marginal-strength attacks (AUC = {r["_marginal"]["auc"]:.4f})')
        ax.plot([0, 1], [0, 1], color=C_MUTED, lw=1, ls=':', label='random')
        for name, t in r['_marginal']['_chosen'].items():
            m = r['_marginal']['criteria'][name]
            ax.scatter([m['fpr']], [m['recall']], s=60, zorder=5,
                       color=CRITERION_COLORS.get(name, C_TEXT),
                       edgecolor=C_SURFACE, linewidth=1.1,
                       label=f'{name}: thr={t:.3f}, '
                             f'TPR={m["recall"]:.3f}, FPR={m["fpr"]:.3f}')
        ax.set_xlabel('False Positive Rate (held-out normal windows)', color=C_MUTED)
        ax.set_ylabel('True Positive Rate (Recall)', color=C_MUTED)
        ax.set_title(f'{platform.upper()} — anomaly score ROC\n'
                     f'operating points chosen on the marginal-strength set',
                     color=C_TEXT, fontsize=11)
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.03)
        ax.legend(fontsize=7.2, frameon=False, labelcolor=C_TEXT, loc='lower right')
    fig.suptitle('Threshold selection — ROC and candidate operating points',
                 color=C_TEXT, fontsize=12.5, fontweight='bold')
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out_path, dpi=150, facecolor=C_SURFACE)
    plt.close(fig)


def plot_tradeoff(report: dict, out_path: Path):
    fig, axes = plt.subplots(2, 2, figsize=(12.6, 8.4), facecolor=C_SURFACE)
    for col, platform in enumerate(('uav', 'ugv')):
        r = report[platform]
        neg = r['_neg']
        pos_nom = r['_pos_nominal']
        pos_mar = r['_pos_marginal']

        ax = axes[0, col]
        _style(ax)
        lo = max(min(neg.min(), pos_mar.min()), 1e-4)
        hi = max(neg.max(), pos_nom.max())
        bins = np.logspace(np.log10(lo * 0.9), np.log10(hi * 1.3), 80)
        ax.hist(neg, bins=bins, color=C_NORMAL, alpha=0.6, label='normal (held-out)')
        ax.hist(pos_mar, bins=bins, color=C_WEAK, alpha=0.6, label='attack (marginal)')
        ax.hist(pos_nom, bins=bins, color=C_ATTACK, alpha=0.6, label='attack (nominal)')
        for name, t in r['_marginal']['_chosen'].items():
            ax.axvline(t, color=CRITERION_COLORS.get(name, C_TEXT), lw=1.4, ls='--',
                       label=f'{name} = {t:.3f}')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel('reconstruction error (log)', color=C_MUTED)
        ax.set_ylabel('window count (log)', color=C_MUTED)
        ax.set_title(f'{platform.upper()} — score distributions and thresholds',
                     color=C_TEXT, fontsize=11)
        ax.legend(fontsize=6.6, frameon=False, labelcolor=C_TEXT, ncol=2)

        ax = axes[1, col]
        _style(ax)
        y, s = r['_marginal']['_y'], r['_marginal']['_s']
        grid = np.quantile(s, np.linspace(0.002, 0.998, 250))
        ax.plot(grid, [(pos_mar > t).mean() for t in grid],
                color=C_ATTACK, lw=2, label='Recall (marginal attacks)')
        ax.plot(grid, [(neg > t).mean() for t in grid],
                color=C_NORMAL, lw=2, label='FPR (normal)')
        ax.plot(grid, [metrics_at(y, s, t)['f1'] for t in grid],
                color=C_OK, lw=1.6, ls='-.', label='F1')
        for name, t in r['_marginal']['_chosen'].items():
            ax.axvline(t, color=CRITERION_COLORS.get(name, C_TEXT), lw=1.2, ls='--')
        ax.set_xscale('log')
        ax.set_xlabel('threshold (log)', color=C_MUTED)
        ax.set_ylabel('rate', color=C_MUTED)
        ax.set_title(f'{platform.upper()} — Recall / FPR / F1 vs threshold',
                     color=C_TEXT, fontsize=11)
        ax.legend(fontsize=8, frameon=False, labelcolor=C_TEXT)

    fig.suptitle('Threshold trade-off — missed detection (FN) vs false alarm (FP)',
                 color=C_TEXT, fontsize=12.5, fontweight='bold')
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out_path, dpi=150, facecolor=C_SURFACE)
    plt.close(fig)


def plot_sensitivity(report: dict, out_path: Path):
    fig, axes = plt.subplots(1, 2, figsize=(12.6, 4.8), facecolor=C_SURFACE)

    ax = axes[0]
    _style(ax)
    markers = {'uav': 'o', 'ugv': 's'}
    palette = [C_ATTACK, C_NORMAL, C_WEAK, C_OK]
    ci = 0
    for platform in ('uav', 'ugv'):
        for family, rows in report[platform]['magnitude_sweep'].items():
            xs = [row['magnitude_sigma'] for row in rows]
            ys = [row['auc'] for row in rows]
            ax.plot(xs, ys, marker=markers[platform], lw=1.8, ms=5,
                    color=palette[ci % len(palette)],
                    label=f'{platform.upper()} {family}')
            ci += 1
    ax.axhline(0.99, color=C_MUTED, lw=1, ls=':')
    ax.text(0.052, 0.992, 'AUC = 0.99', color=C_MUTED, fontsize=8)
    ax.set_xscale('log')
    ax.set_xlabel('injected bias (multiples of normal feature σ, log)', color=C_MUTED)
    ax.set_ylabel('ROC AUC vs held-out normal', color=C_MUTED)
    ax.set_title('Detection limit — AUC vs attack strength', color=C_TEXT, fontsize=11)
    ax.legend(fontsize=8, frameon=False, labelcolor=C_TEXT, loc='lower right')

    ax = axes[1]
    _style(ax)
    ci = 0
    for platform in ('uav', 'ugv'):
        bc = report[platform]['boundary_contamination']['current_normal_p95']
        ks = sorted(int(k) for k in bc['by_k'])
        for name in list(bc['by_k'][str(ks[0])].keys())[:2]:
            ys = [bc['by_k'][str(k)][name]['mean_score'] for k in ks]
            ax.plot(ks, ys, marker='o', ms=4, lw=1.7,
                    color=palette[ci % len(palette)],
                    label=f'{platform.upper()} {name}')
            ci += 1
        ax.axhline(bc['threshold'], color=CRITERION_COLORS['current_normal_p95'],
                   lw=1.2, ls='--')
    ax.set_yscale('log')
    ax.set_xlabel('attack samples inside the 20-sample window (k)', color=C_MUTED)
    ax.set_ylabel('mean anomaly score (log)', color=C_MUTED)
    ax.set_title('Boundary windows — score vs window contamination\n'
                 '(dashed: current p95 threshold)', color=C_TEXT, fontsize=11)
    ax.legend(fontsize=8, frameon=False, labelcolor=C_TEXT)

    fig.suptitle('Threshold sensitivity — attack strength and window contamination',
                 color=C_TEXT, fontsize=12.5, fontweight='bold')
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out_path, dpi=150, facecolor=C_SURFACE)
    plt.close(fig)


# ─────────────────────────────────────────────────────────────
def analyse(ds: dict, cost_ratios) -> dict:
    neg = ds['neg_scores']
    calib = ds['calib_scores']

    nominal = criteria_table(neg, ds['pos_by_attack'], calib, cost_ratios)

    # 강도 스윕 → 개별 AUC
    sweep_rows, marginal_groups = {}, {}
    for (family, m), ps in ds['sweep'].items():
        y = np.r_[np.zeros(len(neg)), np.ones(len(ps))].astype(int)
        s = np.r_[neg, ps]
        auc = roc_auc(y, s)
        sweep_rows.setdefault(family, []).append({
            'magnitude_sigma': m,
            'n_positive': int(len(ps)),
            'attack_score_mean': round(float(ps.mean()), 4),
            'auc': round(auc, 4),
            'recall_at_current_p95': round(
                float((ps > np.percentile(calib, 95)).mean()), 4),
        })
        if MARGINAL_AUC_RANGE[0] <= auc <= MARGINAL_AUC_RANGE[1]:
            marginal_groups[f'{family}_{m}sigma'] = ps
    for family in sweep_rows:
        sweep_rows[family].sort(key=lambda r: r['magnitude_sigma'])

    if not marginal_groups:      # 전 구간이 완벽 분리면 가장 약한 강도만 사용
        weakest = min(ds['sweep'], key=lambda k: k[1])
        marginal_groups[f'{weakest[0]}_{weakest[1]}sigma'] = ds['sweep'][weakest]
    marginal = criteria_table(neg, marginal_groups, calib, cost_ratios)

    pos_nominal = np.concatenate(list(ds['pos_by_attack'].values()))
    pos_marginal = np.concatenate(list(marginal_groups.values()))

    per_attack_auc = {}
    for name, ps in ds['pos_by_attack'].items():
        y = np.r_[np.zeros(len(neg)), np.ones(len(ps))].astype(int)
        per_attack_auc[name] = round(roc_auc(y, np.r_[neg, ps]), 4)

    basic = np.concatenate([ds['pos_by_attack'][k] for k in ds['basic_attacks']])
    auc_basic = roc_auc(np.r_[np.zeros(len(neg)), np.ones(len(basic))].astype(int),
                        np.r_[neg, basic])

    current = float(np.percentile(calib, 95))
    thresholds_of_interest = {
        'current_normal_p95': current,
        'marginal_youden_j': marginal['_chosen']['youden_j'],
        'marginal_min_cost_fn10': marginal['_chosen']['min_cost_fn10'],
    }

    return {
        'dataset': {
            'calibration': ds['calib_desc'],
            'negatives': ds['neg_desc'],
            'positives_nominal': ds['pos_desc'],
            'magnitude_sweep': ds['sweep_desc'],
            'n_calibration_windows': int(len(calib)),
            'n_negative_windows': int(len(neg)),
            'n_positive_windows_nominal': int(len(pos_nominal)),
            'n_positive_windows_marginal': int(len(pos_marginal)),
            'marginal_set_rule':
                f'개별 AUC 가 {MARGINAL_AUC_RANGE[0]}~{MARGINAL_AUC_RANGE[1]} 인 '
                f'주입 강도만 모은 집합',
            'marginal_set_members': list(marginal_groups.keys()),
            # 음성 표본 개수가 곧 FPR 해상도의 하한이다. 981개 윈도우(=98.1 s @10 Hz)로는
            # 1/981 보다 작은 FPR 을 구분할 수 없으므로, 그 아래 구간의 오경보율 주장은
            # 이 평가셋으로 검증 불가 — 실데이터 장시간 수집이 필요한 지점.
            'min_resolvable_fpr': round(1.0 / len(neg), 6),
            'min_resolvable_fpr_alarm_load': alarm_load(1.0 / len(neg)),
            'negative_sample_duration_sec': round(len(neg) / SENSOR_RATE_HZ, 1),
        },
        'score_distribution': {
            'calibration_normal': _describe(calib),
            'heldout_normal': _describe(neg),
            'attack_nominal': _describe(pos_nominal),
            'attack_marginal': _describe(pos_marginal),
        },
        'auc': {
            'nominal_pooled_all_attacks': nominal['auc'],
            'nominal_pooled_basic_attacks': round(auc_basic, 4),
            'nominal_per_attack': per_attack_auc,
            'marginal_pooled': marginal['auc'],
        },
        'separation_margin_nominal': separation_margin(neg, pos_nominal, current),
        'criteria_nominal': nominal['criteria'],
        'criteria_marginal': marginal['criteria'],
        'cost_ratio_sweep_marginal': marginal['cost_ratio_sweep'],
        'magnitude_sweep': sweep_rows,
        'boundary_contamination': boundary_contamination(ds, thresholds_of_interest),
        'recall_decomposition': recall_decomposition(ds, thresholds_of_interest),
        'threshold_for_notebook_style_recall':
            threshold_needed_for_notebook_recall(ds),
        '_nominal': nominal, '_marginal': marginal,
        '_neg': neg, '_pos_nominal': pos_nominal, '_pos_marginal': pos_marginal,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cost-ratios', type=float, nargs='+',
                    default=[1, 3, 10, 30, 100],
                    help='미탐(FN) 비용 / 오탐(FP) 비용 비율 스윕')
    ap.add_argument('--no-plot', action='store_true')
    args = ap.parse_args()

    set_seed(SEED)
    out_dir = Path(__file__).resolve().parent
    img_dir = REPO_ROOT / 'docs' / 'images'
    img_dir.mkdir(parents=True, exist_ok=True)

    print('=' * 72)
    print('임계값 선택 근거 — ROC / AUC 기반 재검토')
    print('=' * 72)

    payload = {
        'meta': {
            'generated_at': time.strftime('%Y-%m-%dT%H:%M:%S%z'),
            'environment': env_info(),
            'seed': SEED,
            'score_path': ('anomaly_detector._infer 와 동일 (VAE forward + MSE, '
                           'z = mu + std·eps 샘플링, torch.manual_seed 고정)'),
            'decision_rule': 'score > threshold (strict) — anomaly_detector 와 동일',
            'sensor_rate_hz': SENSOR_RATE_HZ,
            'alert_cooldown_sec': COOLDOWN_SEC,
            'sigma_levels': list(SIGMA_LEVELS),
            'marginal_auc_range': list(MARGINAL_AUC_RANGE),
            'note': '모든 수치는 본 스크립트 실행으로 실측한 값 (추정치 없음)',
        },
        'platforms': {},
    }

    report = {}
    for platform in ('uav', 'ugv'):
        vae, _ = load_vae(platform, WIN)
        ds = build_uav(vae) if platform == 'uav' else build_ugv(vae)
        r = analyse(ds, args.cost_ratios)
        report[platform] = r

        print(f'\n[{platform.upper()}] 명목 강도 AUC={r["auc"]["nominal_pooled_all_attacks"]:.4f}'
              f' / 한계 근방 AUC={r["auc"]["marginal_pooled"]:.4f}'
              f'  (neg={r["dataset"]["n_negative_windows"]}, '
              f'pos_nom={r["dataset"]["n_positive_windows_nominal"]}, '
              f'pos_mar={r["dataset"]["n_positive_windows_marginal"]})')
        sm = r['separation_margin_nominal']
        print(f'  분리 여유: 정상 max={sm["max_normal_score"]} < 공격 min='
              f'{sm["min_attack_score"]} → {sm["verdict"]}')
        for tag, key in (('명목', 'criteria_nominal'), ('한계근방', 'criteria_marginal')):
            print(f'  [{tag} 평가셋] {"criterion":<22}{"thr":>9}{"P":>8}'
                  f'{"R":>8}{"F1":>8}{"FPR":>9}')
            for name, m in r[key].items():
                print(f'  {"":<11}{name:<22}{m["threshold"]:>9.4f}{m["precision"]:>8.3f}'
                      f'{m["recall"]:>8.3f}{m["f1"]:>8.3f}{m["fpr"]:>9.4f}')
        agg = r['recall_decomposition']['current_normal_p95']['aggregate']
        print(f'  FN 분해 @현재임계값: 라벨아티팩트 {agg["a_label_artifact_FN"]} / '
              f'경계 {agg["b_boundary_FN"]} / 진짜미탐 {agg["c_true_missed_FN"]} '
              f'→ 노트북식 Recall={agg["recall_notebook_style"]:.4f}, '
              f'완전공격윈도우 Recall={agg["recall_full_attack_windows_only"]:.4f}')

    if not args.no_plot:
        plot_roc(report, img_dir / 'threshold_roc.png')
        plot_tradeoff(report, img_dir / 'threshold_tradeoff.png')
        plot_sensitivity(report, img_dir / 'threshold_sensitivity.png')

    for platform, r in report.items():
        payload['platforms'][platform] = {k: v for k, v in r.items()
                                          if not k.startswith('_')}

    with open(out_dir / 'threshold_results.json', 'w') as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print()
    print(f'[OK] {out_dir / "threshold_results.json"}')
    if not args.no_plot:
        for name in ('threshold_roc.png', 'threshold_tradeoff.png',
                     'threshold_sensitivity.png'):
            print(f'[OK] {img_dir / name}')


if __name__ == '__main__':
    main()
