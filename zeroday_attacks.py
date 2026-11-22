#!/usr/bin/env python3
"""
본선 추가 연구: 제로데이 공격 탐지 실험
standalone (ROS2 불필요, 기존 vae_uav.pth 직접 로드)

"제로데이"의 정의 — 예선·advanced_attacks.py 의 9개 시나리오에 **한 번도 등장하지
않은 패턴**의 공격. 예선 공격은 모두 "정상값 + 큰 편향(덧셈)" 형태였다. 여기서는
값을 더하지 않고 **센서 스트림 자체의 시간 구조를 바꾸는** 공격을 다룬다.

  1. GPS Freeze   — GPS 속도가 공격 직전 값에 고정 (수신기 jamming / 동결)
  2. GPS Replay   — 과거 GPS 값을 lag 만큼 지연시켜 재생 (기록-재생 스푸핑)
  3. GPS Drift    — 아주 작은 선형 편향이 상한 없이 누적 (meaconing 류 느린 끌기)
  4. Sensor-bus Freeze (보너스) — 8개 피처 전부 동결 (센서 버스 사망 / 데이터 정지)

advanced_attacks.py 는 수정하지 않고 import 만 한다 (main() 결과는 동결된 증거).
보정 경로(가중치, 정상 1,000샘플 seed=42, 95th percentile, 테스트 seed=123,
ATK_START=100, N_TOTAL=300)는 advanced_attacks.main 과 동일하게 맞춰
advanced_attack_results.json 과 수치를 직접 비교할 수 있게 했다.
"""

import json
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.font_manager as fm

from advanced_attacks import (VAE, Detector, gen_normal, _recalc_residual,
                              run_experiment, plot_scores,
                              WIN, N_FEAT, BASE, UAV_FEATS)

# 한글 폰트가 있으면 사용 (없으면 matplotlib 기본 폰트, 글자만 깨지고 수치는 동일)
_KOR_FONTS = ['AppleGothic', 'Apple SD Gothic Neo', 'NanumGothic',
              'Malgun Gothic', 'Noto Sans CJK KR', 'Noto Sans KR']
_AVAILABLE = {f.name for f in fm.fontManager.ttflist}
KOR_FONT = next((f for f in _KOR_FONTS if f in _AVAILABLE), None)
if KOR_FONT:
    matplotlib.rcParams['font.family'] = KOR_FONT
matplotlib.rcParams['axes.unicode_minus'] = False

GPS_COLS = (0, 1)
SEED = 42  # VAE 는 추론 시에도 재파라미터화 샘플링을 쓰므로 torch 시드를 고정

N_TOTAL   = 300
ATK_START = 100


# ─────────────────────────────────────────────────────────────
# 제로데이 공격 주입 함수
# ─────────────────────────────────────────────────────────────
def atk_gps_freeze(data: np.ndarray, start: int) -> np.ndarray:
    """GPS Freeze: start 이후 gps_vel_x/y 가 start-1 시점 값에 고정.

    GPS 수신기가 jamming 으로 마지막 유효 해(solution)를 계속 내보내는 상황.
    값 자체는 정상 범위 안이므로 단일 샘플로는 구분 불가 — IMU 와의 잔차가 유일한 단서.
    """
    if start < 1:
        raise ValueError('start 는 1 이상이어야 합니다 (start-1 값을 고정값으로 사용)')
    d = data.copy()
    d[start:, GPS_COLS] = d[start - 1, GPS_COLS]
    return _recalc_residual(d)


def atk_gps_replay(data: np.ndarray, start: int, lag: int = 100) -> np.ndarray:
    """GPS Replay: start 이후 gps_vel_x/y 를 lag 스텝 전의 **원본** 값으로 치환.

    공격자가 정상 GPS 스트림을 녹음해 두었다가 지연 재생하는 기록-재생 스푸핑.
    원본 인덱스 t-lag 가 음수가 되면 공격 전 구간(0..start-1) 안에서 순환시켜
    인덱스를 항상 유효하게 유지한다. 재생 값은 모두 실제 정상 샘플이므로
    피처별 분포는 정상과 동일 — 역시 잔차만이 단서.
    """
    if lag < 1:
        raise ValueError('lag 는 1 이상이어야 합니다')
    if start < 1:
        raise ValueError('start 는 1 이상이어야 합니다')
    d = data.copy()
    t = np.arange(start, len(d))
    src = t - lag
    neg = src < 0
    if neg.any():
        src[neg] = np.mod(src[neg], start)   # 공격 전 구간 안으로 순환
    d[start:, GPS_COLS] = data[src][:, GPS_COLS]
    return _recalc_residual(d)


def atk_gps_drift(data: np.ndarray, start: int,
                  rate_per_step: float = 0.02,
                  max_mag: float | None = None) -> np.ndarray:
    """GPS Drift: gps_vel_x 에 rate·(t-start) 의 선형 편향 누적 (y 축은 0.5배).

    예선의 점진적 공격(atk_gradual_gps)은 rate 0.1~0.3/step, 상한 12 로 "결국
    큰 공격이 되는 램프" 였다. 드리프트는 rate 가 5~60배 작고 상한이 없다
    (max_mag=None). 300 스텝 전체에서도 편향이 정상 GPS 표준편차(≈0.58)
    수준에 머물러, 급격 공격용으로 보정된 임계값을 그대로 우회하려는 시나리오.
    """
    d = data.copy()
    n = len(d) - start
    bias = np.arange(n, dtype=np.float32) * np.float32(rate_per_step)
    if max_mag is not None:
        bias = np.minimum(bias, np.float32(max_mag))
    d[start:, 0] += bias
    d[start:, 1] += bias * 0.5
    return _recalc_residual(d)


def atk_sensor_freeze_all(data: np.ndarray, start: int) -> np.ndarray:
    """Sensor-bus Freeze (보너스): start 이후 8개 피처 전부 start-1 행에 고정.

    센서 버스/퍼블리셔가 죽어 마지막 메시지가 반복 전달되는 상황. 모든 피처가
    정상 범위의 값이고 잔차 정합성(residual = gps - imu)도 그대로 유지된다 —
    분포 기반 탐지기에는 가장 보이지 않는 종류의 이상.
    """
    if start < 1:
        raise ValueError('start 는 1 이상이어야 합니다')
    d = data.copy()
    d[start:] = d[start - 1]
    return _recalc_residual(d)


# ─────────────────────────────────────────────────────────────
# 보정 (advanced_attacks.main 과 동일 경로)
# ─────────────────────────────────────────────────────────────
def calibrate(seed: int = SEED):
    vae = VAE(input_dim=WIN * N_FEAT)
    weight_path = BASE / 'vae_uav.pth'
    vae.load_state_dict(torch.load(weight_path, map_location='cpu', weights_only=True))
    vae.eval()

    normal = gen_normal(1000, seed=42)
    mean = normal.mean(0)
    std  = normal.std(0) + 1e-8

    torch.manual_seed(seed)
    cal_scores = []
    for i in range(WIN, len(normal)):
        w = normal[i - WIN: i]
        normed = (w - mean) / std
        x = torch.FloatTensor(normed.reshape(1, -1))
        with torch.no_grad():
            recon, _, _ = vae(x)
            cal_scores.append(torch.mean((recon - x) ** 2).item())
    threshold = float(np.percentile(cal_scores, 95))
    return Detector(vae, threshold, mean, std), normal, np.array(cal_scores)


def normal_contrib_stats(det: Detector, normal: np.ndarray, n_win: int = 500):
    """정상 윈도우의 피처별 재구성 오차 평균/표준편차 (z-score 기준용)."""
    contribs = np.array([det.per_feat_contrib(normal, i)
                         for i in range(WIN, min(n_win, len(normal)))])
    return contribs.mean(0), contribs.std(0) + 1e-9


def top_features(det: Detector, data: np.ndarray, score_idx: int,
                 mu: np.ndarray, sig: np.ndarray, k: int = 3) -> list:
    """score_series 의 score_idx 번째 점수에 해당하는 윈도우(data[idx-WIN+1 : idx+1])
    의 피처별 재구성 오차 상위 k 개. 정상 분포 대비 z-score 도 같이 기록."""
    contrib = det.per_feat_contrib(data, score_idx + 1)
    order = np.argsort(contrib)[::-1][:k]
    return [{'feature':  UAV_FEATS[i],
             'contrib':  round(float(contrib[i]), 4),
             'share':    round(float(contrib[i] / contrib.sum()), 3),
             'z_score':  round(float((contrib[i] - mu[i]) / sig[i]), 1)}
            for i in order]


# ─────────────────────────────────────────────────────────────
# 메인
# ─────────────────────────────────────────────────────────────
def main():
    print('=' * 72)
    print('해금 본선 추가 연구 — 제로데이 공격 탐지 실험 (zeroday_attacks.py)')
    print('=' * 72)

    det, normal, cal_scores = calibrate()
    normal_min = float(cal_scores.min())
    print(f'[OK] VAE 로드 + 보정 완료: 임계값 = {det.th:.4f} (torch seed={SEED})')

    base_normal = gen_normal(N_TOTAL, seed=123)

    ATTACKS = [
        ('제로데이_GPS_Freeze',           atk_gps_freeze(base_normal, ATK_START)),
        ('제로데이_GPS_Replay lag=100',   atk_gps_replay(base_normal, ATK_START, lag=100)),
        ('제로데이_GPS_Drift rate=0.02',  atk_gps_drift(base_normal, ATK_START, rate_per_step=0.02)),
        ('제로데이_GPS_Drift rate=0.005', atk_gps_drift(base_normal, ATK_START, rate_per_step=0.005)),
        ('제로데이_SensorBus_FreezeAll',  atk_sensor_freeze_all(base_normal, ATK_START)),
    ]

    mu, sig = normal_contrib_stats(det, normal)

    results, summary = [], []
    for name, data in ATTACKS:
        torch.manual_seed(SEED)              # 시나리오마다 동일 샘플링 경로
        res = run_experiment(det, data, name, ATK_START, N_TOTAL)
        results.append(res)

        atk_scores = res['scores'][ATK_START + WIN - 1:]   # 공격으로 가득 찬 윈도우
        entry = {k: v for k, v in res.items() if k not in ('scores', 'labels', 'preds')}
        entry['attack_score_mean'] = round(float(atk_scores.mean()), 4)
        entry['attack_score_max']  = round(float(atk_scores.max()), 4)
        entry['attack_score_min']  = round(float(atk_scores.min()), 4)
        # 점수가 정상 보정셋 최솟값보다 *낮은* 윈도우 비율 — 상한 단일 임계값이
        # 놓치는 "지나치게 규칙적인" 이상을 드러내기 위한 보조 지표
        entry['frac_below_normal_min'] = round(float((atk_scores < normal_min).mean()), 3)
        if res['delay_ts'] >= 0:
            first_idx = ATK_START + res['delay_ts']
            entry['first_detect_idx'] = first_idx
            entry['top_feats'] = top_features(det, data, first_idx, mu, sig)
        else:
            entry['first_detect_idx'] = None
            entry['top_feats'] = []
        summary.append(entry)

        d = f'{res["delay_ms"]}ms' if res['delay_ms'] >= 0 else '미탐'
        tf = ', '.join(f'{t["feature"]}({t["share"]:.0%})' for t in entry['top_feats']) or '-'
        print(f'  [{name}]')
        print(f'    TP={res["TP"]} FP={res["FP"]} FN={res["FN"]}  '
              f'P={res["Precision"]:.3f} R={res["Recall"]:.3f} F1={res["F1"]:.3f}  '
              f'delay={d}  score(mean/max)={entry["attack_score_mean"]:.3f}/'
              f'{entry["attack_score_max"]:.3f}')
        print(f'    top feats @first detect: {tf}')

    # 절제 실험(ablation): 잔차를 재계산하지 않으면(= GPS-IMU 교차검증이 없는
    # 시스템이라면) Freeze/Replay 가 보이는가? gen_normal 은 스텝 간 시간 구조가
    # 없으므로, 여기서 탐지가 사라지면 탐지의 근거가 GPS 패턴이 아니라 잔차임이 확인된다.
    ablation = []
    for name, fn in [('Freeze (잔차 고정)', atk_gps_freeze),
                     ('Replay (잔차 고정)', atk_gps_replay)]:
        d_abl = fn(base_normal, ATK_START)
        d_abl[:, 4:6] = base_normal[:, 4:6]          # 잔차를 공격 전 원본으로 되돌림
        torch.manual_seed(SEED)
        r = run_experiment(det, d_abl, name, ATK_START, N_TOTAL)
        a_sc = r['scores'][ATK_START + WIN - 1:]
        ablation.append({'attack': name, 'TP': r['TP'], 'FP': r['FP'], 'FN': r['FN'],
                         'Precision': r['Precision'], 'Recall': r['Recall'], 'F1': r['F1'],
                         'delay_ms': r['delay_ms'],
                         'attack_score_mean': round(float(a_sc.mean()), 4),
                         'attack_score_max':  round(float(a_sc.max()), 4)})
        dd = f'{r["delay_ms"]}ms' if r['delay_ms'] >= 0 else '미탐'
        print(f'  [절제: {name}]  TP={r["TP"]} FN={r["FN"]}  R={r["Recall"]:.3f}  '
              f'delay={dd}  score(mean/max)={a_sc.mean():.3f}/{a_sc.max():.3f}')

    # 그래프 (5개를 한 장에 넣으면 너무 길어 두 장으로 분리)
    img_dir = BASE / 'docs' / 'images'
    img_dir.mkdir(parents=True, exist_ok=True)
    # plot_scores 는 제목에 delay_ms 를 그대로 찍으므로 미탐(-1)은 표시용으로 바꿔 넘긴다
    def _for_plot(r):
        return r if r['delay_ms'] >= 0 else {**r, 'delay_ms': '미탐 '}
    plot_scores([_for_plot(results[i]) for i in (0, 1, 4)], det.th, ATK_START,
                img_dir / 'result_zeroday_attacks.png')
    plot_scores([_for_plot(results[i]) for i in (2, 3)], det.th, ATK_START,
                img_dir / 'result_zeroday_drift.png')

    out = {
        'threshold': det.th,
        'seed': SEED,
        'normal_score_stats': {'min': round(normal_min, 4),
                               'p1':  round(float(np.percentile(cal_scores, 1)), 4),
                               'p5':  round(float(np.percentile(cal_scores, 5)), 4),
                               'mean': round(float(cal_scores.mean()), 4),
                               'p95': round(float(np.percentile(cal_scores, 95)), 4),
                               'max': round(float(cal_scores.max()), 4)},
        'calibration': {'normal_n': 1000, 'normal_seed': 42, 'percentile': 95,
                        'test_seed': 123, 'atk_start': ATK_START, 'n_total': N_TOTAL,
                        'win': WIN},
        'results': summary,
        'ablation_no_residual': ablation,
    }
    out_path = BASE / 'zeroday_results.json'
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f'[저장] {out_path}')

    print('\n' + '=' * 72)
    print(f'{"공격 유형":<32} {"TP":>4} {"FP":>3} {"FN":>4} {"P":>6} {"R":>6} '
          f'{"F1":>6} {"delay":>7}  top feature')
    print('-' * 72)
    for e in summary:
        d = f'{e["delay_ms"]}ms' if e['delay_ms'] >= 0 else '미탐'
        tf = e['top_feats'][0]['feature'] if e['top_feats'] else '-'
        print(f'{e["attack"]:<32} {e["TP"]:>4} {e["FP"]:>3} {e["FN"]:>4} '
              f'{e["Precision"]:>6.3f} {e["Recall"]:>6.3f} {e["F1"]:>6.3f} '
              f'{d:>7}  {tf}')
    print('=' * 72)
    print('실험 완료.')
    return out


if __name__ == '__main__':
    main()
