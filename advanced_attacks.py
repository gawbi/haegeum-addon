#!/usr/bin/env python3
"""
본선 추가 연구: 복합 공격 및 점진적 스푸핑 탐지 실험
standalone (ROS2 불필요, 기존 vae_uav.pth 직접 로드)

실험 목적:
  1. 복합 공격(두 센서 동시 오염)이 탐지 성능에 미치는 영향
  2. 점진적 스푸핑(값을 서서히 올려 임계값 우회 시도)의 탐지 지연
  3. FP/FN 발생 윈도우의 피처 분포 분석 — 어떤 피처·구간에서 실패하는지 체계화
"""

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')  # 화면 없는 환경용
import matplotlib.pyplot as plt
matplotlib.rcParams['axes.unicode_minus'] = False

BASE = Path(__file__).parent

# ─────────────────────────────────────────────────────────────
# VAE (anomaly_detector.py 와 동일한 구조)
# ─────────────────────────────────────────────────────────────
class VAE(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int = 16):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128), nn.ReLU(),
            nn.Linear(128, 64),        nn.ReLU(),
        )
        self.fc_mu     = nn.Linear(64, latent_dim)
        self.fc_logvar = nn.Linear(64, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64), nn.ReLU(),
            nn.Linear(64, 128),        nn.ReLU(),
            nn.Linear(128, input_dim),
        )

    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        return mu + std * torch.randn_like(std)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decoder(z), mu, logvar


# ─────────────────────────────────────────────────────────────
# 정상 데이터 생성 (UAV, 예선 동일 분포)
# ─────────────────────────────────────────────────────────────
UAV_FEATS = ['gps_vel_x','gps_vel_y','imu_ax','imu_ay',
             'residual_x','residual_y','baro_alt','pitch_rate']
N_FEAT    = 8
WIN       = 20

def gen_normal(n: int, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    speed  = rng.uniform(1.0, 3.0, (n, 2))
    imu    = speed + rng.normal(0, 0.05, (n, 2))
    resid  = speed - imu
    alt    = rng.uniform(49.7, 50.3, (n, 1))
    pitch  = rng.normal(0, 0.01, (n, 1))
    return np.concatenate([speed, imu, resid, alt, pitch], axis=1).astype(np.float32)


# ─────────────────────────────────────────────────────────────
# 공격 주입 함수
# ─────────────────────────────────────────────────────────────
def _recalc_residual(d: np.ndarray) -> np.ndarray:
    d[:, 4] = d[:, 0] - d[:, 2]
    d[:, 5] = d[:, 1] - d[:, 3]
    return d

def atk_gps_sudden(data, start, mag=12.0, seed=0):
    d = data.copy()
    n = len(d) - start
    t = np.arange(n)
    d[start:, 0] += mag * np.sin(t * 0.2)
    d[start:, 1] += mag * 0.5 * np.sin(t * 0.2)
    return _recalc_residual(d)

def atk_altitude_sudden(data, start, mag=25.0):
    d = data.copy()
    d[start:, 6] += mag
    return d

def atk_cmd_injection(data, start, seed=77):
    d = data.copy()
    rng = np.random.default_rng(seed)
    d[start:, 7] += rng.uniform(0.5, 0.8, len(d) - start)
    return d

def atk_composite_gps_cmd(data, start):
    """복합 공격 A: GPS Spoofing + Command Injection 동시"""
    return atk_cmd_injection(atk_gps_sudden(data, start), start)

def atk_composite_gps_alt(data, start):
    """복합 공격 B: GPS Spoofing + Altitude Spoofing 동시"""
    return atk_altitude_sudden(atk_gps_sudden(data, start), start)

def atk_gradual_gps(data, start, rate=0.3, max_mag=12.0):
    """점진적 GPS 스푸핑: 임계값 우회를 노린 선형 램프"""
    d = data.copy()
    n = len(d) - start
    ramp = np.minimum(np.arange(n, dtype=np.float32) * rate, max_mag)
    d[start:, 0] += ramp
    d[start:, 1] += ramp * 0.5
    return _recalc_residual(d)

def atk_gradual_alt(data, start, rate=0.5, max_mag=25.0):
    """점진적 Altitude 스푸핑: 고도 서서히 상승"""
    d = data.copy()
    n = len(d) - start
    ramp = np.minimum(np.arange(n, dtype=np.float32) * rate, max_mag)
    d[start:, 6] += ramp
    return d


# ─────────────────────────────────────────────────────────────
# 검출기 (슬라이딩 윈도우 VAE)
# ─────────────────────────────────────────────────────────────
class Detector:
    def __init__(self, vae: VAE, threshold: float, mean: np.ndarray, std: np.ndarray):
        self.vae  = vae
        self.th   = threshold
        self.mean = mean
        self.std  = std

    def score_series(self, data: np.ndarray) -> np.ndarray:
        scores = np.zeros(len(data))
        for i in range(WIN, len(data) + 1):
            w = data[i - WIN: i]
            normed = (w - self.mean) / self.std
            x = torch.FloatTensor(normed.reshape(1, -1))
            with torch.no_grad():
                recon, _, _ = self.vae(x)
                scores[i - 1] = torch.mean((recon - x) ** 2).item()
        return scores

    def per_feat_contrib(self, data: np.ndarray, idx: int) -> np.ndarray:
        """idx 위치 윈도우의 피처별 재구성 오차"""
        w = data[idx - WIN: idx]
        normed = (w - self.mean) / self.std
        x = torch.FloatTensor(normed.reshape(1, -1))
        with torch.no_grad():
            recon, _, _ = self.vae(x)
            per_feat = (recon - x) ** 2
            return per_feat.reshape(WIN, N_FEAT).mean(0).numpy()


# ─────────────────────────────────────────────────────────────
# 실험 실행
# ─────────────────────────────────────────────────────────────
def run_experiment(det: Detector, data: np.ndarray, attack_name: str,
                   atk_start: int, n_total: int) -> dict:
    """
    공격 시계열에 대해 슬라이딩 윈도우 탐지 수행 후 성능 지표 반환.
    레이블: 0~atk_start-1 = 정상, atk_start~n_total-1 = 공격
    """
    scores = det.score_series(data[:n_total])
    labels = np.zeros(n_total, dtype=int)
    labels[atk_start:] = 1

    preds = (scores > det.th).astype(int)

    # 경계 구간(초기 19개 윈도우) 제외하고 지표 계산
    boundary_end = atk_start + WIN - 1
    eval_mask = np.ones(n_total, dtype=bool)
    eval_mask[atk_start: boundary_end] = False  # 경계 구간 제외

    y_true = labels[eval_mask]
    y_pred = preds[eval_mask]

    TP = int(((y_pred == 1) & (y_true == 1)).sum())
    FP = int(((y_pred == 1) & (y_true == 0)).sum())
    TN = int(((y_pred == 0) & (y_true == 0)).sum())
    FN = int(((y_pred == 0) & (y_true == 1)).sum())

    prec   = TP / (TP + FP + 1e-9)
    recall = TP / (TP + FN + 1e-9)
    f1     = 2 * prec * recall / (prec + recall + 1e-9)

    # 탐지 지연: 공격 시작 이후 최초 탐지까지 타임스텝
    detect_idx = None
    for i in range(atk_start, n_total):
        if preds[i] == 1:
            detect_idx = i
            break
    delay_ts = (detect_idx - atk_start) if detect_idx is not None else -1
    delay_ms = delay_ts * 100 if delay_ts >= 0 else -1

    # FP/FN 윈도우 인덱스 수집 (전체 구간 기준)
    fp_windows = [i for i in range(WIN, n_total) if preds[i]==1 and labels[i]==0]
    fn_windows = [i for i in range(atk_start + WIN, n_total) if preds[i]==0 and labels[i]==1]

    return {
        'attack':    attack_name,
        'TP': TP, 'FP': FP, 'TN': TN, 'FN': FN,
        'Precision': round(prec,   4),
        'Recall':    round(recall, 4),
        'F1':        round(f1,     4),
        'delay_ts':  delay_ts,
        'delay_ms':  delay_ms,
        'fp_windows': fp_windows[:5],  # 최대 5개 저장
        'fn_windows': fn_windows[:5],
        'scores':    scores,
        'labels':    labels,
        'preds':     preds,
    }


# ─────────────────────────────────────────────────────────────
# FP/FN 피처 분포 분석
# ─────────────────────────────────────────────────────────────
def analyze_fpfn(det: Detector, data: np.ndarray,
                 fp_wins: list, fn_wins: list,
                 normal_data: np.ndarray) -> dict:
    """
    FP/FN 윈도우에서 피처별 재구성 오차를 추출,
    정상 윈도우 분포와 비교하여 겹침 정도 계산
    """
    # 정상 윈도우 재구성 오차 분포 (피처별)
    normal_contribs = []
    for i in range(WIN, min(500, len(normal_data))):
        c = det.per_feat_contrib(normal_data, i)
        normal_contribs.append(c)
    normal_contribs = np.array(normal_contribs)  # (N, 8)

    result = {'fp': [], 'fn': []}

    for tag, windows in [('fp', fp_wins), ('fn', fn_wins)]:
        for w_idx in windows:
            if w_idx < WIN or w_idx >= len(data):
                continue
            contrib = det.per_feat_contrib(data, w_idx)
            top_feat_idx = int(np.argmax(contrib))
            top_feat     = UAV_FEATS[top_feat_idx]

            # 해당 피처의 오차가 정상 분포 몇 sigma 위에 있는지
            mu  = normal_contribs[:, top_feat_idx].mean()
            sig = normal_contribs[:, top_feat_idx].std() + 1e-9
            z_score = (contrib[top_feat_idx] - mu) / sig

            result[tag].append({
                'window_idx':   w_idx,
                'top_feature':  top_feat,
                'top_contrib':  round(float(contrib[top_feat_idx]), 6),
                'normal_mean':  round(float(mu),  6),
                'normal_std':   round(float(sig), 6),
                'z_score':      round(float(z_score), 3),
                'all_contribs': {f: round(float(contrib[i]), 6)
                                 for i, f in enumerate(UAV_FEATS)},
            })
    return result


# ─────────────────────────────────────────────────────────────
# 시각화
# ─────────────────────────────────────────────────────────────
def plot_scores(results: list, threshold: float, atk_start: int, save_path: Path):
    n = len(results)
    fig, axes = plt.subplots(n, 1, figsize=(14, 3 * n), sharex=False)
    if n == 1:
        axes = [axes]

    colors = {'normal': '#4CAF50', 'attack': '#F44336', 'boundary': '#FF9800'}

    for ax, res in zip(axes, results):
        scores = res['scores']
        labels = res['labels']
        xs = np.arange(len(scores))

        # 정상/경계/공격 구간 음영
        ax.axvspan(0, atk_start, alpha=0.08, color=colors['normal'], label='정상 구간')
        ax.axvspan(atk_start, atk_start + WIN, alpha=0.15, color=colors['boundary'], label='경계 구간')
        ax.axvspan(atk_start + WIN, len(scores), alpha=0.08, color=colors['attack'], label='공격 구간')

        ax.plot(xs, scores, color='#1565C0', lw=1.2, label='Anomaly Score')
        ax.axhline(threshold, color='red', ls='--', lw=1.5, label=f'Threshold {threshold:.3f}')
        ax.set_title(f"{res['attack']}  |  P={res['Precision']:.3f}  R={res['Recall']:.3f}  "
                     f"F1={res['F1']:.3f}  delay={res['delay_ms']}ms", fontsize=10)
        ax.set_ylabel('MSE Score')
        ax.legend(fontsize=7, loc='upper left')

    axes[-1].set_xlabel('Timestep (100ms 간격)')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'[저장] {save_path}')


def plot_fpfn_feature_dist(fpfn: dict, normal_contribs: np.ndarray, save_path: Path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, (tag, label_ko) in zip(axes, [('fp', 'FP 윈도우'), ('fn', 'FN 윈도우')]):
        if not fpfn[tag]:
            ax.set_title(f'{label_ko}: 없음')
            continue
        # 각 FP/FN 윈도우의 all_contribs 막대
        sample = fpfn[tag][0]  # 대표 1개
        feats  = list(sample['all_contribs'].keys())
        vals   = list(sample['all_contribs'].values())
        norm_means = [normal_contribs[:, i].mean() for i in range(N_FEAT)]

        x = np.arange(N_FEAT)
        w = 0.35
        ax.bar(x - w/2, norm_means, w, label='정상 평균', color='#90CAF9')
        ax.bar(x + w/2, vals,       w, label=f'{label_ko}', color='#EF9A9A')
        ax.set_xticks(x)
        ax.set_xticklabels(feats, rotation=45, ha='right', fontsize=8)
        ax.set_ylabel('재구성 오차')
        ax.set_title(f'{label_ko} 피처별 재구성 오차 vs 정상 평균\n'
                     f'top feature: {sample["top_feature"]} (z={sample["z_score"]}σ)')
        ax.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'[저장] {save_path}')


# ─────────────────────────────────────────────────────────────
# 메인
# ─────────────────────────────────────────────────────────────
def main():
    print('=' * 60)
    print('해금 본선 추가 연구 — 복합/점진적 공격 탐지 실험')
    print('=' * 60)

    # 1. VAE 로드
    vae = VAE(input_dim=WIN * N_FEAT)
    weight_path = BASE / 'vae_uav.pth'
    vae.load_state_dict(torch.load(weight_path, map_location='cpu', weights_only=True))
    vae.eval()
    print(f'[OK] VAE 로드 완료: {weight_path}')

    # 2. 정상 데이터 1,000샘플 생성 (예선과 동일)
    normal = gen_normal(1000, seed=42)

    # 3. 정규화 파라미터 (정상 데이터 기준)
    mean = normal.mean(0)
    std  = normal.std(0) + 1e-8

    # 4. 임계값: 정상 검증셋 95th percentile
    cal_scores = []
    for i in range(WIN, len(normal)):
        w = normal[i - WIN: i]
        normed = (w - mean) / std
        x = torch.FloatTensor(normed.reshape(1, -1))
        with torch.no_grad():
            recon, _, _ = vae(x)
            cal_scores.append(torch.mean((recon - x) ** 2).item())
    threshold = float(np.percentile(cal_scores, 95))
    print(f'[OK] 임계값: {threshold:.4f}')

    det = Detector(vae, threshold, mean, std)

    # 5. 테스트 데이터: 정상 100 + 공격 200 (예선과 동일 비율)
    N_TOTAL = 300
    ATK_START = 100
    base_normal = gen_normal(N_TOTAL, seed=123)

    ATTACKS = [
        ('GPS_Spoofing (예선 기준)',      atk_gps_sudden(base_normal, ATK_START)),
        ('Altitude_Spoofing (예선 기준)', atk_altitude_sudden(base_normal, ATK_START)),
        ('Command_Injection (예선 기준)', atk_cmd_injection(base_normal, ATK_START)),
        ('복합_A: GPS+CmdInj (신규)',     atk_composite_gps_cmd(base_normal, ATK_START)),
        ('복합_B: GPS+Alt (신규)',         atk_composite_gps_alt(base_normal, ATK_START)),
        ('점진적_GPS rate=0.3 (신규)',    atk_gradual_gps(base_normal, ATK_START, rate=0.3)),
        ('점진적_GPS rate=0.1 (신규)',    atk_gradual_gps(base_normal, ATK_START, rate=0.1)),
        ('점진적_Alt rate=0.5 (신규)',    atk_gradual_alt(base_normal, ATK_START, rate=0.5)),
        ('점진적_Alt rate=0.2 (신규)',    atk_gradual_alt(base_normal, ATK_START, rate=0.2)),
    ]

    results = []
    for name, data in ATTACKS:
        res = run_experiment(det, data, name, ATK_START, N_TOTAL)
        results.append(res)
        print(f'  [{name}]')
        print(f'    P={res["Precision"]:.3f}  R={res["Recall"]:.3f}  '
              f'F1={res["F1"]:.3f}  delay={res["delay_ms"]}ms  '
              f'FP={res["FP"]}  FN={res["FN"]}')

    # 6. 그래프 저장
    plot_scores(results[:3], threshold, ATK_START,
                BASE / 'result_original_attacks.png')
    plot_scores(results[3:5], threshold, ATK_START,
                BASE / 'result_composite_attacks.png')
    plot_scores(results[5:], threshold, ATK_START,
                BASE / 'result_gradual_attacks.png')

    # 7. FP/FN 분석 (복합 A 기준)
    comp_a_data = atk_composite_gps_cmd(base_normal, ATK_START)
    comp_a_res  = results[3]
    fpfn = analyze_fpfn(det, comp_a_data,
                        comp_a_res['fp_windows'],
                        comp_a_res['fn_windows'],
                        normal)

    # 정상 재구성 오차 분포 (그래프용)
    normal_contribs = []
    for i in range(WIN, min(500, len(normal))):
        normal_contribs.append(det.per_feat_contrib(normal, i))
    normal_contribs = np.array(normal_contribs)

    plot_fpfn_feature_dist(fpfn, normal_contribs,
                           BASE / 'result_fpfn_analysis.png')

    # 8. JSON 결과 저장
    summary = []
    for r in results:
        summary.append({k: v for k, v in r.items()
                        if k not in ('scores', 'labels', 'preds')})
    summary_path = BASE / 'advanced_attack_results.json'
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump({'threshold': threshold, 'results': summary,
                   'fpfn_analysis': fpfn}, f, ensure_ascii=False, indent=2)
    print(f'[저장] {summary_path}')

    # 9. 콘솔 요약 표
    print('\n' + '=' * 60)
    print(f'{"공격 유형":<35} {"P":>6} {"R":>6} {"F1":>6} {"delay":>8}')
    print('-' * 60)
    for r in results:
        d = f'{r["delay_ms"]}ms' if r["delay_ms"] >= 0 else '미탐'
        print(f'{r["attack"]:<35} {r["Precision"]:>6.3f} {r["Recall"]:>6.3f} '
              f'{r["F1"]:>6.3f} {d:>8}')
    print('=' * 60)
    print('실험 완료.')


if __name__ == '__main__':
    main()
