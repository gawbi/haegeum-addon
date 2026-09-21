#!/usr/bin/env python3
"""
L1 이상탐지 VAE — CPU 실시간 추론 성능 벤치마크
(DAH 2026 해금팀 / haegeum-addon 추가 연구)

목적
  온보드(엣지) CPU 에서 VAE 이상탐지가 센서 발행 주기 안에 끝나는지 정량 검증하고,
  동적 양자화(qint8)로 경량화했을 때 지연시간·모델 크기·탐지 성능이 어떻게 변하는지 비교.

측정 대상
  - 기존 anomaly_detector.VAE 구조 + 기존 가중치(vae_uav.pth / vae_ugv.pth) 그대로 사용
  - 단일 윈도우(batch=1) 및 배치(8, 32) 추론 지연시간 p50/p95/p99
  - 처리량 (windows/sec)
  - torch.ao.quantization.quantize_dynamic(qint8) 적용 전/후 비교
  - 양자화 후 재구성 오차 분포 및 기존 임계값 기준 탐지 성능(F1) 변화

실행:  python benchmarks/latency_benchmark.py
출력:  benchmarks/results.json, docs/images/latency_distribution.png
"""

import argparse
import json
import os
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from utils.seed import set_seed, env_info          # noqa: E402
from utils.load_model import load_vae              # noqa: E402
from utils import ugv_data                          # noqa: E402

import matplotlib                                   # noqa: E402
matplotlib.use('Agg')
import matplotlib.pyplot as plt                     # noqa: E402

WIN = 20
SEED = 42

# 라이트/다크 어느 쪽 README 테마에서도 읽히도록 밝은 중립 배경 + 고채도 2색
C_FP32 = '#2a78d6'
C_INT8 = '#eb6834'
C_SURFACE = '#fcfcfb'
C_TEXT = '#0b0b0b'
C_MUTED = '#52514e'


# ─────────────────────────────────────────────────────────────
# 양자화
# ─────────────────────────────────────────────────────────────
def select_qengine() -> str:
    """사용 가능한 양자화 백엔드 선택 (arm64 → qnnpack, x86 → fbgemm)."""
    engines = [e for e in torch.backends.quantized.supported_engines
               if e != 'none']
    for pref in ('qnnpack', 'fbgemm', 'x86'):
        if pref in engines:
            torch.backends.quantized.engine = pref
            return pref
    if engines:
        torch.backends.quantized.engine = engines[0]
        return engines[0]
    raise RuntimeError('사용 가능한 양자화 엔진이 없습니다')


def quantize(model: nn.Module) -> nn.Module:
    """Linear 레이어 동적 양자화 (qint8)."""
    select_qengine()
    try:
        q = torch.ao.quantization.quantize_dynamic
    except AttributeError:                      # 구버전 torch 호환
        q = torch.quantization.quantize_dynamic
    qmodel = q(model, {nn.Linear}, dtype=torch.qint8)
    qmodel.eval()
    return qmodel


def state_dict_bytes(model: nn.Module) -> int:
    with tempfile.NamedTemporaryFile(suffix='.pth', delete=False) as f:
        tmp = f.name
    try:
        torch.save(model.state_dict(), tmp)
        return os.path.getsize(tmp)
    finally:
        os.unlink(tmp)


# ─────────────────────────────────────────────────────────────
# 지연시간 측정
# ─────────────────────────────────────────────────────────────
def measure_latency(model: nn.Module, input_dim: int, batch: int,
                    iters: int, warmup: int, seed: int = SEED) -> dict:
    """
    anomaly_detector._infer 와 동일한 경로(VAE forward + MSE)를 1회 추론으로 계산.
    반복마다 perf_counter 로 개별 측정 → 분위수 산출.
    """
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(batch, input_dim, generator=g)

    with torch.no_grad():
        for _ in range(warmup):
            recon, _, _ = model(x)
            _ = torch.mean((recon - x) ** 2).item()

        samples = np.empty(iters, dtype=np.float64)
        for i in range(iters):
            t0 = time.perf_counter()
            recon, _, _ = model(x)
            _ = torch.mean((recon - x) ** 2).item()
            samples[i] = (time.perf_counter() - t0) * 1000.0  # ms

    mean_ms = float(samples.mean())
    return {
        'batch': batch,
        'iters': iters,
        'warmup': warmup,
        'mean_ms': round(mean_ms, 5),
        'p50_ms': round(float(np.percentile(samples, 50)), 5),
        'p95_ms': round(float(np.percentile(samples, 95)), 5),
        'p99_ms': round(float(np.percentile(samples, 99)), 5),
        'min_ms': round(float(samples.min()), 5),
        'max_ms': round(float(samples.max()), 5),
        'std_ms': round(float(samples.std()), 5),
        'throughput_windows_per_sec': round(batch / (mean_ms / 1000.0), 1),
        '_samples': samples,
    }


# ─────────────────────────────────────────────────────────────
# 탐지 성능 (양자화 전/후)
# ─────────────────────────────────────────────────────────────
def score_windows(model: nn.Module, X: np.ndarray, seed: int = SEED) -> np.ndarray:
    """윈도우 배열 → 윈도우별 재구성 오차(MSE). VAE 샘플링 재현을 위해 시드 고정."""
    torch.manual_seed(seed)
    out = np.empty(len(X), dtype=np.float64)
    with torch.no_grad():
        for i, row in enumerate(X):
            x = torch.from_numpy(row).unsqueeze(0)
            recon, _, _ = model(x)
            out[i] = torch.mean((recon - x) ** 2).item()
    return out


def prf1(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    TP = int(((y_pred == 1) & (y_true == 1)).sum())
    FP = int(((y_pred == 1) & (y_true == 0)).sum())
    TN = int(((y_pred == 0) & (y_true == 0)).sum())
    FN = int(((y_pred == 0) & (y_true == 1)).sum())
    prec = TP / (TP + FP + 1e-9)
    rec = TP / (TP + FN + 1e-9)
    f1 = 2 * prec * rec / (prec + rec + 1e-9)
    return {'TP': TP, 'FP': FP, 'TN': TN, 'FN': FN,
            'precision': round(prec, 4), 'recall': round(rec, 4),
            'f1': round(f1, 4)}


def uav_detection_eval(vae, qvae) -> dict:
    """
    advanced_attacks.py 의 기존 평가 하네스(gen_normal / atk_* / 경계 윈도우 제외)를
    그대로 재사용해 FP32 와 INT8 의 탐지 성능을 비교.
    임계값은 FP32 정상 데이터 95th percentile 로 한 번 정하고 양쪽에 동일 적용.
    """
    import advanced_attacks as AA

    normal = AA.gen_normal(1000, seed=42)
    mean, std = normal.mean(0), normal.std(0) + 1e-8

    # 임계값 보정 (FP32 기준) — advanced_attacks.main() 과 동일 절차
    torch.manual_seed(SEED)
    cal = []
    with torch.no_grad():
        for i in range(WIN, len(normal)):
            w = (normal[i - WIN:i] - mean) / std
            x = torch.FloatTensor(w.reshape(1, -1))
            recon, _, _ = vae(x)
            cal.append(torch.mean((recon - x) ** 2).item())
    threshold = float(np.percentile(cal, 95))

    N_TOTAL, ATK_START = 300, 100
    base = AA.gen_normal(N_TOTAL, seed=123)
    attacks = [
        ('GPS_Spoofing', AA.atk_gps_sudden(base, ATK_START)),
        ('Altitude_Spoofing', AA.atk_altitude_sudden(base, ATK_START)),
        ('Command_Injection', AA.atk_cmd_injection(base, ATK_START)),
        ('Composite_GPS+CmdInj', AA.atk_composite_gps_cmd(base, ATK_START)),
        ('Gradual_GPS_rate0.3', AA.atk_gradual_gps(base, ATK_START, rate=0.3)),
    ]

    out = {'threshold_fp32_p95': round(threshold, 4), 'attacks': {}}
    for tag, model in (('fp32', vae), ('int8', qvae)):
        det = AA.Detector(model, threshold, mean, std)
        for name, data in attacks:
            torch.manual_seed(SEED)
            res = AA.run_experiment(det, data, name, ATK_START, N_TOTAL)
            out['attacks'].setdefault(name, {})[tag] = {
                'precision': res['Precision'], 'recall': res['Recall'],
                'f1': res['F1'], 'FP': res['FP'], 'FN': res['FN'],
                'delay_ms': res['delay_ms'],
            }

    # 정상 데이터 재구성 오차 분포 비교
    torch.manual_seed(SEED)
    cal_q = []
    with torch.no_grad():
        for i in range(WIN, len(normal)):
            w = (normal[i - WIN:i] - mean) / std
            x = torch.FloatTensor(w.reshape(1, -1))
            recon, _, _ = qvae(x)
            cal_q.append(torch.mean((recon - x) ** 2).item())
    cal, cal_q = np.array(cal), np.array(cal_q)
    out['normal_score_distribution'] = {
        'fp32': {'mean': round(float(cal.mean()), 4),
                 'p95': round(float(np.percentile(cal, 95)), 4),
                 'p99': round(float(np.percentile(cal, 99)), 4)},
        'int8': {'mean': round(float(cal_q.mean()), 4),
                 'p95': round(float(np.percentile(cal_q, 95)), 4),
                 'p99': round(float(np.percentile(cal_q, 99)), 4)},
        'pearson_r': round(float(np.corrcoef(cal, cal_q)[0, 1]), 4),
        'fp_rate_int8_at_fp32_threshold': round(float((cal_q > threshold).mean()), 4),
    }
    return out


def ugv_detection_eval(vae, qvae) -> dict:
    """
    UGV: 노트북과 동일한 합성 데이터(utils/ugv_data)로 공격 3종 평가.
    경계 윈도우(공격 시작과 겹치는 윈도우)는 제외.
    """
    normal = ugv_data.generate_normal(1000, seed=42)
    mean, std = normal.mean(0), normal.std(0) + 1e-8

    Xn = ugv_data.sliding_windows(normal, mean, std)
    split = int(len(Xn) * 0.8)
    Xn_val = Xn[split:]

    s_val = score_windows(vae, Xn_val)
    threshold = float(np.percentile(s_val, 95))

    n = 1000
    attacks = {
        'GPS_Spoofing': ugv_data.generate_gps_spoofing(n),
        'Wheel_Slip': ugv_data.generate_wheel_slip(n),
        'Command_Anomaly': ugv_data.generate_command_anomaly(n),
    }
    regions = ugv_data.attack_regions(n)

    out = {'threshold_fp32_p95': round(threshold, 4),
           'note': ('공격 주입 구간에 완전히 포함된 윈도우만 양성, 무관한 윈도우만 '
                    '음성으로 라벨링하고 경계 윈도우는 평가에서 제외'),
           'attacks': {}}
    for name, data in attacks.items():
        X = ugv_data.sliding_windows(data, mean, std)
        label = ugv_data.window_labels(len(data), regions[name])
        keep = label >= 0
        for tag, model in (('fp32', vae), ('int8', qvae)):
            s = score_windows(model, X)
            pred = (s > threshold).astype(int)
            m = prf1(label[keep], pred[keep])
            out['attacks'].setdefault(name, {})[tag] = m

    s_val_q = score_windows(qvae, Xn_val)
    out['normal_score_distribution'] = {
        'fp32': {'mean': round(float(s_val.mean()), 4),
                 'p95': round(float(np.percentile(s_val, 95)), 4),
                 'p99': round(float(np.percentile(s_val, 99)), 4)},
        'int8': {'mean': round(float(s_val_q.mean()), 4),
                 'p95': round(float(np.percentile(s_val_q, 95)), 4),
                 'p99': round(float(np.percentile(s_val_q, 99)), 4)},
        'pearson_r': round(float(np.corrcoef(s_val, s_val_q)[0, 1]), 4),
        'fp_rate_int8_at_fp32_threshold': round(float((s_val_q > threshold).mean()), 4),
    }
    return out


# ─────────────────────────────────────────────────────────────
# 플롯
# ─────────────────────────────────────────────────────────────
def plot_latency(samples: dict, out_path: Path, sensor_period_ms: float):
    """batch=1 지연시간 분포 (UAV/UGV × FP32/INT8)."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6), facecolor=C_SURFACE)

    for ax, platform in zip(axes, ['uav', 'ugv']):
        ax.set_facecolor(C_SURFACE)
        lo = min(samples[(platform, t)].min() for t in ('fp32', 'int8'))
        hi = max(np.percentile(samples[(platform, t)], 99.5)
                 for t in ('fp32', 'int8'))
        bins = np.logspace(np.log10(lo * 0.9), np.log10(hi * 1.3), 70)
        for tag, color in (('fp32', C_FP32), ('int8', C_INT8)):
            s = samples[(platform, tag)]
            ax.hist(s, bins=bins, color=color, alpha=0.6,
                    label=f'{tag.upper()}  p50={np.percentile(s,50):.3f} ms  '
                          f'p99={np.percentile(s,99):.3f} ms')
            ax.axvline(np.percentile(s, 99), color=color, lw=2, ls='--')
        ax.set_xscale('log')
        ax.set_title(f'{platform.upper()} VAE — single-window inference latency '
                     f'(batch=1, CPU)', color=C_TEXT, fontsize=11)
        ax.set_xlabel('latency (ms, log scale) — dashed = p99', color=C_MUTED)
        ax.set_ylabel('count', color=C_MUTED)
        ax.tick_params(colors=C_MUTED)
        for spine in ('top', 'right'):
            ax.spines[spine].set_visible(False)
        for spine in ('left', 'bottom'):
            ax.spines[spine].set_color(C_MUTED)
        ax.grid(axis='y', color='#d8d8d4', lw=0.6)
        ax.set_axisbelow(True)
        ax.legend(fontsize=8.5, frameon=False, labelcolor=C_TEXT)

    fig.suptitle(f'VAE anomaly detection latency vs sensor period '
                 f'({sensor_period_ms:.0f} ms @ sensor_publisher default rate)',
                 color=C_TEXT, fontsize=12.5, fontweight='bold')
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, facecolor=C_SURFACE)
    plt.close(fig)


def plot_batch_throughput(latency: dict, out_path: Path):
    """배치 크기별 처리량 비교 (FP32 vs INT8)."""
    batches = [1, 8, 32]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), facecolor=C_SURFACE)
    width = 0.36
    xs = np.arange(len(batches))

    for ax, platform in zip(axes, ['uav', 'ugv']):
        ax.set_facecolor(C_SURFACE)
        for k, (tag, color) in enumerate((('fp32', C_FP32), ('int8', C_INT8))):
            vals = [latency[platform][tag][str(b)]['throughput_windows_per_sec']
                    for b in batches]
            bars = ax.bar(xs + (k - 0.5) * width, vals, width * 0.94,
                          color=color, label=tag.upper())
            for b, v in zip(bars, vals):
                ax.text(b.get_x() + b.get_width() / 2, v, f'{v:,.0f}',
                        ha='center', va='bottom', fontsize=8, color=C_TEXT)
        ax.set_xticks(xs)
        ax.set_xticklabels([f'batch={b}' for b in batches], color=C_MUTED)
        ax.set_title(f'{platform.upper()} throughput (windows/sec, CPU)',
                     color=C_TEXT, fontsize=11)
        ax.set_ylabel('windows / sec', color=C_MUTED)
        ax.tick_params(colors=C_MUTED)
        for spine in ('top', 'right'):
            ax.spines[spine].set_visible(False)
        for spine in ('left', 'bottom'):
            ax.spines[spine].set_color(C_MUTED)
        ax.grid(axis='y', color='#d8d8d4', lw=0.6)
        ax.set_axisbelow(True)
        ax.margins(y=0.16)
        ax.legend(fontsize=9, frameon=False, labelcolor=C_TEXT)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, facecolor=C_SURFACE)
    plt.close(fig)


# ─────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--iters', type=int, default=1000)
    ap.add_argument('--warmup', type=int, default=200)
    ap.add_argument('--threads', type=int, default=1,
                    help='기준 측정 스레드 수 (온보드 단일 추론 스레드 가정)')
    args = ap.parse_args()

    set_seed(SEED)
    torch.set_num_threads(args.threads)

    out_dir = Path(__file__).resolve().parent
    img_dir = REPO_ROOT / 'docs' / 'images'
    img_dir.mkdir(parents=True, exist_ok=True)

    results = {
        'meta': {
            'generated_at': time.strftime('%Y-%m-%dT%H:%M:%S%z'),
            'environment': env_info(),
            'torch_num_threads_used': args.threads,
            'iters': args.iters,
            'warmup': args.warmup,
            'seed': SEED,
            'quantization_engine': None,
            'note': '모든 수치는 본 스크립트 실행으로 실측한 값 (추정치 없음)',
        },
        'sensor': {},
        'model': {},
        'latency': {},
        'thread_scaling': {},
        'detection': {},
    }
    results['meta']['environment']['torch_num_threads'] = args.threads

    # 센서 발행 주기 (sensor_publisher.py 기본값)
    rate_hz = 10.0
    results['sensor'] = {
        'source': 'sensor_publisher.py (declare_parameter rate_hz 기본값)',
        'rate_hz': rate_hz,
        'period_ms': 1000.0 / rate_hz,
        'window_size': WIN,
    }

    samples_for_plot = {}
    print('=' * 68)
    print('VAE 실시간 추론 벤치마크 (CPU)')
    print('=' * 68)
    for k, v in results['meta']['environment'].items():
        print(f'  {k}: {v}')
    print()

    for platform in ('uav', 'ugv'):
        vae, cfg = load_vae(platform, WIN)
        qvae = quantize(vae)
        results['meta']['quantization_engine'] = torch.backends.quantized.engine
        input_dim = WIN * cfg['n_feat']

        fp32_bytes = state_dict_bytes(vae)
        int8_bytes = state_dict_bytes(qvae)
        results['model'][platform] = {
            'weight_file': cfg['weight_file'],
            'input_dim': input_dim,
            'n_features': cfg['n_feat'],
            'params': int(sum(p.numel() for p in vae.parameters())),
            'state_dict_bytes_fp32': fp32_bytes,
            'state_dict_bytes_int8': int8_bytes,
            'size_reduction_pct': round((1 - int8_bytes / fp32_bytes) * 100, 1),
        }

        results['latency'][platform] = {'fp32': {}, 'int8': {}}
        for tag, model in (('fp32', vae), ('int8', qvae)):
            for batch in (1, 8, 32):
                r = measure_latency(model, input_dim, batch,
                                    args.iters, args.warmup)
                s = r.pop('_samples')
                if batch == 1:
                    samples_for_plot[(platform, tag)] = s
                results['latency'][platform][tag][str(batch)] = r
                print(f'  [{platform}/{tag}/batch={batch:2d}] '
                      f'p50={r["p50_ms"]:.4f} p95={r["p95_ms"]:.4f} '
                      f'p99={r["p99_ms"]:.4f} ms | '
                      f'{r["throughput_windows_per_sec"]:,.0f} win/s')

        # 스레드 수 스케일링 (batch=1)
        results['thread_scaling'][platform] = {}
        for nth in (1, 2, 4):
            if nth > (os.cpu_count() or 1):
                continue
            torch.set_num_threads(nth)
            row = {}
            for tag, model in (('fp32', vae), ('int8', qvae)):
                r = measure_latency(model, input_dim, 1,
                                    max(300, args.iters // 3), 100)
                r.pop('_samples')
                row[tag] = {'p50_ms': r['p50_ms'], 'p99_ms': r['p99_ms'],
                            'throughput_windows_per_sec':
                                r['throughput_windows_per_sec']}
            results['thread_scaling'][platform][str(nth)] = row
        torch.set_num_threads(args.threads)

        # 탐지 성능 비교
        print(f'  [{platform}] 양자화 전/후 탐지 성능 평가 중...')
        if platform == 'uav':
            results['detection']['uav'] = uav_detection_eval(vae, qvae)
        else:
            results['detection']['ugv'] = ugv_detection_eval(vae, qvae)

    # 실시간성 판정
    period_ms = results['sensor']['period_ms']
    rt = {}
    for platform in ('uav', 'ugv'):
        p99 = results['latency'][platform]['fp32']['1']['p99_ms']
        p99q = results['latency'][platform]['int8']['1']['p99_ms']
        rt[platform] = {
            'sensor_period_ms': period_ms,
            'p99_ms_fp32': p99,
            'p99_ms_int8': p99q,
            'budget_used_pct_fp32': round(p99 / period_ms * 100, 3),
            'budget_used_pct_int8': round(p99q / period_ms * 100, 3),
            'headroom_x_fp32': round(period_ms / p99, 1),
            'max_sustainable_rate_hz_fp32': round(1000.0 / p99, 1),
            'realtime_ok_fp32': bool(p99 < period_ms),
        }
    results['realtime_assessment'] = rt

    plot_latency(samples_for_plot, img_dir / 'latency_distribution.png', period_ms)
    plot_batch_throughput(results['latency'],
                          img_dir / 'latency_batch_throughput.png')

    with open(out_dir / 'results.json', 'w') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print()
    print(f'[OK] {out_dir / "results.json"}')
    print(f'[OK] {img_dir / "latency_distribution.png"}')
    print(f'[OK] {img_dir / "latency_batch_throughput.png"}')
    for platform, v in rt.items():
        print(f'  {platform}: p99={v["p99_ms_fp32"]:.4f} ms / '
              f'센서주기 {period_ms:.0f} ms → 예산 {v["budget_used_pct_fp32"]:.2f}% 사용, '
              f'여유 {v["headroom_x_fp32"]}배')


if __name__ == '__main__':
    main()
