#!/usr/bin/env python3
"""
UGV 합성 센서 데이터 생성기.

`UGV_anomaly_vae.ipynb` (vae_ugv.pth 를 학습시킨 노트북)의 generate_* 함수와
동일한 분포·동일한 공격 주입 방식을 옮긴 것입니다. 두 가지만 다릅니다.

  1) 재현성을 위해 전역 np.random 대신 seed 를 받는 Generator 사용.
  2) 정상/공격 시계열 길이를 n=1000 으로 통일. 노트북은 정상 n=1000, 공격 n=300
     을 섞어 썼는데, generate_normal 이 t=linspace(0,10,n) 의 gradient 로 속도를
     만들기 때문에 n 이 다르면 속도 스케일 자체가 달라져 정상 기준 정규화
     (mean/std)가 어긋납니다. 이 상태로 평가하면 공격이 아닌 구간도 전부 이상으로
     잡혀 오탐이 과대 계상됩니다. (GPS 스푸핑 주입 길이도 동일 비율 n/20 로 맞춤)

피처 순서 (anomaly_detector.PLATFORM_CONFIG['ugv'] 와 동일):
  0 gps_vel_x  1 gps_vel_y  2 imu_ax     3 imu_ay
  4 residual_x 5 residual_y 6 wheel_vel_l 7 wheel_vel_r
  8 cmd_vel_x  9 cmd_vel_w
"""

import numpy as np

FEATURE_NAMES = ['gps_vel_x', 'gps_vel_y',
                 'imu_ax', 'imu_ay',
                 'residual_x', 'residual_y',
                 'wheel_vel_l', 'wheel_vel_r',
                 'cmd_vel_x', 'cmd_vel_w']
N_FEAT = 10
WIN = 20


def attack_start(n: int) -> int:
    """generate_* 공통 공격 개시 인덱스 (노트북과 동일하게 n // 3)."""
    return n // 3


def generate_normal(n: int = 1000, seed: int = 42) -> np.ndarray:
    """정상 UGV 직선 주행."""
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 10, n)
    gps_x = t * 0.5 + rng.normal(0, 0.01, n)
    gps_y = np.sin(t * 0.1) * 0.3 + rng.normal(0, 0.01, n)
    gps_vel_x = np.gradient(gps_x)
    gps_vel_y = np.gradient(gps_y)
    imu_ax = gps_vel_x + rng.normal(0, 0.005, n)
    imu_ay = gps_vel_y + rng.normal(0, 0.005, n)
    residual_x = gps_vel_x - imu_ax
    residual_y = gps_vel_y - imu_ay
    wheel_l = 0.5 + rng.normal(0, 0.01, n)
    wheel_r = 0.5 + rng.normal(0, 0.01, n)
    cmd_vel_x = 0.5 + rng.normal(0, 0.01, n)
    cmd_vel_w = 0.0 + rng.normal(0, 0.005, n)
    return np.stack([gps_vel_x, gps_vel_y, imu_ax, imu_ay,
                     residual_x, residual_y,
                     wheel_l, wheel_r, cmd_vel_x, cmd_vel_w],
                    axis=1).astype(np.float32)


def generate_gps_spoofing(n: int = 1000, seed: int = 7) -> np.ndarray:
    """GPS Spoofing: gps_vel 순간 폭발, IMU 는 정상 → residual 폭발."""
    rng = np.random.default_rng(seed)
    data = generate_normal(n, seed=seed)
    jump = attack_start(n)
    span = max(15, n // 20)
    data[jump:jump + span, 0] += 8.0
    data[jump:jump + span, 1] += 4.0
    data[jump:jump + span, 4] += 8.0
    data[jump:jump + span, 5] += 4.0
    _ = rng  # 노트북과 동일하게 추가 난수 사용 없음
    return data


def generate_wheel_slip(n: int = 1000, seed: int = 7) -> np.ndarray:
    """Wheel Slip: 바퀴는 급회전하지만 IMU ≈ 0 (실제 이동 없음)."""
    rng = np.random.default_rng(seed + 1)
    data = generate_normal(n, seed=seed)
    start = attack_start(n)
    data[start:, 6] += rng.normal(0.8, 0.1, n - start)
    data[start:, 7] += rng.normal(0.8, 0.1, n - start)
    data[start:, 2] = rng.normal(0, 0.005, n - start)
    data[start:, 3] = rng.normal(0, 0.005, n - start)
    return data


def generate_command_anomaly(n: int = 1000, seed: int = 7) -> np.ndarray:
    """Command Anomaly: cmd_vel 은 정상인데 바퀴가 무반응."""
    rng = np.random.default_rng(seed + 2)
    data = generate_normal(n, seed=seed)
    start = attack_start(n)
    data[start:, 8] = 0.5 + rng.normal(0, 0.01, n - start)
    data[start:, 6] = rng.normal(0, 0.02, n - start)
    data[start:, 7] = rng.normal(0, 0.02, n - start)
    return data


def attack_regions(n: int = 1000) -> dict:
    """공격이 실제로 주입된 샘플 구간 [lo, hi) — 라벨링용."""
    start = attack_start(n)
    span = max(15, n // 20)
    return {
        'GPS_Spoofing': (start, start + span),   # 순간 버스트
        'Wheel_Slip': (start, n),                # 이후 지속
        'Command_Anomaly': (start, n),           # 이후 지속
    }


def window_labels(n: int, region: tuple, win: int = WIN) -> np.ndarray:
    """
    슬라이딩 윈도우 라벨: 1=공격 구간에 완전히 포함, 0=공격 구간과 무관,
    -1=경계(공격 시작/종료와 걸침) → 평가에서 제외.
    """
    lo, hi = region
    idx = np.arange(n - win + 1)
    label = np.full(len(idx), -1)
    label[(idx >= lo) & (idx + win <= hi)] = 1
    label[(idx + win <= lo) | (idx >= hi)] = 0
    return label


def sliding_windows(data: np.ndarray, mean, std, win: int = WIN) -> np.ndarray:
    """정규화 후 (N, win*N_FEAT) 평탄화 윈도우 생성."""
    normed = (data - mean) / std
    idx = np.arange(len(normed) - win + 1)
    out = np.stack([normed[i:i + win] for i in idx])
    return out.reshape(len(idx), win * data.shape[1]).astype(np.float32)
