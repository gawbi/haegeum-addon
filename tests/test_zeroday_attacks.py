#!/usr/bin/env python3
"""
제로데이 공격 주입 함수(zeroday_attacks.py)의 계약 검증.

예선 공격(advanced_attacks)은 모두 "정상값 + 편향" 이었지만, 제로데이 공격은
센서 스트림의 **시간 구조**(동결·재생·무상한 누적)를 바꾼다. 따라서 여기서는
값의 크기가 아니라 다음을 확인한다.
  - Freeze: start 이후 GPS 열이 start-1 값과 정확히 같은가
  - Replay: start 이후 GPS 열이 lag 스텝 전 원본과 정확히 같은가
  - Drift: 편향이 상한 없이 선형으로 자라는가 (예선 램프와의 차이)
  - 공통: residual = gps - imu 정합성, 공격 전 구간 불변, 입력 불변, 결정론
"""

import numpy as np
import pytest

import advanced_attacks as AA
import zeroday_attacks as Z

START = 100
N = 300
GPS_COLS = (0, 1)
IMU_COLS = (2, 3)
RESIDUAL_COLS = (4, 5)
FLOAT32_NOISE = 1e-5   # residual 재계산 시 float32 최소 단위 흔들림 허용


@pytest.fixture
def base():
    return AA.gen_normal(N, seed=123)


def changed_columns(before, after, atol=FLOAT32_NOISE):
    delta = np.abs(after - before) > atol
    return set(np.where(delta.any(axis=0))[0].tolist())


ALL_ATTACKS = [
    Z.atk_gps_freeze,
    Z.atk_gps_replay,
    Z.atk_gps_drift,
    Z.atk_sensor_freeze_all,
]


# ─────────────────────────────────────────────────────────────
# 1. GPS Freeze
# ─────────────────────────────────────────────────────────────
def test_gps_freeze_holds_last_value_after_start(base):
    out = Z.atk_gps_freeze(base, START)
    frozen = base[START - 1, GPS_COLS]
    for col, val in zip(GPS_COLS, frozen):
        assert np.all(out[START:, col] == val)      # 비트 단위로 동일
        assert np.ptp(out[START:, col]) == 0.0          # 범위(max-min)가 정확히 0
    # IMU 는 그대로 — 그래서 잔차가 튄다
    assert np.array_equal(out[:, IMU_COLS], base[:, IMU_COLS])
    assert changed_columns(base, out) == set(GPS_COLS) | set(RESIDUAL_COLS)


def test_gps_freeze_makes_residual_jump(base):
    """동결 자체는 정상 범위 값이지만, 잔차는 IMU 변동만큼 커져야 한다."""
    out = Z.atk_gps_freeze(base, START)
    normal_resid_sd = base[:START, 4].std()           # ≈ 0.05
    attack_resid_sd = out[START:, 4].std()
    assert attack_resid_sd > 5 * normal_resid_sd


# ─────────────────────────────────────────────────────────────
# 2. GPS Replay
# ─────────────────────────────────────────────────────────────
def test_gps_replay_equals_lagged_original_values(base):
    lag = 100
    out = Z.atk_gps_replay(base, START, lag=lag)
    for col in GPS_COLS:
        np.testing.assert_array_equal(out[START:, col], base[START - lag: N - lag, col])
    assert changed_columns(base, out) == set(GPS_COLS) | set(RESIDUAL_COLS)


def test_gps_replay_wraps_when_lag_exceeds_start(base):
    """lag > start 여도 원본 인덱스가 공격 전 구간 안에 머물러야 한다."""
    lag = 150
    out = Z.atk_gps_replay(base, START, lag=lag)
    t = np.arange(START, N)
    src = t - lag
    src[src < 0] = np.mod(src[src < 0], START)
    assert src.min() >= 0 and src.max() < N
    for col in GPS_COLS:
        np.testing.assert_array_equal(out[START:, col], base[src, col])


def test_gps_replay_values_stay_in_normal_range(base):
    """재생 값은 전부 실제 정상 샘플 — 피처 분포만 보면 정상과 구분 불가."""
    out = Z.atk_gps_replay(base, START)
    for col in GPS_COLS:
        assert out[START:, col].min() >= base[:, col].min()
        assert out[START:, col].max() <= base[:, col].max()


# ─────────────────────────────────────────────────────────────
# 3. GPS Drift
# ─────────────────────────────────────────────────────────────
@pytest.mark.parametrize('rate', [0.02, 0.005])
def test_gps_drift_bias_grows_linearly_without_cap(base, rate):
    out = Z.atk_gps_drift(base, START, rate_per_step=rate)
    delta = out[START:, 0] - base[START:, 0]
    expected = np.arange(N - START, dtype=np.float32) * rate
    np.testing.assert_allclose(delta, expected, atol=FLOAT32_NOISE)
    # 상한 없음: 마지막 편향이 예선 램프의 중간값보다 작아도 계속 커져야 한다
    assert delta[-1] == pytest.approx(rate * (N - START - 1), rel=1e-5)
    assert np.all(np.diff(delta) > 0)
    # y 축은 x 축의 절반
    dy = out[START:, 1] - base[START:, 1]
    np.testing.assert_allclose(dy, delta * 0.5, atol=FLOAT32_NOISE)


def test_gps_drift_is_far_weaker_than_preliminary_gradual(base):
    """제로데이 드리프트는 예선 점진 공격(rate=0.1, cap 12)보다 훨씬 약해야 한다."""
    drift = Z.atk_gps_drift(base, START, rate_per_step=0.02)
    gradual = AA.atk_gradual_gps(base, START, rate=0.1)
    d_drift = np.abs(drift[-1, 0] - base[-1, 0])      # 0.02*199 = 3.98
    d_grad = np.abs(gradual[-1, 0] - base[-1, 0])     # cap 12
    assert d_drift < d_grad / 2
    # 300 스텝 내내 정상 GPS 표준편차의 몇 배 안에 머문다 (최대 3.98 / 0.59 ≈ 6.7σ)
    assert d_drift < 10 * base[:, 0].std()


def test_gps_drift_respects_optional_cap(base):
    out = Z.atk_gps_drift(base, START, rate_per_step=0.02, max_mag=1.0)
    delta = out[START:, 0] - base[START:, 0]
    assert delta.max() == pytest.approx(1.0, rel=1e-5)


# ─────────────────────────────────────────────────────────────
# 4. Sensor-bus Freeze (보너스)
# ─────────────────────────────────────────────────────────────
def test_sensor_freeze_all_holds_every_column(base):
    out = Z.atk_sensor_freeze_all(base, START)
    ref = out[START - 1]
    for col in range(AA.N_FEAT):
        np.testing.assert_allclose(out[START:, col], ref[col], atol=FLOAT32_NOISE, rtol=0)
        assert out[START:, col].std() < FLOAT32_NOISE
    # 동결 행은 원본 start-1 행과 같다 (residual 재계산 노이즈만 허용)
    np.testing.assert_allclose(ref, base[START - 1], atol=FLOAT32_NOISE, rtol=0)


# ─────────────────────────────────────────────────────────────
# 공통 계약
# ─────────────────────────────────────────────────────────────
@pytest.mark.parametrize('fn', ALL_ATTACKS)
def test_residual_identity_holds_after_attack(base, fn):
    out = fn(base, START)
    np.testing.assert_allclose(out[:, 4], out[:, 0] - out[:, 2], atol=FLOAT32_NOISE)
    np.testing.assert_allclose(out[:, 5], out[:, 1] - out[:, 3], atol=FLOAT32_NOISE)


@pytest.mark.parametrize('fn', ALL_ATTACKS)
def test_prefix_before_start_is_untouched(base, fn):
    out = fn(base, START)
    np.testing.assert_allclose(out[:START], base[:START], atol=FLOAT32_NOISE, rtol=0)


@pytest.mark.parametrize('fn', ALL_ATTACKS)
def test_attack_does_not_mutate_input(base, fn):
    snapshot = base.copy()
    fn(base, START)
    np.testing.assert_array_equal(base, snapshot)


@pytest.mark.parametrize('fn', ALL_ATTACKS)
def test_attack_is_deterministic(base, fn):
    a = fn(base, START)
    b = fn(AA.gen_normal(N, seed=123), START)
    np.testing.assert_array_equal(a, b)
    assert a.dtype == np.float32 and a.shape == base.shape


def test_invalid_arguments_rejected(base):
    with pytest.raises(ValueError):
        Z.atk_gps_freeze(base, 0)
    with pytest.raises(ValueError):
        Z.atk_gps_replay(base, START, lag=0)
    with pytest.raises(ValueError):
        Z.atk_sensor_freeze_all(base, 0)
