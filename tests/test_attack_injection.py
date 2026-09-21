#!/usr/bin/env python3
"""
공격 주입 함수가 **의도한 피처만** 변형하는지.

이 계약이 깨지면 XAI/SHAP 분석의 "주입 열 ↔ 지목 피처 일치" 결론이 통째로
무의미해집니다(설명 대상 데이터가 실제 주입과 다르므로). 또한 주입 함수가
입력 배열을 제자리에서 바꿔 버리면, 같은 base 로 만든 다른 공격 시계열이
서로 오염되어 README 의 공격별 비교표가 성립하지 않습니다.
"""

import numpy as np
import pytest

import advanced_attacks as AA
from utils import ugv_data

WIN = AA.WIN
START = 100
N = 300

# advanced_attacks 의 UAV 피처 인덱스
GPS_COLS = (0, 1)
IMU_COLS = (2, 3)
RESIDUAL_COLS = (4, 5)
BARO_COL = 6
PITCH_COL = 7


@pytest.fixture
def base():
    return AA.gen_normal(N, seed=123)


# residual 열은 주입 후 d[:,4] = d[:,0] - d[:,2] 로 **전 구간** 재계산됩니다.
# 원본 residual 은 float64 계산 후 float32 로 저장된 값이라, 재계산하면 주입과
# 무관한 구간에서도 float32 최소 단위(≈2e-7)만큼 값이 흔들립니다. 실제 주입
# (최소 0.5 이상)과 구분하기 위해 비교에 허용 오차를 둡니다.
FLOAT32_NOISE = 1e-5


def changed_columns(before: np.ndarray, after: np.ndarray,
                    atol: float = FLOAT32_NOISE) -> set:
    delta = np.abs(after - before) > atol
    return set(np.where(delta.any(axis=0))[0].tolist())


def changed_rows(before: np.ndarray, after: np.ndarray,
                 atol: float = FLOAT32_NOISE) -> np.ndarray:
    return np.where((np.abs(after - before) > atol).any(axis=1))[0]


# ─────────────────────────────────────────────────────────────
# UAV (advanced_attacks.py)
# ─────────────────────────────────────────────────────────────
def test_gps_spoofing_touches_gps_and_residual_only(base):
    out = AA.atk_gps_sudden(base, START)
    assert changed_columns(base, out) == set(GPS_COLS) | set(RESIDUAL_COLS)
    assert np.array_equal(out[:, IMU_COLS], base[:, IMU_COLS])
    # 주입은 mag·sin(0.2·t) 형태라 t=0(= START 행)에서 진폭이 정확히 0 이다.
    # 그래서 실제로 값이 달라지는 첫 행은 START + 1 — 이 1 샘플이 GPS 스푸핑의
    # 경계 윈도우 탐지가 다른 공격보다 한 샘플 늦는(k=2 vs k=1) 이유다.
    assert out[START, 0] == pytest.approx(base[START, 0], abs=FLOAT32_NOISE)
    assert changed_rows(base, out).min() == START + 1
    # 주입 이전 구간의 잔차 재계산 오차는 float32 최소 단위 수준이어야 한다
    assert np.abs(out[:START] - base[:START]).max() < FLOAT32_NOISE


def test_gps_spoofing_preserves_residual_identity(base):
    """residual = GPS - IMU 라는 물리 정합성이 주입 후에도 유지돼야 한다."""
    out = AA.atk_gps_sudden(base, START)
    np.testing.assert_allclose(out[:, 4], out[:, 0] - out[:, 2], rtol=1e-5)
    np.testing.assert_allclose(out[:, 5], out[:, 1] - out[:, 3], rtol=1e-5)


def test_altitude_spoofing_touches_baro_only(base):
    out = AA.atk_altitude_sudden(base, START, mag=25.0)
    assert changed_columns(base, out) == {BARO_COL}
    assert changed_rows(base, out).min() == START
    np.testing.assert_allclose(out[START:, BARO_COL],
                               base[START:, BARO_COL] + 25.0, rtol=1e-5)


def test_command_injection_touches_pitch_rate_only(base):
    out = AA.atk_cmd_injection(base, START)
    assert changed_columns(base, out) == {PITCH_COL}
    delta = out[START:, PITCH_COL] - base[START:, PITCH_COL]
    assert delta.min() >= 0.5 and delta.max() <= 0.8


def test_attack_functions_do_not_mutate_input(base):
    snapshot = base.copy()
    for fn in (AA.atk_gps_sudden, AA.atk_altitude_sudden, AA.atk_cmd_injection,
               AA.atk_composite_gps_cmd, AA.atk_composite_gps_alt,
               AA.atk_gradual_gps, AA.atk_gradual_alt):
        fn(base, START)
        np.testing.assert_array_equal(base, snapshot)


def test_prefix_before_attack_start_is_untouched(base):
    for fn in (AA.atk_gps_sudden, AA.atk_altitude_sudden, AA.atk_cmd_injection,
               AA.atk_composite_gps_cmd, AA.atk_composite_gps_alt,
               AA.atk_gradual_gps, AA.atk_gradual_alt):
        out = fn(base, START)
        np.testing.assert_allclose(out[:START], base[:START],
                                   atol=FLOAT32_NOISE, rtol=0)


def test_composite_attacks_are_union_of_their_parts(base):
    gps = changed_columns(base, AA.atk_gps_sudden(base, START))
    alt = changed_columns(base, AA.atk_altitude_sudden(base, START))
    cmd = changed_columns(base, AA.atk_cmd_injection(base, START))
    assert changed_columns(base, AA.atk_composite_gps_cmd(base, START)) == gps | cmd
    assert changed_columns(base, AA.atk_composite_gps_alt(base, START)) == gps | alt


@pytest.mark.parametrize('rate', [0.1, 0.3, 0.5])
def test_gradual_gps_ramp_is_monotone_and_clipped(base, rate):
    out = AA.atk_gradual_gps(base, START, rate=rate, max_mag=12.0)
    delta = out[START:, 0] - base[START:, 0]
    assert np.all(np.diff(delta) >= -FLOAT32_NOISE)   # 단조 증가
    assert delta[0] == pytest.approx(0.0, abs=FLOAT32_NOISE)  # 시작은 0
    assert delta.max() == pytest.approx(12.0, rel=1e-5)
    # y 축은 x 축의 절반 크기로 주입
    dy = out[START:, 1] - base[START:, 1]
    np.testing.assert_allclose(dy, delta * 0.5, rtol=1e-5)


@pytest.mark.parametrize('rate', [0.2, 0.5])
def test_gradual_altitude_ramp_is_monotone_and_clipped(base, rate):
    out = AA.atk_gradual_alt(base, START, rate=rate, max_mag=25.0)
    assert changed_columns(base, out) == {BARO_COL}
    delta = out[START:, BARO_COL] - base[START:, BARO_COL]
    assert np.all(np.diff(delta) >= -FLOAT32_NOISE)
    assert delta.max() == pytest.approx(25.0, rel=1e-5)


def test_gradual_attack_is_initially_weaker_than_sudden(base):
    """점진적 공격은 초기에 급격 공격보다 작아야 한다 (탐지 지연의 원인)."""
    sudden = AA.atk_gps_sudden(base, START)
    gradual = AA.atk_gradual_gps(base, START, rate=0.1)
    d_sudden = np.abs(sudden[START:START + 5, 0] - base[START:START + 5, 0])
    d_gradual = np.abs(gradual[START:START + 5, 0] - base[START:START + 5, 0])
    assert d_gradual.sum() < d_sudden.sum()


# ─────────────────────────────────────────────────────────────
# UGV (utils/ugv_data.py)
# ─────────────────────────────────────────────────────────────
@pytest.mark.parametrize('generator,expected_cols', [
    ('generate_gps_spoofing', {0, 1, 4, 5}),        # gps_vel + residual
    ('generate_wheel_slip', {2, 3, 6, 7}),          # imu 정지 + wheel 급회전
    ('generate_command_anomaly', {6, 7, 8}),        # wheel 무반응 + cmd_vel
])
def test_ugv_generators_touch_expected_columns(generator, expected_cols):
    n = 1000
    normal = ugv_data.generate_normal(n, seed=7)
    out = getattr(ugv_data, generator)(n, seed=7)
    assert changed_columns(normal, out) == expected_cols


@pytest.mark.parametrize('generator,region_key', [
    ('generate_gps_spoofing', 'GPS_Spoofing'),
    ('generate_wheel_slip', 'Wheel_Slip'),
    ('generate_command_anomaly', 'Command_Anomaly'),
])
def test_ugv_changed_rows_match_declared_attack_region(generator, region_key):
    """라벨링에 쓰는 attack_regions 가 실제 변형 구간과 일치해야 한다."""
    n = 1000
    normal = ugv_data.generate_normal(n, seed=7)
    out = getattr(ugv_data, generator)(n, seed=7)
    lo, hi = ugv_data.attack_regions(n)[region_key]
    rows = changed_rows(normal, out)
    assert rows.min() == lo
    assert rows.max() == hi - 1


def test_ugv_gps_spoofing_is_a_burst_not_persistent():
    """GPS 스푸핑만 순간 버스트(n/20) — 라벨 구간 길이 계약."""
    n = 1000
    lo, hi = ugv_data.attack_regions(n)['GPS_Spoofing']
    assert hi - lo == max(15, n // 20)
    assert ugv_data.attack_regions(n)['Wheel_Slip'][1] == n


def test_ugv_generators_do_not_mutate_shared_normal_data():
    n = 200
    a = ugv_data.generate_normal(n, seed=7)
    ugv_data.generate_wheel_slip(n, seed=7)
    np.testing.assert_array_equal(a, ugv_data.generate_normal(n, seed=7))
