#!/usr/bin/env python3
"""
센서 윈도우 구성 로직의 경계 조건.

슬라이딩 윈도우는 탐지 지연과 라벨 정확도를 동시에 좌우합니다.
  - 윈도우 개수/내용/정규화 기준이 어긋나면 점수 분포 자체가 바뀐다.
  - 라벨(정상/공격/경계)의 경계가 한 칸만 밀려도 Recall 이 크게 달라진다
    (README「실험 결과」1)의 Recall 0.714 가 바로 이 라벨 경계 문제였다).
"""

import numpy as np
import pytest

from utils import ugv_data

WIN = ugv_data.WIN
N_FEAT = ugv_data.N_FEAT


@pytest.fixture
def unit_scale():
    """정규화를 항등으로 만드는 mean/std — 윈도우 배치 자체만 검증할 때 사용."""
    return np.zeros(N_FEAT, dtype=np.float32), np.ones(N_FEAT, dtype=np.float32)


def test_window_count_is_n_minus_win_plus_one(unit_scale):
    mean, std = unit_scale
    for n in (WIN, WIN + 1, 100, 1000):
        data = np.zeros((n, N_FEAT), dtype=np.float32)
        assert len(ugv_data.sliding_windows(data, mean, std)) == n - WIN + 1


def test_exactly_one_window_when_length_equals_window_size(unit_scale):
    """가장 빠듯한 경계: 길이 == 윈도우 크기 → 정확히 1개."""
    mean, std = unit_scale
    data = np.arange(WIN * N_FEAT, dtype=np.float32).reshape(WIN, N_FEAT)
    X = ugv_data.sliding_windows(data, mean, std)
    assert X.shape == (1, WIN * N_FEAT)
    np.testing.assert_allclose(X[0], data.reshape(-1))


def test_window_shorter_than_window_size_raises(unit_scale):
    """
    윈도우를 채우지 못하는 길이는 빈 배열이 아니라 예외로 끝난다(np.stack).
    즉 "윈도우가 찰 때까지 호출하지 않는다"가 호출자의 계약이다 —
    anomaly_detector.sensor_callback 이 `len(self.window) < window_size` 에서
    early return 하는 이유가 여기에 있다. 이 계약이 깨지면 조용한 오작동이 아니라
    즉시 실패하도록 현재 동작을 고정해 둔다.
    """
    mean, std = unit_scale
    data = np.zeros((WIN - 1, N_FEAT), dtype=np.float32)
    with pytest.raises(ValueError):
        ugv_data.sliding_windows(data, mean, std)


def test_windows_are_row_major_time_then_feature(unit_scale):
    """
    평탄화 순서가 (타임스텝, 피처) 여야 _classify 의 reshape(WIN, n_feat) 이 맞는다.
    순서가 바뀌면 XAI 가 엉뚱한 피처를 지목한다.
    """
    mean, std = unit_scale
    data = np.arange(30 * N_FEAT, dtype=np.float32).reshape(30, N_FEAT)
    X = ugv_data.sliding_windows(data, mean, std)
    np.testing.assert_allclose(X[0].reshape(WIN, N_FEAT), data[:WIN])
    np.testing.assert_allclose(X[5].reshape(WIN, N_FEAT), data[5:5 + WIN])


def test_windows_apply_normalisation():
    data = np.full((25, N_FEAT), 3.0, dtype=np.float32)
    mean = np.full(N_FEAT, 1.0, dtype=np.float32)
    std = np.full(N_FEAT, 2.0, dtype=np.float32)
    X = ugv_data.sliding_windows(data, mean, std)
    np.testing.assert_allclose(X, np.ones_like(X))


def test_windows_are_float32(unit_scale):
    """torch.FloatTensor 로 바로 넘어가므로 dtype 이 float32 여야 한다."""
    mean, std = unit_scale
    data = np.zeros((40, N_FEAT), dtype=np.float64)
    assert ugv_data.sliding_windows(data, mean, std).dtype == np.float32


# ─────────────────────────────────────────────────────────────
# 라벨 경계
# ─────────────────────────────────────────────────────────────
def test_window_labels_exact_boundaries():
    """
    공격 구간 [lo, hi) 에 대해
      라벨 1 : 윈도우 전체가 구간 안 (j >= lo 이고 j+WIN <= hi)
      라벨 0 : 윈도우가 구간과 전혀 겹치지 않음
      라벨 -1: 걸쳐 있음 (평가 제외)
    """
    n, lo, hi = 200, 50, 150
    lab = ugv_data.window_labels(n, (lo, hi))
    assert len(lab) == n - WIN + 1

    assert lab[lo - WIN] == 0        # 마지막 샘플이 lo 직전 → 완전 정상
    assert lab[lo - WIN + 1] == -1   # 한 샘플만 걸침 → 경계
    assert lab[lo - 1] == -1         # 19 샘플 걸침 → 경계
    assert lab[lo] == 1              # 완전 공격
    assert lab[hi - WIN] == 1        # 마지막 완전 공격 윈도우
    assert lab[hi - WIN + 1] == -1   # 종료 경계
    assert lab[hi] == 0              # 구간 종료 후


def test_boundary_window_count_is_win_minus_one_per_edge():
    n, lo, hi = 1000, 300, 700
    lab = ugv_data.window_labels(n, (lo, hi))
    assert int((lab == -1).sum()) == 2 * (WIN - 1)
    assert int((lab == 1).sum()) == (hi - lo) - WIN + 1


def test_labels_cover_every_window_exactly_once():
    lab = ugv_data.window_labels(1000, (333, 383))
    assert set(np.unique(lab)) <= {-1, 0, 1}
    assert len(lab) == 1000 - WIN + 1


def test_short_attack_region_has_no_fully_contained_window():
    """공격 구간이 윈도우보다 짧으면 '완전 공격' 윈도우가 존재할 수 없다."""
    lab = ugv_data.window_labels(200, (100, 100 + WIN - 1))
    assert int((lab == 1).sum()) == 0
    assert int((lab == -1).sum()) > 0


def test_attack_start_matches_generator_convention():
    for n in (300, 600, 1000):
        assert ugv_data.attack_start(n) == n // 3


def test_labels_align_with_real_generated_attack_rows():
    """라벨 1 윈도우는 실제로 변형된 샘플만 담고 있어야 한다."""
    n = 1000
    normal = ugv_data.generate_normal(n, seed=7)
    attacked = ugv_data.generate_wheel_slip(n, seed=7)
    diff_rows = np.where((np.abs(attacked - normal) > 1e-5).any(axis=1))[0]
    lab = ugv_data.window_labels(n, ugv_data.attack_regions(n)['Wheel_Slip'])

    for j in np.where(lab == 1)[0]:
        rows = set(range(j, j + WIN))
        assert rows <= set(diff_rows.tolist())

    for j in np.where(lab == 0)[0]:
        rows = set(range(j, j + WIN))
        assert not (rows & set(diff_rows.tolist()))
