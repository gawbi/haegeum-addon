#!/usr/bin/env python3
"""
이상 판정 로직 — 임계값 경계 조건, 포화 케이스, 경계 윈도우 처리, XAI 분류.

`anomaly_detector.AnomalyDetector` 는 ROS2 노드라 생성할 수 없지만, 판정의 핵심인
`_infer` / `_classify` 는 self 의 몇 개 속성만 쓰므로 **원본 메서드를 그대로**
가짜 self 에 바인딩해 검증합니다. (원본 코드는 수정하지 않습니다.)

평가 하네스(advanced_attacks.run_experiment)의 경계 윈도우 제외·탐지 지연 계산은
스텁 검출기로 점수 시퀀스를 직접 주입해 확인합니다.
"""

import types

import numpy as np
import pytest
import torch
import torch.nn as nn

import advanced_attacks as AA

WIN = AA.WIN
N_FEAT = AA.N_FEAT


# ─────────────────────────────────────────────────────────────
# 임계값 경계 조건 (판정 규칙: score > threshold, strict)
# ─────────────────────────────────────────────────────────────
class _StubDetector:
    """미리 정한 점수 시퀀스를 돌려주는 검출기 (run_experiment 하네스 검증용)."""

    def __init__(self, scores: np.ndarray, threshold: float):
        self.scores = np.asarray(scores, dtype=float)
        self.th = threshold

    def score_series(self, data):
        return self.scores[:len(data)]


def _run(scores, threshold, n_total=60, atk_start=30):
    det = _StubDetector(scores, threshold)
    data = np.zeros((n_total, N_FEAT), dtype=np.float32)
    return AA.run_experiment(det, data, 'stub', atk_start, n_total)


def test_score_exactly_at_threshold_is_not_an_anomaly():
    """strict `>` 규칙 — 임계값과 같은 점수는 경보가 아니다."""
    th = 1.2301
    scores = np.full(60, th)
    res = _run(scores, th)
    assert res['TP'] == 0 and res['FP'] == 0
    assert res['delay_ms'] == -1          # 한 번도 탐지되지 않음


def test_score_just_above_threshold_is_an_anomaly():
    th = 1.2301
    scores = np.full(60, th)
    scores[55] = np.nextafter(th, np.inf)   # 표현 가능한 최소 증분
    res = _run(scores, th)
    assert res['TP'] == 1
    assert res['FN'] == 10                  # 평가 대상 양성 11개 중 1개만 탐지
    assert res['delay_ts'] == 25 and res['delay_ms'] == 2500


def test_score_just_below_threshold_is_not_an_anomaly():
    th = 1.2301
    scores = np.full(60, np.nextafter(th, -np.inf))
    res = _run(scores, th)
    assert res['TP'] == 0 and res['FP'] == 0


def test_boundary_windows_are_excluded_from_metrics():
    """
    공격 시작 직후 WIN-1 개 윈도우(정상+공격 혼합)는 지표 집계에서 빠지고,
    탐지 지연 계산에는 그대로 쓰여야 한다.
    """
    th = 1.0
    scores = np.zeros(60)
    scores[30:49] = 5.0        # 경계 구간만 탐지
    res = _run(scores, th)
    assert res['TP'] == 0      # 경계 구간은 TP 로 집계되지 않음
    assert res['FP'] == 0
    assert res['FN'] == 11     # 평가 대상 양성(49~59)은 전부 미탐
    assert res['delay_ts'] == 0    # 지연은 경계 구간 탐지를 인정 → 0 타임스텝


def test_saturated_scores_are_all_detected():
    """점수가 임계값의 10¹² 배로 포화돼도 예외 없이 전부 양성 판정."""
    th = 1.2301
    scores = np.zeros(60)
    scores[30:] = 1e12
    res = _run(scores, th)
    assert res['TP'] == 11 and res['FN'] == 0 and res['FP'] == 0
    assert res['Recall'] == pytest.approx(1.0, abs=1e-6)
    assert res['delay_ts'] == 0


def test_infinite_score_does_not_break_comparison():
    th = 1.2301
    scores = np.zeros(60)
    scores[40:] = np.inf
    res = _run(scores, th)
    assert res['TP'] == 11
    assert np.isfinite(res['F1'])


def test_all_normal_scores_high_counts_as_false_positive():
    th = 1.0
    scores = np.full(60, 2.0)
    res = _run(scores, th)
    assert res['FP'] == 30      # 정상 구간 30개 전부 오탐
    assert res['TP'] == 11
    assert res['Precision'] < res['Recall']


def test_metric_formulas_are_consistent():
    th = 1.0
    rng = np.random.default_rng(0)
    scores = rng.uniform(0, 2, 60)
    res = _run(scores, th)
    TP, FP, FN = res['TP'], res['FP'], res['FN']
    assert res['Precision'] == pytest.approx(TP / (TP + FP), abs=1e-3)
    assert res['Recall'] == pytest.approx(TP / (TP + FN), abs=1e-3)
    p, r = res['Precision'], res['Recall']
    assert res['F1'] == pytest.approx(2 * p * r / (p + r), abs=1e-3)


# ─────────────────────────────────────────────────────────────
# 원본 노드 메서드 (_infer / _classify) 직접 검증
# ─────────────────────────────────────────────────────────────
class _ColumnErrorModel(nn.Module):
    """지정한 피처 열에만 재구성 오차를 만드는 가짜 VAE."""

    def __init__(self, col: int, n_feat: int, magnitude: float = 3.0):
        super().__init__()
        self.col, self.n_feat, self.magnitude = col, n_feat, magnitude

    def forward(self, x):
        delta = torch.zeros(x.shape[0], WIN, self.n_feat)
        delta[:, :, self.col] = self.magnitude
        mu = torch.zeros(x.shape[0], 16)
        return x + delta.reshape(x.shape[0], -1), mu, mu


def _fake_node(detector_module, vae, n_feat, names):
    node = types.SimpleNamespace(vae=vae, window_size=WIN,
                                 n_feat=n_feat, feat_names=names)
    node._infer = types.MethodType(detector_module.AnomalyDetector._infer, node)
    node._classify = types.MethodType(
        detector_module.AnomalyDetector._classify, node)
    return node


def test_node_infer_matches_manual_computation(detector_module, uav_vae):
    vae, cfg = uav_vae
    node = _fake_node(detector_module, vae, cfg['n_feat'], cfg['names'])
    x = torch.randn(1, WIN * cfg['n_feat'])

    torch.manual_seed(7)
    score = node._infer(x)
    torch.manual_seed(7)
    with torch.no_grad():
        recon, _, _ = vae(x)
        expected = torch.mean((recon - x) ** 2).item()

    assert score == pytest.approx(expected, rel=1e-6)
    assert score >= 0


def test_node_infer_exact_value_with_known_model(detector_module):
    """오차를 한 열(8개 중 1개)에만 3.0 심으면 점수는 3²/8 = 1.125."""
    model = _ColumnErrorModel(col=6, n_feat=8, magnitude=3.0)
    node = _fake_node(detector_module, model, 8, list('abcdefgh'))
    x = torch.zeros(1, WIN * 8)
    assert node._infer(x) == pytest.approx(9.0 / 8.0, rel=1e-9)


@pytest.mark.parametrize('col,expected_attack', [
    (0, 'GPS_SPOOFING'),        # gps_vel_x
    (4, 'GPS_SPOOFING'),        # residual_x
    (6, 'ALTITUDE_SPOOFING'),   # baro_alt
    (7, 'COMMAND_INJECTION'),   # pitch_rate
])
def test_node_classify_maps_top_feature_to_attack_type(
        detector_module, col, expected_attack):
    """XAI 분류: 오차가 몰린 피처가 곧 공격 유형이 되어야 한다."""
    names = detector_module.PLATFORM_CONFIG['uav']['names']
    model = _ColumnErrorModel(col=col, n_feat=8)
    node = _fake_node(detector_module, model, 8, names)
    x = torch.zeros(1, WIN * 8)

    attack_type, scenario_ids, top_feat = node._classify(x)
    assert top_feat == names[col]
    assert attack_type == expected_attack
    assert scenario_ids


def test_node_classify_on_real_weights_points_at_injected_sensor(
        detector_module, uav_vae, uav_normal):
    """실제 가중치로도 고도 스푸핑 윈도우에서 baro_alt 가 1위로 나와야 한다."""
    vae, cfg = uav_vae
    data, mean, std = uav_normal
    attacked = AA.atk_altitude_sudden(data, 100)
    w = ((attacked[150:150 + WIN] - mean) / std).reshape(1, -1).astype(np.float32)

    node = _fake_node(detector_module, vae, cfg['n_feat'], cfg['names'])
    torch.manual_seed(42)
    attack_type, _, top_feat = node._classify(torch.from_numpy(w))

    assert top_feat == 'baro_alt'
    assert attack_type == 'ALTITUDE_SPOOFING'
