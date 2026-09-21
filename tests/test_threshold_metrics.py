#!/usr/bin/env python3
"""
임계값 분석(analysis/threshold_selection.py)의 지표 계산 검증.

ROC/AUC 와 기준별 임계값은 README 에 실리는 수치이므로, 다음을 고정합니다.
  - ROC 좌표가 실제 판정 규칙(strict `score > threshold`)과 한 점도 어긋나지 않는다
  - AUC 가 표준 구현(scikit-learn)과 일치한다
  - 각 선택 기준이 정의대로(예: 목표 FPR 을 넘지 않게) 고른다
"""

import numpy as np
import pytest

from analysis import threshold_selection as TS


# ─────────────────────────────────────────────────────────────
# 혼동행렬 / 지표
# ─────────────────────────────────────────────────────────────
def test_confusion_uses_strict_greater_than():
    y = np.array([0, 1])
    s = np.array([1.0, 1.0])
    c = TS.confusion_at(y, s, 1.0)
    assert c == {'TP': 0, 'FP': 0, 'TN': 1, 'FN': 1}

    c = TS.confusion_at(y, s, np.nextafter(1.0, -np.inf))
    assert c == {'TP': 1, 'FP': 1, 'TN': 0, 'FN': 0}


def test_metrics_known_confusion_matrix():
    y = np.array([1, 1, 1, 1, 0, 0, 0, 0, 0, 0])
    s = np.array([2, 2, 2, 0, 2, 0, 0, 0, 0, 0], dtype=float)
    m = TS.metrics_at(y, s, 1.0)
    assert (m['TP'], m['FP'], m['TN'], m['FN']) == (3, 1, 5, 1)
    assert m['precision'] == pytest.approx(0.75)
    assert m['recall'] == pytest.approx(0.75)
    assert m['f1'] == pytest.approx(0.75)
    assert m['fpr'] == pytest.approx(1 / 6, abs=1e-6)


def test_metrics_handle_no_positive_prediction():
    y = np.array([1, 0])
    s = np.array([0.1, 0.1])
    m = TS.metrics_at(y, s, 1.0)
    assert m['precision'] == 0.0 and m['recall'] == 0.0 and m['f1'] == 0.0


# ─────────────────────────────────────────────────────────────
# ROC / AUC
# ─────────────────────────────────────────────────────────────
def test_roc_points_match_metrics_at_every_threshold(rng):
    """ROC 의 모든 좌표가 같은 임계값의 실제 판정 결과와 일치해야 한다."""
    y = rng.integers(0, 2, 200)
    s = rng.normal(size=200) + y * 0.8
    fpr, tpr, thr = TS.roc_curve_strict(y, s)

    assert len(fpr) == len(tpr) == len(thr)
    # metrics_at 은 JSON 기록용으로 반올림하므로, 여기서는 원 혼동행렬로 비교한다.
    for f, t, th in zip(fpr, tpr, thr):
        c = TS.confusion_at(y, s, th)
        assert c['TP'] / (c['TP'] + c['FN']) == pytest.approx(t, abs=1e-12)
        assert c['FP'] / (c['FP'] + c['TN']) == pytest.approx(f, abs=1e-12)


def test_roc_is_monotone_and_spans_the_unit_square(rng):
    y = rng.integers(0, 2, 100)
    s = rng.normal(size=100)
    fpr, tpr, _ = TS.roc_curve_strict(y, s)
    assert np.all(np.diff(fpr) >= -1e-12)
    assert np.all(np.diff(tpr) >= -1e-12)
    assert (fpr[0], tpr[0]) == (0.0, 0.0)
    assert (fpr[-1], tpr[-1]) == (1.0, 1.0)


def test_auc_is_one_for_perfect_separation():
    y = np.r_[np.zeros(50), np.ones(50)]
    s = np.r_[np.linspace(0, 1, 50), np.linspace(2, 3, 50)]
    assert TS.roc_auc(y, s) == pytest.approx(1.0)


def test_auc_is_half_for_indistinguishable_scores():
    y = np.r_[np.zeros(50), np.ones(50)]
    s = np.ones(100)
    assert TS.roc_auc(y, s) == pytest.approx(0.5)


def test_auc_is_zero_for_perfectly_inverted_scores():
    y = np.r_[np.zeros(50), np.ones(50)]
    s = np.r_[np.ones(50) * 5, np.zeros(50)]
    assert TS.roc_auc(y, s) == pytest.approx(0.0)


def test_auc_matches_scikit_learn(rng):
    sklearn_metrics = pytest.importorskip('sklearn.metrics')
    for _ in range(5):
        y = rng.integers(0, 2, 300)
        # 동점을 일부러 만들어 tie 처리까지 비교
        s = np.round(rng.normal(size=300) + y * 0.6, 1)
        assert TS.roc_auc(y, s) == pytest.approx(
            sklearn_metrics.roc_auc_score(y, s), abs=1e-9)


def test_roc_requires_both_classes():
    with pytest.raises(ValueError):
        TS.roc_curve_strict(np.ones(10), np.arange(10.0))


# ─────────────────────────────────────────────────────────────
# 임계값 선택 기준
# ─────────────────────────────────────────────────────────────
@pytest.fixture
def toy():
    y = np.r_[np.zeros(100), np.ones(100)].astype(int)
    s = np.r_[np.linspace(0.0, 1.0, 100), np.linspace(0.5, 1.5, 100)]
    return y, s


def test_target_fpr_never_exceeds_the_budget(toy):
    y, s = toy
    fpr, tpr, thr = TS.roc_curve_strict(y, s)
    for target in (0.01, 0.05, 0.10):
        t = TS.pick_target_fpr(fpr, tpr, thr, target)
        assert TS.metrics_at(y, s, t)['fpr'] <= target + 1e-12


def test_target_fpr_takes_the_best_recall_within_budget(toy):
    y, s = toy
    fpr, tpr, thr = TS.roc_curve_strict(y, s)
    t = TS.pick_target_fpr(fpr, tpr, thr, 0.05)
    best = max(TS.metrics_at(y, s, c)['recall']
               for c in thr[np.isfinite(thr)]
               if TS.metrics_at(y, s, c)['fpr'] <= 0.05 + 1e-12)
    assert TS.metrics_at(y, s, t)['recall'] == pytest.approx(best)


def test_looser_fpr_budget_gives_lower_threshold_and_higher_recall(toy):
    y, s = toy
    fpr, tpr, thr = TS.roc_curve_strict(y, s)
    strict_t = TS.pick_target_fpr(fpr, tpr, thr, 0.01)
    loose_t = TS.pick_target_fpr(fpr, tpr, thr, 0.10)
    assert loose_t <= strict_t
    assert TS.metrics_at(y, s, loose_t)['recall'] >= \
        TS.metrics_at(y, s, strict_t)['recall']


def test_youden_maximises_tpr_minus_fpr(toy):
    y, s = toy
    fpr, tpr, thr = TS.roc_curve_strict(y, s)
    t = TS.pick_youden(fpr, tpr, thr)
    j = TS.metrics_at(y, s, t)
    best = max(tpr - fpr)
    assert (j['recall'] - j['fpr']) == pytest.approx(best, abs=1e-9)


def test_max_f1_is_not_beaten_by_any_candidate(toy):
    y, s = toy
    candidates = TS.roc_curve_strict(y, s)[2]
    candidates = candidates[np.isfinite(candidates)]
    t = TS.pick_max_f1(y, s, candidates)
    best = TS.metrics_at(y, s, t)['f1']
    assert all(TS.metrics_at(y, s, c)['f1'] <= best + 1e-12 for c in candidates)


def test_higher_fn_cost_lowers_the_threshold(toy):
    """미탐 비용이 커질수록 임계값은 내려가고 Recall 은 올라가야 한다."""
    y, s = toy
    candidates = TS.roc_curve_strict(y, s)[2]
    candidates = candidates[np.isfinite(candidates)]
    thresholds = [TS.pick_min_cost(y, s, candidates, cost_fn=k)
                  for k in (1, 10, 100)]
    assert thresholds[0] >= thresholds[1] >= thresholds[2]
    recalls = [TS.metrics_at(y, s, t)['recall'] for t in thresholds]
    assert recalls[0] <= recalls[1] <= recalls[2]


def test_min_cost_objective_is_actually_minimal(toy):
    y, s = toy
    candidates = TS.roc_curve_strict(y, s)[2]
    candidates = candidates[np.isfinite(candidates)]
    t = TS.pick_min_cost(y, s, candidates, cost_fn=10.0)

    def cost(th):
        c = TS.confusion_at(y, s, th)
        return 10.0 * c['FN'] + c['FP']

    assert all(cost(t) <= cost(c) + 1e-12 for c in candidates)


# ─────────────────────────────────────────────────────────────
# 운용 환산 / 주입 유틸
# ─────────────────────────────────────────────────────────────
def test_alarm_load_conversion():
    """FPR 1 % @10 Hz → 시간당 360건, 쿨다운 2 s 상한은 1800건."""
    load = TS.alarm_load(0.01)
    assert load['false_alarms_per_hour_raw'] == pytest.approx(360.0)
    assert load['false_alarms_per_hour_with_cooldown'] == pytest.approx(360.0)

    saturated = TS.alarm_load(1.0)
    assert saturated['false_alarms_per_hour_raw'] == pytest.approx(36000.0)
    assert saturated['false_alarms_per_hour_with_cooldown'] == pytest.approx(1800.0)

    assert TS.alarm_load(0.0)['false_alarms_per_hour_raw'] == 0.0


def test_inject_sigma_touches_only_target_columns_after_start():
    std = np.arange(1, 9, dtype=np.float32)
    data = np.zeros((50, 8), dtype=np.float32)
    out = TS.inject_sigma(data, 20, (6,), 2.0, std)

    assert np.array_equal(out[:20], data[:20])
    assert np.all(out[20:, 6] == pytest.approx(2.0 * std[6]))
    other = [c for c in range(8) if c != 6]
    assert np.array_equal(out[:, other], data[:, other])
    assert np.array_equal(data, np.zeros((50, 8), dtype=np.float32))  # 원본 불변


def test_inject_sigma_recomputes_residual_columns():
    std = np.ones(8, dtype=np.float32)
    data = np.zeros((30, 8), dtype=np.float32)
    out = TS.inject_sigma(data, 10, (0, 1), 3.0, std, residual_cols=(4, 5))
    np.testing.assert_allclose(out[:, 4], out[:, 0] - out[:, 2])
    np.testing.assert_allclose(out[:, 5], out[:, 1] - out[:, 3])
    assert out[15, 4] == pytest.approx(3.0)


def test_separation_margin_flags_a_dominated_threshold():
    neg = np.linspace(0.5, 1.5, 100)
    pos = np.linspace(10.0, 20.0, 100)
    sm = TS.separation_margin(neg, pos, current=1.2)
    assert sm['perfectly_separable'] is True
    assert sm['current_below_max_normal'] is True
    assert sm['current_fpr_on_heldout_normal'] > 0

    ok = TS.separation_margin(neg, pos, current=5.0)
    assert ok['current_below_max_normal'] is False
    assert ok['current_fpr_on_heldout_normal'] == 0.0
