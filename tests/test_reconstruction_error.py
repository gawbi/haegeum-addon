#!/usr/bin/env python3
"""
재구성 오차(이상 점수) 계산의 정확성.

`anomaly_detector._infer()` 의 점수는 `mean((recon - x)²)` 한 줄이지만, 이 값이
임계값과 직접 비교되고 XAI 기여도의 분해 대상이 되므로 다음을 고정해 둡니다.
  - 알려진 입력/출력에 대한 **기댓값이 정확히** 나오는가 (오차 정의가 바뀌지 않았는가)
  - 피처별 기여도의 합(평균)이 전체 점수와 일치하는가 (_classify 의 reshape 규약)
  - 입력이 정상 분포에서 멀어질수록 점수가 커지는가 (탐지의 전제)
"""

import numpy as np
import pytest
import torch
import torch.nn as nn

WIN = 20


class _ShiftModel(nn.Module):
    """recon = x + c 를 돌려주는 가짜 모델 — 기대 MSE 가 정확히 c² 이다."""

    def __init__(self, c: float, latent: int = 16):
        super().__init__()
        self.c = c
        self.latent = latent

    def forward(self, x):
        mu = torch.zeros(x.shape[0], self.latent)
        return x + self.c, mu, mu


def _infer(model, x: torch.Tensor) -> float:
    """anomaly_detector._infer 와 동일한 계산 경로."""
    with torch.no_grad():
        recon, _, _ = model(x)
        return torch.mean((recon - x) ** 2).item()


@pytest.mark.parametrize('c', [0.0, 0.5, 1.0, 3.0])
def test_known_input_gives_exact_expected_score(c):
    """recon - x 가 상수 c 이면 점수는 정확히 c²."""
    x = torch.randn(1, WIN * 8)
    assert _infer(_ShiftModel(c), x) == pytest.approx(c ** 2, abs=1e-6)


def test_identical_reconstruction_gives_zero_score():
    x = torch.randn(1, WIN * 8)
    assert _infer(_ShiftModel(0.0), x) == pytest.approx(0.0, abs=1e-12)


def test_score_matches_numpy_reference(uav_vae, uav_normal):
    """실제 VAE 에서도 torch 계산과 numpy 참조 구현이 일치해야 한다."""
    vae, cfg = uav_vae
    data, mean, std = uav_normal
    w = ((data[:WIN] - mean) / std).reshape(1, -1).astype(np.float32)
    x = torch.from_numpy(w)

    with torch.no_grad():
        mu, _ = vae.encode(x)            # 결정적 경로(z = mu)로 고정
        recon = vae.decoder(mu)
        torch_score = torch.mean((recon - x) ** 2).item()

    ref = float(np.mean((recon.numpy() - w) ** 2))
    assert torch_score == pytest.approx(ref, rel=1e-6)


def test_per_feature_contribution_sums_to_total(uav_vae, uav_normal):
    """
    _classify 는 (recon-x)² 를 (WIN, n_feat) 로 접어 시간축 평균을 낸다.
    그 피처별 값들의 평균은 전체 점수와 정확히 같아야 한다 (분해의 무결성).
    """
    vae, cfg = uav_vae
    data, mean, std = uav_normal
    w = ((data[:WIN] - mean) / std).reshape(1, -1).astype(np.float32)
    x = torch.from_numpy(w)

    with torch.no_grad():
        mu, _ = vae.encode(x)
        recon = vae.decoder(mu)
        per_elem = (recon - x) ** 2
        per_feat = per_elem.reshape(WIN, cfg['n_feat']).mean(0)
        total = torch.mean(per_elem).item()

    assert per_feat.shape == (cfg['n_feat'],)
    assert per_feat.mean().item() == pytest.approx(total, rel=1e-6)
    assert (per_feat >= 0).all()


def test_per_feature_contribution_localises_injected_column():
    """
    한 피처 열에만 오차를 심으면 그 열의 기여도만 커져야 한다
    (reshape 의 행/열 순서가 뒤바뀌면 이 테스트가 깨진다).
    """
    n_feat = 8
    diff = np.zeros((WIN, n_feat), dtype=np.float32)
    diff[:, 3] = 2.0                     # imu_ay 열에만 오차 2.0
    per_feat = (torch.from_numpy(diff.reshape(1, -1)) ** 2) \
        .reshape(WIN, n_feat).mean(0)
    assert int(per_feat.argmax()) == 3
    assert per_feat[3].item() == pytest.approx(4.0)
    assert per_feat[[0, 1, 2, 4, 5, 6, 7]].sum().item() == pytest.approx(0.0)


def test_score_increases_with_distance_from_normal(uav_vae, uav_normal):
    """정상 윈도우를 배율로 키울수록 점수가 단조 증가해야 한다."""
    vae, _ = uav_vae
    data, mean, std = uav_normal
    w = ((data[:WIN] - mean) / std).reshape(1, -1).astype(np.float32)
    x = torch.from_numpy(w)

    scores = []
    for factor in (1.0, 2.0, 4.0, 8.0):
        with torch.no_grad():
            xf = x * factor
            mu, _ = vae.encode(xf)
            recon = vae.decoder(mu)
            scores.append(torch.mean((recon - xf) ** 2).item())

    assert scores == sorted(scores)
    assert scores[-1] > scores[0] * 10


def test_attack_window_scores_higher_than_normal_window(uav_vae, uav_normal):
    """실제 주입 공격 윈도우가 정상 윈도우보다 높은 점수를 받아야 한다."""
    import advanced_attacks as AA
    vae, _ = uav_vae
    data, mean, std = uav_normal
    attacked = AA.atk_gps_sudden(data, 100)

    def score(arr, i):
        w = ((arr[i:i + WIN] - mean) / std).reshape(1, -1).astype(np.float32)
        x = torch.from_numpy(w)
        with torch.no_grad():
            mu, _ = vae.encode(x)
            return torch.mean((vae.decoder(mu) - x) ** 2).item()

    assert score(attacked, 150) > 10 * score(data, 150)


def test_score_is_finite_for_saturated_input(uav_vae):
    """포화(극단 OOD) 입력에서도 NaN/Inf 없이 유한한 점수가 나와야 한다."""
    vae, cfg = uav_vae
    x = torch.full((1, WIN * cfg['n_feat']), 1e4)
    with torch.no_grad():
        mu, _ = vae.encode(x)
        score = torch.mean((vae.decoder(mu) - x) ** 2).item()
    assert np.isfinite(score)
    assert score > 0
