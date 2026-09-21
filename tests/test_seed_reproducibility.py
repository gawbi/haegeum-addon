#!/usr/bin/env python3
"""
시드 고정이 실제로 재현성을 보장하는지.

벤치마크·SHAP·임계값 분석의 모든 수치는 "같은 시드면 같은 값"을 전제로 기록돼
있습니다. VAE 가 재파라미터화 샘플링(z = mu + std·eps)을 쓰기 때문에 추론 경로에도
난수가 들어가므로, 시드 고정이 깨지면 표의 수치가 재현되지 않습니다.
"""

import random

import numpy as np
import pytest
import torch

from utils.seed import DEFAULT_SEED, env_info, set_seed

WIN = 20


def _draw():
    return (random.random(),
            float(np.random.rand()),
            float(torch.rand(1).item()))


def test_set_seed_returns_seed_and_is_reproducible():
    assert set_seed(123) == 123
    first = _draw()
    set_seed(123)
    assert _draw() == first


def test_default_seed_is_42():
    assert DEFAULT_SEED == 42


def test_different_seeds_give_different_draws():
    set_seed(1)
    a = _draw()
    set_seed(2)
    b = _draw()
    assert a != b


def test_pythonhashseed_is_exported():
    import os
    set_seed(7)
    assert os.environ['PYTHONHASHSEED'] == '7'


def test_vae_stochastic_inference_is_reproducible_under_seed(uav_vae):
    """샘플링 경로에서도 시드를 고정하면 점수가 비트 단위로 같아야 한다."""
    vae, cfg = uav_vae
    x = torch.zeros(1, WIN * cfg['n_feat'])

    def score():
        with torch.no_grad():
            recon, _, _ = vae(x)
            return torch.mean((recon - x) ** 2).item()

    set_seed(DEFAULT_SEED)
    first = [score() for _ in range(5)]
    set_seed(DEFAULT_SEED)
    second = [score() for _ in range(5)]
    assert first == second


def test_vae_sampling_actually_varies_without_reseeding(uav_vae, uav_normal):
    """
    시드를 다시 고정하지 않으면 같은 입력에도 점수가 달라진다 —
    README 가 임계값 1.2283 / 1.2301 차이를 설명한 근거.
    """
    vae, cfg = uav_vae
    data, mean, std = uav_normal
    w = ((data[:WIN] - mean) / std).reshape(1, -1).astype(np.float32)
    x = torch.from_numpy(w)

    set_seed(DEFAULT_SEED)
    with torch.no_grad():
        scores = [torch.mean((vae(x)[0] - x) ** 2).item() for _ in range(50)]

    assert len(set(scores)) > 1                     # 확률적 경로임을 확인
    # 다만 흔들림은 정상 윈도우 점수의 10 % 미만 — 임계값 판정을 뒤집을 크기가 아니다
    assert np.std(scores) / np.mean(scores) < 0.10


def test_deterministic_path_is_bit_identical_without_reseeding(uav_vae):
    """z = mu 결정적 경로는 시드와 무관하게 항상 같은 값이어야 한다 (SHAP 전제)."""
    vae, cfg = uav_vae
    x = torch.zeros(1, WIN * cfg['n_feat'])
    with torch.no_grad():
        mu, _ = vae.encode(x)
        a = torch.mean((vae.decoder(mu) - x) ** 2).item()
        mu, _ = vae.encode(x)
        b = torch.mean((vae.decoder(mu) - x) ** 2).item()
    assert a == b


@pytest.mark.parametrize('generator', ['generate_normal', 'generate_gps_spoofing',
                                       'generate_wheel_slip',
                                       'generate_command_anomaly'])
def test_ugv_generators_are_seed_deterministic(generator):
    from utils import ugv_data
    fn = getattr(ugv_data, generator)
    assert np.array_equal(fn(200, seed=5), fn(200, seed=5))
    assert not np.array_equal(fn(200, seed=5), fn(200, seed=6))


def test_uav_generator_is_seed_deterministic():
    import advanced_attacks as AA
    assert np.array_equal(AA.gen_normal(100, seed=3), AA.gen_normal(100, seed=3))
    assert not np.array_equal(AA.gen_normal(100, seed=3), AA.gen_normal(100, seed=4))


def test_env_info_records_measurement_environment():
    info = env_info()
    for key in ('python', 'platform', 'machine', 'cpu_count'):
        assert key in info
    assert info['torch'] == torch.__version__
    assert info['numpy'] == np.__version__
