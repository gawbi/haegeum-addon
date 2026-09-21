#!/usr/bin/env python3
"""
VAE 로드 / 출력 형상 / 플랫폼 설정 정합성.

여기서 지키려는 계약:
  - ROS2 없이도 `anomaly_detector.py` **원본**의 VAE 정의와 기존 가중치를 그대로 쓴다.
  - 가중치 파일(vae_*.pth)의 형상이 PLATFORM_CONFIG 의 피처 수와 어긋나지 않는다.
  - XAI 매핑(FEATURE_TO_ATTACK)에 구멍이 없다 — 어떤 피처가 1위로 나와도 분류된다.
"""

import sys

import pytest
import torch

WIN = 20


def test_import_without_rclpy(detector_module):
    """rclpy 가 없어도 anomaly_detector 를 import 할 수 있어야 한다."""
    assert hasattr(detector_module, 'VAE')
    assert hasattr(detector_module, 'PLATFORM_CONFIG')
    assert hasattr(detector_module, 'FEATURE_TO_ATTACK')
    assert hasattr(detector_module, 'AnomalyDetector')


def test_ros2_stub_blocks_node_construction(detector_module):
    """스텁 환경에서 노드 생성은 조용히 실패하지 않고 명시적으로 막혀야 한다."""
    if getattr(sys.modules.get('rclpy'), '__file__', None):
        pytest.skip('실제 rclpy 환경 — 스텁 계약 검증 대상이 아님')
    with pytest.raises(RuntimeError):
        detector_module.AnomalyDetector()


@pytest.mark.parametrize('platform,n_feat,weight', [
    ('uav', 8, 'vae_uav.pth'),
    ('ugv', 10, 'vae_ugv.pth'),
])
def test_platform_config(detector_module, platform, n_feat, weight):
    cfg = detector_module.PLATFORM_CONFIG[platform]
    assert cfg['n_feat'] == n_feat
    assert len(cfg['names']) == n_feat
    assert len(set(cfg['names'])) == n_feat      # 중복 피처명 없음
    assert cfg['weight_file'] == weight


def test_feature_to_attack_covers_every_feature(detector_module):
    """_classify 의 argmax 결과가 어떤 피처든 UNKNOWN 으로 새지 않아야 한다."""
    for cfg in detector_module.PLATFORM_CONFIG.values():
        for name in cfg['names']:
            assert name in detector_module.FEATURE_TO_ATTACK, name


def test_feature_names_match_data_generators(detector_module):
    """데이터 생성기와 노드의 피처 순서가 어긋나면 정규화·XAI 가 전부 틀어진다."""
    import advanced_attacks as AA
    from utils import ugv_data
    assert detector_module.PLATFORM_CONFIG['uav']['names'] == AA.UAV_FEATS
    assert detector_module.PLATFORM_CONFIG['ugv']['names'] == ugv_data.FEATURE_NAMES


@pytest.mark.parametrize('fixture_name,input_dim', [
    ('uav_vae', WIN * 8),
    ('ugv_vae', WIN * 10),
])
def test_vae_output_shapes(request, fixture_name, input_dim):
    vae, cfg = request.getfixturevalue(fixture_name)
    assert WIN * cfg['n_feat'] == input_dim

    batch = 4
    x = torch.zeros(batch, input_dim)
    with torch.no_grad():
        recon, mu, logvar = vae(x)

    assert recon.shape == (batch, input_dim)
    assert mu.shape == (batch, 16)          # latent_dim 기본값
    assert logvar.shape == (batch, 16)
    assert torch.isfinite(recon).all()


@pytest.mark.parametrize('fixture_name', ['uav_vae', 'ugv_vae'])
def test_vae_is_in_eval_mode_and_frozen_shapes(request, fixture_name):
    vae, cfg = request.getfixturevalue(fixture_name)
    assert not vae.training, 'load_vae 는 eval() 상태로 반환해야 한다'
    # 첫 인코더 레이어의 입력 차원 = window_size × n_feat
    assert vae.encoder[0].in_features == WIN * cfg['n_feat']
    assert vae.decoder[-1].out_features == WIN * cfg['n_feat']
    assert vae.fc_mu.out_features == vae.fc_logvar.out_features == 16


@pytest.mark.parametrize('fixture_name', ['uav_vae', 'ugv_vae'])
def test_encode_decode_roundtrip_shapes(request, fixture_name):
    """설명가능성 스크립트가 쓰는 결정적 경로(z = mu)도 같은 형상을 내야 한다."""
    vae, cfg = request.getfixturevalue(fixture_name)
    x = torch.randn(3, WIN * cfg['n_feat'])
    with torch.no_grad():
        mu, logvar = vae.encode(x)
        recon = vae.decoder(mu)
    assert recon.shape == x.shape
    assert mu.shape == logvar.shape == (3, 16)


def test_unknown_platform_raises():
    from utils.load_model import load_vae
    with pytest.raises(KeyError):
        load_vae('usv', WIN)
