#!/usr/bin/env python3
"""
pytest 공통 설정.

이 저장소의 테스트는 **ROS2(rclpy) 없이** 돌아야 합니다. `utils/load_model.py` 가
rclpy 부재 시 최소 스텁을 주입하므로, 테스트도 반드시 그 경로로만 원본
`anomaly_detector.py` 를 import 합니다 (원본 파일은 수정하지 않습니다).
"""

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

WIN = 20


def pytest_configure(config):
    config.addinivalue_line('markers', 'slow: 모델 가중치를 로드하는 테스트')


@pytest.fixture(scope='session')
def detector_module():
    """ROS2 스텁 경유로 import 한 anomaly_detector 모듈."""
    from utils.load_model import import_anomaly_detector
    return import_anomaly_detector()


@pytest.fixture(scope='session')
def uav_vae():
    from utils.load_model import load_vae
    vae, cfg = load_vae('uav', WIN)
    return vae, cfg


@pytest.fixture(scope='session')
def ugv_vae():
    from utils.load_model import load_vae
    vae, cfg = load_vae('ugv', WIN)
    return vae, cfg


@pytest.fixture(scope='session')
def uav_normal():
    """UAV 정상 시계열 + 정규화 파라미터 (advanced_attacks 와 동일 생성기)."""
    import advanced_attacks as AA
    data = AA.gen_normal(300, seed=42)
    return data, data.mean(0), data.std(0) + 1e-8


@pytest.fixture
def rng():
    return np.random.default_rng(1234)
