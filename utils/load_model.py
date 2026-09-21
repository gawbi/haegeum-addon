#!/usr/bin/env python3
"""
기존 anomaly_detector.py 의 VAE 정의와 가중치를 그대로 재사용하기 위한 로더.

anomaly_detector.py 는 ROS2 노드 모듈이라 import 시 rclpy / std_msgs 가 필요합니다.
ROS2 가 없는 환경(벤치마크·SHAP 분석용 CPU 컨테이너)에서도 **원본 파일을 수정하지 않고**
같은 VAE 클래스·PLATFORM_CONFIG 를 쓰기 위해, rclpy 가 없을 때만 최소 스텁을 주입합니다.
"""

import sys
import types
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent


def _install_ros2_stubs() -> bool:
    """rclpy 가 없으면 import 만 통과시키는 최소 스텁 주입. 주입했으면 True."""
    try:
        import rclpy  # noqa: F401
        return False
    except ImportError:
        pass

    rclpy = types.ModuleType('rclpy')
    rclpy.init = lambda *a, **k: None
    rclpy.shutdown = lambda *a, **k: None
    rclpy.spin = lambda *a, **k: None

    node_mod = types.ModuleType('rclpy.node')

    class _Node:  # AnomalyDetector 의 베이스 클래스 자리만 채움
        def __init__(self, *a, **k):
            raise RuntimeError('ROS2 스텁 환경에서는 노드를 실행할 수 없습니다')

    node_mod.Node = _Node

    qos_mod = types.ModuleType('rclpy.qos')

    class _QoSProfile:
        def __init__(self, *a, **k):
            pass

    class _ReliabilityPolicy:
        BEST_EFFORT = 0
        RELIABLE = 1

    qos_mod.QoSProfile = _QoSProfile
    qos_mod.ReliabilityPolicy = _ReliabilityPolicy

    std_msgs = types.ModuleType('std_msgs')
    std_msgs_msg = types.ModuleType('std_msgs.msg')

    class _Float32MultiArray:
        def __init__(self):
            self.data = []

    std_msgs_msg.Float32MultiArray = _Float32MultiArray
    std_msgs.msg = std_msgs_msg

    rclpy.node = node_mod
    rclpy.qos = qos_mod

    sys.modules.update({
        'rclpy': rclpy,
        'rclpy.node': node_mod,
        'rclpy.qos': qos_mod,
        'std_msgs': std_msgs,
        'std_msgs.msg': std_msgs_msg,
    })
    return True


def import_anomaly_detector():
    """anomaly_detector 모듈을 그대로 import 해서 반환 (VAE, PLATFORM_CONFIG 등)."""
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    _install_ros2_stubs()
    import anomaly_detector
    return anomaly_detector


def load_vae(platform: str, window_size: int = 20):
    """
    기존 가중치(vae_uav.pth / vae_ugv.pth)를 anomaly_detector.VAE 에 로드.

    Returns: (vae, cfg) — cfg 는 PLATFORM_CONFIG[platform]
    """
    mod = import_anomaly_detector()
    cfg = mod.PLATFORM_CONFIG[platform]
    vae = mod.VAE(input_dim=window_size * cfg['n_feat'])
    state = torch.load(REPO_ROOT / cfg['weight_file'],
                       map_location='cpu', weights_only=True)
    vae.load_state_dict(state)
    vae.eval()
    return vae, cfg
