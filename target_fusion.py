#!/usr/bin/env python3
"""
DAH 2026 해금팀 — 다중 소스 표적 융합 노드
위치: L2 중대 엣지 클라우드

융합 소스:
  1) VAE 이상탐지 (/target_detected)     — anomaly_detector.py 출력
  2) 레이더 트랙   (/radar_tracks)        — 국내 C-UAS 드론 탐지 레이더
  3) EO/IR 카메라  (/camera_detections)  — 광전자/적외선 센서

융합 알고리즘: 가중 평균 기반 신뢰도 융합
  - VAE 탐지: 신뢰도 가중치 0.5 (사이버 공격 특화)
  - 레이더:   신뢰도 가중치 0.35 (물리적 위협 특화, 탐지거리 ~4km)
  - 카메라:   신뢰도 가중치 0.15 (근거리 식별 보조, ~500m)

출력: /fused_targets (TargetDetected) → ai_commander
"""

import time
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from std_msgs.msg import Float32MultiArray

try:
    from haegeum_interfaces.msg import TargetDetected
    from geometry_msgs.msg import Point
    HAS_HAEGEUM = True
except ImportError:
    HAS_HAEGEUM = False

# 소스별 신뢰도 가중치
SOURCE_WEIGHT = {
    'VAE':    0.50,
    'RADAR':  0.35,
    'CAMERA': 0.15,
}

# 동일 표적 판단 거리 임계값 (m)
FUSION_RADIUS = 20.0


class FusedTarget:
    def __init__(self, target_id, target_type, position, confidence, source):
        self.target_id   = target_id
        self.target_type = target_type
        self.position    = position      # [x, y, z]
        self.confidence  = confidence
        self.sources     = {source: confidence}
        self.last_update = time.time()

    def fuse(self, confidence, source, position):
        """새 관측값을 가중 평균으로 융합"""
        w = SOURCE_WEIGHT.get(source, 0.1)
        self.sources[source] = confidence
        # 위치: 가중 평균
        w_sum = sum(SOURCE_WEIGHT.get(s, 0.1) for s in self.sources)
        for i in range(3):
            self.position[i] = (
                self.position[i] * (w_sum - w) + position[i] * w
            ) / w_sum
        # 신뢰도: 소스별 가중 평균
        self.confidence = sum(
            SOURCE_WEIGHT.get(s, 0.1) * c
            for s, c in self.sources.items()
        ) / w_sum
        self.last_update = time.time()

    def is_stale(self, ttl=5.0):
        return time.time() - self.last_update > ttl


class TargetFusion(Node):

    def __init__(self):
        super().__init__('target_fusion')

        self.fused: dict[str, FusedTarget] = {}

        qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT)

        # 구독
        if HAS_HAEGEUM:
            self.create_subscription(
                TargetDetected, '/target_detected',
                self.vae_callback, 10,
            )
        self.create_subscription(
            Float32MultiArray, '/radar_tracks',
            self.radar_callback, qos,
        )
        self.create_subscription(
            Float32MultiArray, '/camera_detections',
            self.camera_callback, qos,
        )

        # 발행
        if HAS_HAEGEUM:
            self.fused_pub = self.create_publisher(
                TargetDetected, '/fused_targets', 10,
            )
        else:
            self.fused_pub = None

        # 0.5초마다 융합 결과 발행 + stale 제거
        self.create_timer(0.5, self.publish_fused)
        self.get_logger().info('TargetFusion 시작')

    # ── VAE 탐지 입력 ───────────────────────────────────────
    def vae_callback(self, msg):
        pos = [msg.position.x, msg.position.y, msg.position.z]
        self._update_or_create(
            target_id   = msg.target_id,
            target_type = msg.target_type,
            position    = pos,
            confidence  = msg.confidence,
            source      = 'VAE',
        )

    # ── 레이더 트랙 입력 ────────────────────────────────────
    def radar_callback(self, msg):
        data = list(msg.data)
        n_tracks = len(data) // 6
        for i in range(n_tracks):
            chunk = data[i*6:(i+1)*6]
            if len(chunk) < 6:
                break
            rng = chunk[1]
            az  = np.radians(chunk[2])
            el  = np.radians(chunk[3])
            rcs = chunk[5]

            # 극좌표 → 직교좌표
            x = rng * np.cos(el) * np.sin(az)
            y = rng * np.cos(el) * np.cos(az)
            z = rng * np.sin(el)

            # RCS 기반 신뢰도 (소형 드론 -10~0 dBsm → confidence 0.7~0.9)
            confidence = float(np.clip(0.7 + (rcs + 10) * 0.02, 0.5, 0.99))

            self._update_or_create(
                target_id   = f'radar_{int(chunk[0])}',
                target_type = 'Enemy_UAV',
                position    = [x, y, z],
                confidence  = confidence,
                source      = 'RADAR',
            )

    # ── 카메라 탐지 입력 ────────────────────────────────────
    def camera_callback(self, msg):
        """
        카메라 탐지 형식: [x, y, z, confidence, class_id, ...]
        class_id: 0=드론, 1=새, 2=구름
        """
        data = list(msg.data)
        n_det = len(data) // 5
        for i in range(n_det):
            chunk = data[i*5:(i+1)*5]
            if len(chunk) < 5:
                break
            class_id = int(chunk[4])
            if class_id != 0:   # 드론 아니면 무시
                continue
            self._update_or_create(
                target_id   = f'cam_{i}_{int(time.time())}',
                target_type = 'Enemy_UAV',
                position    = [chunk[0], chunk[1], chunk[2]],
                confidence  = float(chunk[3]),
                source      = 'CAMERA',
            )

    # ── 융합 로직 ───────────────────────────────────────────
    def _update_or_create(self, target_id, target_type, position,
                          confidence, source):
        # 동일 표적 탐색 (거리 기반)
        best_key = None
        best_dist = FUSION_RADIUS
        for key, ft in self.fused.items():
            dist = np.linalg.norm(np.array(position) - np.array(ft.position))
            if dist < best_dist:
                best_dist = dist
                best_key  = key

        if best_key:
            self.fused[best_key].fuse(confidence, source, position)
        else:
            self.fused[target_id] = FusedTarget(
                target_id, target_type, position, confidence, source
            )

    # ── 융합 결과 발행 ──────────────────────────────────────
    def publish_fused(self):
        # stale 제거
        stale = [k for k, v in self.fused.items() if v.is_stale()]
        for k in stale:
            del self.fused[k]

        if not HAS_HAEGEUM or not self.fused_pub:
            return

        for ft in self.fused.values():
            msg = TargetDetected()
            msg.target_id   = ft.target_id
            msg.target_type = ft.target_type
            msg.position    = Point(
                x=float(ft.position[0]),
                y=float(ft.position[1]),
                z=float(ft.position[2]),
            )
            msg.detected_by = 'target_fusion'
            msg.confidence  = float(ft.confidence)
            self.fused_pub.publish(msg)


def main():
    rclpy.init()
    node = TargetFusion()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
