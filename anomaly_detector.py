#!/usr/bin/env python3
"""
DAH 2026 해금팀 — VAE 기반 실시간 이상탐지 ROS2 노드 (레이더 소스 확장 버전)
위치: L1 물리 실행층 (UAV/UGV 온보드)
역할: STANAG 4586 DLI 계층 센서 무결성 실시간 검증

센서 소스:
  - GPS/IMU: MAVLink GLOBAL_POSITION_INT / HIGHRES_IMU
  - 레이더:  LIG Nex1 드론 탐지 레이더 (탐지 거리: ~4km, 방위각 분해능: 1°)
             → /radar_tracks 토픽으로 수신, 위협 물체 포지션/속도 포함
  - 카메라:  EO/IR 센서 → target_fusion 노드에서 융합

ROS2 토픽:
  구독: /sensor_data/{robot_id}  (Float32MultiArray) — GPS/IMU
  구독: /radar_tracks            (Float32MultiArray) — 레이더 탐지 트랙
  발행: /events                  (Event)             → cloud_logger → L3
  발행: /target_detected         (TargetDetected)    → ai_commander → 방어 임무 할당

커버 시나리오: G-4, A-1, A-4, G-1, J-3 + 군집 드론 재밍(SWARM_JAMMING)
"""

import json
import time
import numpy as np
import torch
import torch.nn as nn
from collections import deque

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from std_msgs.msg import Float32MultiArray

try:
    from haegeum_interfaces.msg import Event, TargetDetected
    from geometry_msgs.msg import Point
    HAS_HAEGEUM = True
except ImportError:
    HAS_HAEGEUM = False
    print('[WARN] haegeum_interfaces 없음 — 시뮬레이션 모드')


# ── VAE 모델 ──────────────────────────────────────────────
class VAE(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int = 16):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128), nn.ReLU(),
            nn.Linear(128, 64),        nn.ReLU(),
        )
        self.fc_mu     = nn.Linear(64, latent_dim)
        self.fc_logvar = nn.Linear(64, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64), nn.ReLU(),
            nn.Linear(64, 128),        nn.ReLU(),
            nn.Linear(128, input_dim),
        )

    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        return mu + std * torch.randn_like(std)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decoder(z), mu, logvar


# ── 플랫폼별 피처 정의 ────────────────────────────────────
PLATFORM_CONFIG = {
    'ugv': {
        'n_feat': 10,
        'names': [
            'gps_vel_x', 'gps_vel_y',
            'imu_ax',    'imu_ay',
            'residual_x','residual_y',
            'wheel_vel_l','wheel_vel_r',
            'cmd_vel_x', 'cmd_vel_w',
        ],
        'weight_file': 'vae_ugv.pth',
    },
    'uav': {
        'n_feat': 8,
        'names': [
            'gps_vel_x', 'gps_vel_y',
            'imu_ax',    'imu_ay',
            'residual_x','residual_y',
            'baro_alt',  'pitch_rate',
        ],
        'weight_file': 'vae_uav.pth',
    },
}

# 피처 → STANAG 공격 유형 매핑
FEATURE_TO_ATTACK = {
    'residual_x':  ('GPS_SPOOFING',     'G-4,A-1'),
    'residual_y':  ('GPS_SPOOFING',     'G-4,A-1'),
    'gps_vel_x':   ('GPS_SPOOFING',     'G-4,A-1'),
    'gps_vel_y':   ('GPS_SPOOFING',     'G-4,A-1'),
    'baro_alt':    ('ALTITUDE_SPOOFING','A-4'),
    'pitch_rate':  ('COMMAND_INJECTION','A-4,G-1'),
    'imu_ax':      ('COMMAND_INJECTION','A-4,G-1'),
    'imu_ay':      ('COMMAND_INJECTION','A-4,G-1'),
    'wheel_vel_l': ('COMMAND_INJECTION','G-1'),
    'wheel_vel_r': ('COMMAND_INJECTION','G-1'),
    'cmd_vel_x':   ('COMMAND_INJECTION','G-1'),
    'cmd_vel_w':   ('COMMAND_INJECTION','G-1'),
}

# 레이더 트랙 이상 임계값
# LIG Nex1 드론 탐지 레이더 기준 (탐지 거리 ~4km, 속도 분해능 0.5m/s)
RADAR_SWARM_THRESHOLD = 3      # 동시 탐지 트랙 수 ≥ 3 → 군집 드론 의심
RADAR_CLOSING_SPEED   = 15.0   # m/s 이상 접근 속도 → 위협 드론 의심


class AnomalyDetector(Node):
    """
    L1 온보드 VAE 이상탐지 노드 (레이더 융합 버전).

    탐지 레이어:
    1) VAE 기반 센서 이상탐지 (GPS/IMU 신호)
    2) 레이더 트랙 기반 군집 드론 탐지 (SWARM_JAMMING)

    두 레이어 중 하나라도 이상 감지 시 /target_detected 발행.
    """

    def __init__(self):
        super().__init__('anomaly_detector')

        self.declare_parameter('platform',             'ugv')
        self.declare_parameter('robot_id',             'ugv_01')
        self.declare_parameter('window_size',           20)
        self.declare_parameter('calib_size',            200)
        self.declare_parameter('threshold_percentile',  95.0)
        self.declare_parameter('cooldown_sec',          2.0)
        self.declare_parameter('enable_radar',          True)

        platform    = self.get_parameter('platform').value
        self.robot_id    = self.get_parameter('robot_id').value
        self.window_size = self.get_parameter('window_size').value
        calib_size  = self.get_parameter('calib_size').value
        thresh_pct  = self.get_parameter('threshold_percentile').value
        self.cooldown    = self.get_parameter('cooldown_sec').value
        self.enable_radar = self.get_parameter('enable_radar').value

        cfg = PLATFORM_CONFIG[platform]
        self.n_feat     = cfg['n_feat']
        self.feat_names = cfg['names']

        # VAE 로드
        self.vae = VAE(input_dim=self.window_size * self.n_feat)
        try:
            self.vae.load_state_dict(
                torch.load(cfg['weight_file'], map_location='cpu')
            )
            self.get_logger().info(f'VAE 가중치 로드: {cfg["weight_file"]}')
        except FileNotFoundError:
            self.get_logger().warn(f'가중치 없음({cfg["weight_file"]}) — 보정 후 운용')
        self.vae.eval()

        # 상태 변수
        self.window      = deque(maxlen=self.window_size)
        self.calib_buf   = []
        self.calib_size  = calib_size
        self.thresh_pct  = thresh_pct
        self.threshold   = None
        self.mean        = None
        self.std         = None
        self.last_alert  = 0.0
        self.alert_count = 0

        # 레이더 트랙 버퍼 (최근 1초치)
        self.radar_tracks = []

        qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT)

        # 구독
        self.create_subscription(
            Float32MultiArray,
            f'/sensor_data/{self.robot_id}',
            self.sensor_callback, qos,
        )
        if self.enable_radar:
            self.create_subscription(
                Float32MultiArray,
                '/radar_tracks',
                self.radar_callback, qos,
            )

        # 발행
        if HAS_HAEGEUM:
            self.event_pub  = self.create_publisher(Event,          '/events',          10)
            self.target_pub = self.create_publisher(TargetDetected, '/target_detected', 10)
        else:
            self.event_pub = self.target_pub = None

        self.create_timer(30.0, self.status_report)
        self.get_logger().info(
            f'AnomalyDetector 시작 | platform={platform} | '
            f'robot={self.robot_id} | radar={self.enable_radar}'
        )

    # ── GPS/IMU 센서 콜백 ──────────────────────────────────
    def sensor_callback(self, msg: Float32MultiArray):
        raw = np.array(msg.data, dtype=np.float32)
        if len(raw) < self.n_feat:
            return
        feat = raw[:self.n_feat].copy()

        # GPS-IMU residual 자동 계산
        if len(feat) >= 6:
            feat[4] = feat[0] - feat[2]
            feat[5] = feat[1] - feat[3]

        self.window.append(feat)
        if len(self.window) < self.window_size:
            return

        window_arr = np.array(self.window, dtype=np.float32)
        if self.mean is None:
            self.mean = window_arr.mean(0)
            self.std  = window_arr.std(0) + 1e-8

        normed = (window_arr - self.mean) / self.std
        flat   = torch.FloatTensor(normed.reshape(1, -1))
        score  = self._infer(flat)

        if self.threshold is None:
            self.calib_buf.append(score)
            if len(self.calib_buf) >= self.calib_size:
                self.threshold = np.percentile(self.calib_buf, self.thresh_pct)
                self.get_logger().info(f'임계값 보정: {self.threshold:.4f}')
            return

        if score > self.threshold:
            now = time.time()
            if now - self.last_alert > self.cooldown:
                attack_type, scenario_ids, top_feat = self._classify(flat)
                self._publish_alert(score, attack_type, scenario_ids, top_feat)
                self.last_alert = now
                self.alert_count += 1

    # ── 레이더 콜백 — 군집 드론 탐지 ─────────────────────
    def radar_callback(self, msg: Float32MultiArray):
        """
        레이더 트랙 수신 형식 (LIG Nex1 드론 탐지 레이더 기준):
        [track_id, range_m, azimuth_deg, elevation_deg, radial_vel_mps, rcs_dbsm, ...]
        6개 값이 1개 트랙. 복수 트랙이 연속으로 들어옴.
        """
        data = list(msg.data)
        n_tracks = len(data) // 6
        self.radar_tracks = []

        for i in range(n_tracks):
            chunk = data[i*6:(i+1)*6]
            if len(chunk) < 6:
                break
            self.radar_tracks.append({
                'track_id':    int(chunk[0]),
                'range_m':     chunk[1],
                'azimuth_deg': chunk[2],
                'elevation_deg': chunk[3],
                'radial_vel':  chunk[4],   # 음수 = 접근
                'rcs_dbsm':    chunk[5],   # Radar Cross Section (소형 드론 ≈ -10~0 dBsm)
            })

        # 군집 드론 판단
        if n_tracks >= RADAR_SWARM_THRESHOLD:
            closing = [t for t in self.radar_tracks if t['radial_vel'] < -RADAR_CLOSING_SPEED]
            if len(closing) >= RADAR_SWARM_THRESHOLD:
                now = time.time()
                if now - self.last_alert > self.cooldown:
                    score = float(len(closing))
                    self._publish_alert(
                        score=score,
                        attack_type='SWARM_JAMMING',
                        scenario_ids='G-4,A-1,J-3',
                        top_feat=f'radar_tracks×{len(closing)}',
                        source='RADAR',
                    )
                    self.last_alert = now
                    self.alert_count += 1

    # ── VAE 추론 ───────────────────────────────────────────
    def _infer(self, x: torch.Tensor) -> float:
        with torch.no_grad():
            recon, _, _ = self.vae(x)
            return torch.mean((recon - x) ** 2).item()

    # ── XAI: 피처별 기여도 → 공격 유형 분류 ──────────────
    def _classify(self, x: torch.Tensor):
        with torch.no_grad():
            recon, _, _ = self.vae(x)
            per_feat = (recon - x) ** 2
            per_feat = per_feat.reshape(self.window_size, self.n_feat).mean(0)
            top_idx  = int(per_feat.argmax().item())
        top_feat = self.feat_names[top_idx]
        attack_type, scenario_ids = FEATURE_TO_ATTACK.get(
            top_feat, ('UNKNOWN_ANOMALY', 'J-3')
        )
        return attack_type, scenario_ids, top_feat

    # ── 이벤트 발행 ────────────────────────────────────────
    def _publish_alert(self, score, attack_type, scenario_ids, top_feat,
                       source='VAE'):
        ts = time.time()
        detail = json.dumps({
            'attack_type':  attack_type,
            'score':        round(float(score), 4),
            'threshold':    round(self.threshold, 4) if self.threshold else None,
            'top_feature':  top_feat,
            'scenario_ids': scenario_ids,
            'stanag_layer': 'DLI',
            'loi_risk':     'Level4-5' if attack_type == 'COMMAND_INJECTION' else 'Level1-3',
            'source':       source,
            'timestamp':    ts,
            'alert_count':  self.alert_count + 1,
        })

        self.get_logger().warn(
            f'[ANOMALY #{self.alert_count+1}] {attack_type} | '
            f'score={score:.4f} | feat={top_feat} | src={source}'
        )

        if HAS_HAEGEUM and self.event_pub and self.target_pub:
            event = Event()
            event.source  = self.robot_id
            event.level   = 'ERROR'
            event.message = detail
            self.event_pub.publish(event)

            target = TargetDetected()
            target.target_id   = f'anomaly_{self.robot_id}_{int(ts)}'
            target.target_type = attack_type
            target.position    = Point(x=0.0, y=0.0, z=0.0)
            target.detected_by = self.robot_id
            target.confidence  = min(1.0, float(score / ((self.threshold or 1.0) * 3)))
            self.target_pub.publish(target)
        else:
            print(f'[SIM] {detail}')

    def status_report(self):
        status = 'CALIBRATING' if self.threshold is None else 'ACTIVE'
        self.get_logger().info(
            f'[STATUS] {status} | threshold={self.threshold} | '
            f'alerts={self.alert_count} | radar_tracks={len(self.radar_tracks)}'
        )


def main():
    rclpy.init()
    node = AnomalyDetector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
