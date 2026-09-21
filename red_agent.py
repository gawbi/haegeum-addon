#!/usr/bin/env python3
"""
DAH 2026 해금팀 — Red Agent (공격 AI) ROS2 노드
참고 체계: 국내 C-UAS(대드론) 탐지 레이더, 지능형 방공 체계 (공개 자료 기반 일반 구조)

공격 캠페인 4단계 (APT 구조):
  Phase 1 — 정찰 (Reconnaissance)
    스카우트 드론 1대 저고도 침투 → 방어 레이더 탐지 범위·토픽 구조 파악
    MITRE ATT&CK ICS: T0842 (Network Sniffing)

  Phase 2 — 전자전 (Electronic Warfare)
    GPS 재머 드론 투입 → /sensor_data에 스푸핑된 GPS 데이터 주입
    MAVLink 데이터링크 재밍 → Altitude Spoofing / Command Injection
    MITRE ATT&CK ICS: T0830 (Adversary-in-the-Middle), T0855 (Unauthorized Command)

  Phase 3 — 포화 공격 (Swarm Saturation)
    군집 드론 N대 다방향 동시 접근 → ai_commander 의사결정 포화
    C-UAS 체계가 처리 가능한 트랙 수 초과 시도 (탐지 한계 초과)
    MITRE ATT&CK ICS: T0814 (Denial of Service)

  Phase 4 — 페이로드 (Payload)
    목표 근접 후 우군 UGV cmd_vel 위조 → 경로 이탈 유도
    우군 UAV pitch_rate 위조 → 비행 제어 탈취
    MITRE ATT&CK ICS: T0803 (Block Command Message)

ROS2 토픽:
  발행(공격): /sensor_data/ugv_01  — GPS Spoofing 데이터 주입
  발행(공격): /sensor_data/uav_01  — Altitude/Command Injection 주입
  발행(공격): /radar_tracks         — 군집 드론 트랙 시뮬레이션
  발행(공격): /target_detected      — 허위 표적 발행 (ai_commander 포화)
  구독(정찰): /robot_status         — 우군 위치·상태 파악
  구독(정찰): /assignment           — 우군 임무 할당 盜聽
"""

import json
import math
import random
import time
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from std_msgs.msg import Float32MultiArray

try:
    from haegeum_interfaces.msg import TargetDetected, RobotStatus
    from geometry_msgs.msg import Point
    HAS_HAEGEUM = True
except ImportError:
    HAS_HAEGEUM = False
    print('[WARN] haegeum_interfaces 없음 — 시뮬레이션 모드')


# ── 공격 캠페인 파라미터 ──────────────────────────────────

# Phase 1: 정찰
SCOUT_ALTITUDE_M      = 15.0    # 저고도 침투 고도 (레이더 탐지 회피)
SCOUT_SPEED_MPS       = 8.0     # 정찰 드론 속도 (느릴수록 RCS 감소)
RECON_DURATION_SEC    = 30.0    # 정찰 완료까지 대기

# Phase 2: EW
GPS_SPOOF_MAGNITUDE   = 12.0    # GPS 속도 조작 크기 (m/s) — residual 폭발 유도
ALT_SPOOF_DELTA_M     = 25.0    # 고도계 조작 크기 (m)
CMD_INJECT_PITCH      = 0.8     # 비정상 pitch_rate (rad/s)
EW_DURATION_SEC       = 20.0

# Phase 3: 군집
SWARM_SIZE            = 6       # 군집 드론 수 (C-UAS 체계 포화 목표)
SWARM_CLOSING_SPEED   = 20.0    # 접근 속도 (m/s)
SWARM_APPROACH_DIRS   = [       # 다방향 동시 접근 방위각 (도)
    0, 60, 120, 180, 240, 300
]
SWARM_DURATION_SEC    = 40.0

# Phase 4: 페이로드
PAYLOAD_CMD_VEL_FAKE  = -1.0    # 역방향 명령 위조


class RedAgent(Node):
    """
    4단계 공격 캠페인을 자동 실행하는 Red Agent.

    공격 전략:
    - 지능형 방공 체계 분석: 탐지→분류→규모파악→방책결심 파이프라인의
      '분류' 단계를 혼란시킴 (낮은 RCS 드론 + GPS없는 SLAM 항법으로 탐지 회피)
    - C-UAS 체계 대응: 레이더 처리 트랙 수 포화 → 과부하 유도
    - Blue Agent(광빈) 대응: VAE 임계값 초과 전에 정상 패턴 유지하다
      갑작스러운 신호 점프로 슬라이딩 윈도우 경계 구간에서 공격 실행
      (Recall 취약점 노린 타이밍 공격)
    """

    PHASES = ['IDLE', 'RECON', 'EW', 'SWARM', 'PAYLOAD', 'COMPLETE']

    def __init__(self):
        super().__init__('red_agent')

        self.declare_parameter('target_ugv', 'ugv_01')
        self.declare_parameter('target_uav', 'uav_01')
        self.declare_parameter('auto_start', True)
        self.declare_parameter('swarm_size', SWARM_SIZE)

        self.target_ugv = self.get_parameter('target_ugv').value
        self.target_uav = self.get_parameter('target_uav').value
        self.auto_start = self.get_parameter('auto_start').value
        self.swarm_size = self.get_parameter('swarm_size').value

        self.phase      = 'IDLE'
        self.phase_start = time.time()
        self.friendly_status = {}   # 정찰로 수집한 우군 상태
        self.campaign_log = []

        qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT)

        # ── 공격용 발행자 ──────────────────────────────────
        self.ugv_sensor_pub = self.create_publisher(
            Float32MultiArray,
            f'/sensor_data/{self.target_ugv}',
            qos,
        )
        self.uav_sensor_pub = self.create_publisher(
            Float32MultiArray,
            f'/sensor_data/{self.target_uav}',
            qos,
        )
        self.radar_pub = self.create_publisher(
            Float32MultiArray, '/radar_tracks', qos,
        )

        if HAS_HAEGEUM:
            self.fake_target_pub = self.create_publisher(
                TargetDetected, '/target_detected', 10,
            )
            # 정찰: 우군 상태 도청
            self.create_subscription(
                RobotStatus, '/robot_status',
                self.recon_callback, 10,
            )

        # 10Hz 공격 루프
        self.create_timer(0.1, self.attack_loop)
        self.t = 0.0

        if self.auto_start:
            self._transition('RECON')

        self.get_logger().info('RedAgent 시작 — 공격 캠페인 준비')

    # ── 상태 전이 ──────────────────────────────────────────
    def _transition(self, next_phase: str):
        elapsed = time.time() - self.phase_start
        self.get_logger().warn(
            f'[PHASE] {self.phase} → {next_phase} '
            f'(이전 단계 {elapsed:.1f}초 경과)'
        )
        self.campaign_log.append({
            'phase':   next_phase,
            'time':    time.time(),
            'elapsed': elapsed,
        })
        self.phase       = next_phase
        self.phase_start = time.time()

    # ── 메인 공격 루프 (10Hz) ──────────────────────────────
    def attack_loop(self):
        self.t += 0.1
        elapsed = time.time() - self.phase_start

        if self.phase == 'RECON':
            self._phase_recon(elapsed)
        elif self.phase == 'EW':
            self._phase_ew(elapsed)
        elif self.phase == 'SWARM':
            self._phase_swarm(elapsed)
        elif self.phase == 'PAYLOAD':
            self._phase_payload(elapsed)
        elif self.phase == 'COMPLETE':
            pass

    # ── Phase 1: 정찰 ──────────────────────────────────────
    def _phase_recon(self, elapsed: float):
        """
        저고도(15m) 저속(8m/s) 비행으로 방어 레이더 탐지 최소화.
        RCS가 작은 소형 드론(~-10 dBsm) + GPS없는 SLAM 항법 사용으로
        AESA 방공 레이더 탐지 확률 저감 (Swerling-1 모델 기준 Pd ≈ 0.3).
        """
        # 스카우트 드론 트랙 1개 — 낮은 RCS(-10 dBsm), 느린 속도
        track = Float32MultiArray()
        angle = (elapsed * SCOUT_SPEED_MPS / 500.0) * 360  # 원형 정찰
        rng   = 300.0 - elapsed * 2.0   # 서서히 접근
        track.data = [
            float(1),                   # track_id
            float(max(50.0, rng)),       # range_m
            float(angle % 360),          # azimuth_deg
            float(5.0),                  # elevation_deg (저고도)
            float(-SCOUT_SPEED_MPS * 0.3),  # radial_vel (느린 접근)
            float(-10.0),                # rcs_dbsm (소형 드론)
        ]
        self.radar_pub.publish(track)

        if elapsed >= RECON_DURATION_SEC:
            ugv_count = sum(
                1 for s in self.friendly_status.values()
                if s.get('type') == 'UGV'
            )
            uav_count = sum(
                1 for s in self.friendly_status.values()
                if s.get('type') == 'UAV'
            )
            self.get_logger().info(
                f'[RECON 완료] 우군 UGV={ugv_count}, UAV={uav_count}'
            )
            self._transition('EW')

    # ── Phase 2: 전자전 ────────────────────────────────────
    def _phase_ew(self, elapsed: float):
        """
        GPS Spoofing: GPS 속도 신호에 급격한 점프값 주입.
        VAE 슬라이딩 윈도우(20 타임스텝) 경계 구간에서 공격 시작 →
        윈도우 절반이 정상 데이터로 채워질 때까지 탐지 지연 유도.

        Altitude Spoofing: baro_alt에 +25m 지속 오프셋.
        Command Injection: pitch_rate에 비정상 진동 주입.
        """
        # UGV GPS Spoofing
        ugv_msg = Float32MultiArray()
        spoof_x = GPS_SPOOF_MAGNITUDE * np.sin(elapsed * 2.0)
        ugv_msg.data = [
            float(0.5 + spoof_x),       # gps_vel_x 조작
            float(0.1 * spoof_x),        # gps_vel_y 조작
            float(0.5),                  # imu_ax 정상 (residual 폭발 유도)
            float(0.0),                  # imu_ay 정상
            float(spoof_x),              # residual_x 폭발
            float(0.1 * spoof_x),        # residual_y 폭발
            float(0.5),                  # wheel_vel_l 정상
            float(0.5),                  # wheel_vel_r 정상
            float(0.5),                  # cmd_vel_x 정상
            float(0.0),                  # cmd_vel_w 정상
        ]
        self.ugv_sensor_pub.publish(ugv_msg)

        # UAV Altitude + Command Injection
        uav_msg = Float32MultiArray()
        pitch_inject = CMD_INJECT_PITCH * np.sin(elapsed * 5.0)
        uav_msg.data = [
            float(5.0),                            # gps_vel_x 정상
            float(0.0),                            # gps_vel_y 정상
            float(5.0),                            # imu_ax 정상
            float(0.0),                            # imu_ay 정상
            float(0.0),                            # residual_x 정상
            float(0.0),                            # residual_y 정상
            float(50.0 + ALT_SPOOF_DELTA_M),       # baro_alt 고도 조작
            float(pitch_inject),                   # pitch_rate 진동 주입
        ]
        self.uav_sensor_pub.publish(uav_msg)

        # 허위 표적 ai_commander로 발행 (방어 자원 분산)
        if HAS_HAEGEUM and elapsed % 3.0 < 0.1:
            fake = TargetDetected()
            fake.target_id   = f'fake_ew_{int(elapsed)}'
            fake.target_type = 'Enemy_UAV'
            fake.position    = Point(
                x=random.uniform(-100, 100),
                y=random.uniform(-100, 100),
                z=random.uniform(10, 50),
            )
            fake.detected_by = 'red_agent_ew'
            fake.confidence  = random.uniform(0.55, 0.75)
            self.fake_target_pub.publish(fake)

        if elapsed >= EW_DURATION_SEC:
            self._transition('SWARM')

    # ── Phase 3: 군집 포화 공격 ────────────────────────────
    def _phase_swarm(self, elapsed: float):
        """
        SWARM_SIZE대의 드론이 360°/N 간격으로 동시 접근.
        C-UAS 체계의 처리 가능 트랙 수를 초과하도록 설계.
        지능형 방공 체계의 AI 분류 단계에서 소형 드론 RCS(-5~0 dBsm)와
        속도 프로파일이 새떼와 유사하도록 조정 → 오분류 유도.
        """
        tracks_data = []
        for i in range(self.swarm_size):
            az   = SWARM_APPROACH_DIRS[i % len(SWARM_APPROACH_DIRS)]
            rng  = max(50.0, 800.0 - elapsed * SWARM_CLOSING_SPEED)
            el   = random.uniform(5.0, 25.0)

            # 새떼 유사 RCS(-5 dBsm) + 불규칙 속도 → AI 분류 혼란
            rcs_noise = random.gauss(-5.0, 2.0)
            vel_noise = -SWARM_CLOSING_SPEED + random.gauss(0, 3.0)

            tracks_data.extend([
                float(100 + i),     # track_id
                float(rng),          # range_m
                float(az),           # azimuth_deg
                float(el),           # elevation_deg
                float(vel_noise),    # radial_vel (접근)
                float(rcs_noise),    # rcs_dbsm
            ])

            # 각 드론마다 허위 표적도 발행
            if HAS_HAEGEUM and elapsed % 1.0 < 0.1:
                fake = TargetDetected()
                fake.target_id   = f'swarm_{i}_{int(elapsed)}'
                fake.target_type = random.choice(['Enemy_UAV', 'Enemy_UGV'])
                az_rad = math.radians(az)
                fake.position = Point(
                    x=float(rng * math.sin(az_rad)),
                    y=float(rng * math.cos(az_rad)),
                    z=float(el),
                )
                fake.detected_by = 'red_agent_swarm'
                fake.confidence  = random.uniform(0.6, 0.95)
                self.fake_target_pub.publish(fake)

        msg = Float32MultiArray()
        msg.data = tracks_data
        self.radar_pub.publish(msg)

        if elapsed >= SWARM_DURATION_SEC:
            self._transition('PAYLOAD')

    # ── Phase 4: 페이로드 ──────────────────────────────────
    def _phase_payload(self, elapsed: float):
        """
        군집 공격으로 방어 자원이 분산된 틈에 핵심 공격 실행.
        UGV cmd_vel을 역방향으로 위조 → 임무 구역 이탈.
        """
        ugv_msg = Float32MultiArray()
        ugv_msg.data = [
            float(PAYLOAD_CMD_VEL_FAKE),  # gps_vel_x 역방향
            float(random.gauss(0, 0.1)),
            float(PAYLOAD_CMD_VEL_FAKE),  # imu_ax 역방향
            float(random.gauss(0, 0.05)),
            float(0.0), float(0.0),
            float(-0.5), float(-0.5),      # wheel_vel 역방향
            float(PAYLOAD_CMD_VEL_FAKE),  # cmd_vel_x 역방향
            float(random.gauss(0, 0.02)),
        ]
        self.ugv_sensor_pub.publish(ugv_msg)

        if elapsed >= 30.0:
            self._log_campaign_result()
            self._transition('COMPLETE')

    # ── 정찰 콜백 — 우군 상태 수집 ────────────────────────
    def recon_callback(self, msg: RobotStatus):
        if self.phase == 'RECON':
            self.friendly_status[msg.robot_id] = {
                'type':  msg.robot_type,
                'state': msg.state,
                'x':     msg.pose.position.x,
                'y':     msg.pose.position.y,
                'z':     msg.pose.position.z,
            }

    # ── 캠페인 결과 로그 ───────────────────────────────────
    def _log_campaign_result(self):
        total = time.time() - self.campaign_log[0]['time'] if self.campaign_log else 0
        result = {
            'campaign_total_sec': round(total, 1),
            'phases': self.campaign_log,
            'friendlies_observed': list(self.friendly_status.keys()),
            'swarm_size': self.swarm_size,
            'attack_vectors': [
                'GPS_SPOOFING (UGV residual)',
                'ALTITUDE_SPOOFING (UAV baro_alt)',
                'COMMAND_INJECTION (UAV pitch_rate)',
                'SWARM_SATURATION (레이더 포화)',
                'FAKE_TARGET_FLOOD (ai_commander 포화)',
            ],
        }
        self.get_logger().warn(
            f'[캠페인 완료] {json.dumps(result, ensure_ascii=False, indent=2)}'
        )


def main():
    rclpy.init()
    node = RedAgent()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
