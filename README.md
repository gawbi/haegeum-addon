# DAH 2026 해금팀 — L1 방어 + Red Agent 모듈
**담당**: 광빈 (김광빈) | **소속**: 한성대학교 AI응용학과

> "클라우드 기반 UAV/UGV 무인체계의 AI Red/Blue Agent 보안 프레임워크"  
> 메인 레포: [seohee-k/haegeum](https://github.com/seohee-k/haegeum)

---

## 모듈 구성

| 파일 | 역할 | 레이어 |
|---|---|---|
| `anomaly_detector.py` | VAE 기반 센서 이상탐지 + 레이더 융합 | L1 물리 실행층 |
| `target_fusion.py` | 레이더/VAE/카메라 다중 소스 융합 | L2 엣지 클라우드 |
| `red_agent.py` | 4단계 공격 캠페인 자동 실행 | Red Agent |
| `sensor_publisher.py` | 정상 센서 데이터 발행 (테스트용) | L1 |
| `vae_ugv.pth` | UGV VAE 학습 가중치 (input=200) | L1 |
| `vae_uav.pth` | UAV VAE 학습 가중치 (input=160) | L1 |

---

## Red Agent — 4단계 공격 캠페인

한화시스템 지능형 방공 체계 및 LIG Nex1 C-UAS를 역설계한 공격 시나리오.

```
Phase 1 RECON   — 저고도(15m) 스카우트 드론 정찰
                  RCS ≈ -10 dBsm, SLAM 항법(GPS 없음)
                  → LIG AESA 레이더 탐지 확률 Pd ≈ 0.3 수준으로 저감

Phase 2 EW      — GPS Spoofing + Altitude Spoofing + Command Injection
                  GPS速度 조작(±12m/s) → VAE residual 폭발 유도
                  baro_alt +25m → UAV 고도 오인
                  pitch_rate 진동 → 비행 제어 교란

Phase 3 SWARM   — 군집 드론 6대 360°/60° 간격 동시 접근(20m/s)
                  RCS ≈ -5 dBsm + 불규칙 속도 → AI 분류기 오분류 유도(새떼 위장)
                  LIG C-UAS 처리 트랙 수 포화 목표

Phase 4 PAYLOAD — cmd_vel 역방향 위조 → UGV 임무 구역 이탈
```

### MITRE ATT&CK for ICS 매핑

| 단계 | 전술 ID | 전술명 |
|---|---|---|
| Phase 1 | T0842 | Network Sniffing |
| Phase 2 | T0830 | Adversary-in-the-Middle |
| Phase 2 | T0855 | Unauthorized Command Message |
| Phase 3 | T0814 | Denial of Service |
| Phase 4 | T0803 | Block Command Message |

---

## Blue Agent (anomaly_detector.py) — 방어 탐지

### VAE 성능 (합성 데이터 기반)

| 플랫폼 | 공격 유형 | Precision | Recall | F1 |
|---|---|---|---|---|
| UGV | GPS Spoofing | 1.000 | 1.000 | 1.000 |
| UGV | Wheel Slip | 1.000 | 1.000 | 1.000 |
| UGV | Command Anomaly | 1.000 | 1.000 | 1.000 |
| UAV | GPS Spoofing | 1.000 | 0.850 | 0.850 |
| UAV | Altitude Spoofing | 1.000 | 0.855 | 0.855 |
| UAV | Command Injection | 1.000 | 1.000 | 1.000 |

**Precision=1.000**: 오탐 없음 → 채점 공식 `(공격점수 + 방어점수) × 가용성계수` 에서 가용성계수 최대 유지 (J-3)

### 센서 소스

```
GPS/IMU  — MAVLink GLOBAL_POSITION_INT / HIGHRES_IMU
레이더   — LIG Nex1 드론 탐지 레이더 (AESA, 탐지거리 ~4km)
           /radar_tracks 토픽, 포맷: [track_id, range_m, az_deg, el_deg, vel_mps, rcs_dbsm]
카메라   — EO/IR 센서 → target_fusion 노드 융합
```

### STANAG 4586 커버리지

| 시나리오 | 탐지 피처 | 탐지 소스 |
|---|---|---|
| G-4 UGV 센서 스푸핑 | residual_x/y 폭발 | VAE |
| A-1 DLI 권한상승 | LOI 불일치 + residual | VAE |
| A-4 데이터링크 하이재킹 | baro_alt + pitch_rate | VAE |
| G-1 UGV 페이로드 탈취 | wheel_vel + cmd_vel | VAE |
| J-3 가용성 보전 | Precision=1.0 | VAE |
| SWARM_JAMMING 군집 재밍 | 레이더 트랙 수 ≥ 3 | 레이더 |

---

## 실행 방법

```bash
# 의존성
pip install torch numpy rclpy

# anomaly_detector (UGV)
ros2 run haegeum_addon anomaly_detector --ros-args -p platform:=ugv -p robot_id:=ugv_01

# anomaly_detector (UAV)
ros2 run haegeum_addon anomaly_detector --ros-args -p platform:=uav -p robot_id:=uav_01

# target_fusion
ros2 run haegeum_addon target_fusion

# Red Agent (공격 시뮬레이션)
ros2 run haegeum_addon red_agent --ros-args -p swarm_size:=6

# 센서 데이터 퍼블리셔 (테스트)
ros2 run haegeum_addon sensor_publisher --ros-args -p platform:=ugv
```

---

## 참고

- LIG Nex1 C-UAS: AESA 레이더 기반 탐지·추적·재밍, 500회+ 시험 비행 검증 ([Army Recognition](https://armyrecognition.com/news/army-news/army-news-2024/lig-nex1-unveils-innovative-solutions-for-heavy-transport-and-anti-drone-defense-at-kadex-2024))
- 한화시스템 지능형 방공: AI 탐지→분류→규모파악→방책결심 ([Hanwha Systems](https://www.hanwhasystems.com/en/business/defense/land.do))
- MITRE ATT&CK for ICS: [https://attack.mitre.org/matrices/ics/](https://attack.mitre.org/matrices/ics/)
- STANAG 4586: UAV 지상관제 스테이션 인터페이스 표준
