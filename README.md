# 해금(Haegeum) — UAV/UGV 통합 VAE 이상탐지 & Red Agent 공격 시뮬레이션

> 유·무인 복합체계(MUM-T)의 센서 무결성을 실시간으로 검증하는 **L1 방어(Blue) 모듈**과, 이를 시험·평가하기 위한 **4단계 자동 공격 캠페인(Red Agent)**을 함께 담은 방산 AI 보안 프레임워크입니다.
>
> DAH 2026(국방 AI 해커톤) 출전작 · 팀 **해금**

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-VAE-EE4C2C?logo=pytorch&logoColor=white)
![ROS2](https://img.shields.io/badge/ROS2-rclpy-22314E?logo=ros&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![Status](https://img.shields.io/badge/scope-simulation%20only-orange.svg)

> ⚠️ **범위 고지**: 이 저장소의 모든 공격/방어 코드는 **합성 데이터 기반 시뮬레이션**이며, ROS2 토픽 메시지 위·변조 실험에 한정됩니다. 실제 무기체계나 실기체에 대한 운용 정보를 포함하지 않습니다.

---

## 문제 정의

유·무인 복합체계(UAV/UGV)는 GPS·IMU·기압계·레이더·EO/IR 등 다수의 센서에 의존해 항법과 임무를 수행합니다. 그러나 이 센서 계층은 다음과 같은 위협에 직접 노출됩니다.

- **GPS 스푸핑 / 재밍** — 위조 위성 신호로 위치·속도 추정을 오염시켜 경로 이탈을 유도
- **데이터링크 하이재킹 / 명령 주입** — MAVLink 등 제어 링크를 가로채 고도·자세 명령을 위조
- **군집 드론 포화(Saturation)** — 다방향 동시 접근으로 탐지·분류·의사결정 파이프라인을 과부하

이러한 공격은 임무 시작 후 **수백 ms 단위**로 전개되므로, 클라우드 재분석을 기다릴 수 없습니다. 따라서 **기체 온보드(L1)에서 센서 스트림 자체의 이상 거동을 실시간으로 탐지**하는 자기 방어 계층이 필요합니다.

본 프로젝트는 정상 센서 패턴만 학습한 **VAE(Variational Autoencoder)** 로 재구성 오차 기반 이상탐지를 수행하고, **XAI(피처별 기여도)** 로 어떤 센서가 공격받았는지 식별합니다. 동시에, 방어 성능을 정량 평가하기 위한 **Red Agent 자동 공격 캠페인**을 함께 구현했습니다.

---

## 시스템 구조

각 노드는 ROS2(rclpy) 노드로 동작하며, `haegeum_interfaces` 커스텀 메시지가 없는 환경에서는 자동으로 시뮬레이션 모드로 폴백합니다.

```
                        ┌─────────────────────────────────────────────┐
                        │            Red Agent (공격 AI)               │
                        │   red_agent.py · advanced_attacks.py         │
                        │   RECON → EW → SWARM → PAYLOAD (4단계)        │
                        └───────────────┬─────────────────────────────┘
                                        │ 위조 센서/레이더/허위표적 주입
                                        ▼
   ┌────────────────┐        ┌────────────────────────┐
   │ sensor_publisher│──정상─▶│  /sensor_data/{robot}  │
   │  (정상 패턴)     │        │  /radar_tracks         │
   └────────────────┘        └───────────┬────────────┘
                                          ▼
                        ┌──────────────────────────────────────┐
                        │   anomaly_detector.py  (L1 방어)      │
                        │   ① VAE 센서 이상탐지 (GPS/IMU)        │
                        │   ② 레이더 트랙 군집 탐지 (SWARM)      │
                        │   ③ XAI 피처 기여도 → 공격유형 분류    │
                        └───────────────┬──────────────────────┘
                                        │ /target_detected (Event)
                                        ▼
                        ┌──────────────────────────────────────┐
                        │   target_fusion.py  (L2 엣지)         │
                        │   VAE(0.50)+Radar(0.35)+Camera(0.15)  │
                        │   가중 신뢰도 융합 → /fused_targets    │
                        └───────────────┬──────────────────────┘
                                        ▼
                        ┌──────────────────────────────────────┐
                        │  ai_commander (외부) → /assignment    │
                        │  ugv_agent.py / ugv_patrol.py 로 임무  │
                        └──────────────────────────────────────┘
```

![시스템 아키텍처](docs/images/architecture_diagram.png)

### VAE 구조 (anomaly_detector.py / advanced_attacks.py 기준)

| 항목 | 값 |
|---|---|
| 입력 | 슬라이딩 윈도우 20 타임스텝 × 피처 수 (UAV 8 → 160차원, UGV 10 → 200차원) |
| 인코더 | Linear(input→128) → ReLU → Linear(128→64) → ReLU |
| Latent | `latent_dim = 16` (mu / logvar 분기, reparameterization) |
| 디코더 | Linear(16→64) → ReLU → Linear(64→128) → ReLU → Linear(128→input) |
| 이상 점수 | 재구성 MSE `mean((recon − x)²)` |
| 임계값 | **정상 검증셋 재구성 오차의 95th percentile** (calib_size=200 온라인 보정) |
| XAI | 윈도우 내 피처별 재구성 오차 평균 → argmax 피처를 `FEATURE_TO_ATTACK` 매핑으로 공격유형 분류 |

- **UAV 피처(8)**: `gps_vel_x/y`, `imu_ax/ay`, `residual_x/y`, `baro_alt`, `pitch_rate`
- **UGV 피처(10)**: `gps_vel_x/y`, `imu_ax/ay`, `residual_x/y`, `wheel_vel_l/r`, `cmd_vel_x/w`
- **핵심 아이디어**: `residual = GPS속도 − IMU가속도`. 정상 주행/비행에서는 ≈0 이지만, GPS만 위조되면 residual이 폭발 → 스푸핑을 물리 정합성 위반으로 탐지.

### 센서 융합 로직 (target_fusion.py)

- 3개 소스를 **가중 평균 신뢰도 융합**: VAE 0.50 / 레이더 0.35 / 카메라 0.15
- 동일 표적 판단: 유클리드 거리 `< 20m`(`FUSION_RADIUS`) 이면 동일 표적으로 병합
- 레이더 트랙 `[track_id, range_m, az_deg, el_deg, radial_vel, rcs_dbsm]` 을 극좌표→직교좌표 변환, RCS 기반 신뢰도 산정
- 5초 미갱신 표적은 stale 제거

---

## 위협 모델 / 공격 시나리오

Red Agent는 APT(지능형 지속 위협) 구조를 모사한 **4단계 캠페인**을 10Hz 루프로 자동 전개합니다. (`red_agent.py`)

| 단계 | 이름 | 공격 벡터 | 주입 파라미터 | MITRE ATT&CK for ICS |
|---|---|---|---|---|
| Phase 1 | RECON (정찰) | 저고도(15m)·저속(8m/s) 스카우트 드론 1대, 낮은 RCS(≈−10 dBsm) | `/radar_tracks` 단일 트랙 | T0842 Network Sniffing |
| Phase 2 | EW (전자전) | GPS 스푸핑 + 고도 스푸핑 + 명령 주입 | `gps_vel ±12m/s`, `baro_alt +25m`, `pitch_rate` 진동 | T0830 AiTM · T0855 Unauthorized Command |
| Phase 3 | SWARM (포화) | 군집 드론 6대, 360°/60° 간격 동시 접근(20m/s), 새떼 위장 RCS | `/radar_tracks` 다중 트랙 + 허위표적 flood | T0814 Denial of Service |
| Phase 4 | PAYLOAD (페이로드) | 우군 UGV `cmd_vel` 역방향 위조 → 임무 구역 이탈 | `cmd_vel = −1.0` | T0803 Block Command Message |

### 방어 관점 STANAG 4586 시나리오 커버리지

| 시나리오 | 탐지 피처 | 탐지 소스 |
|---|---|---|
| G-4 UGV 센서 스푸핑 | `residual_x/y` 폭발 | VAE |
| A-4 데이터링크 하이재킹 | `baro_alt`, `pitch_rate` | VAE |
| G-1 UGV 페이로드 탈취 | `wheel_vel`, `cmd_vel` | VAE |
| J-3 가용성 보전 | 낮은 오탐(FP) 유지 | VAE |
| SWARM 군집 재밍 | 동시 트랙 수 ≥ 3 & 접근속도 ≥ 15 m/s | 레이더 |

### 심화 실험 공격 유형 (advanced_attacks.py)

예선 이후 본선 추가 연구로, 단일 공격을 넘어선 **복합·점진적 공격**의 탐지 강건성을 검증했습니다.

- **복합 공격**: GPS+명령주입, GPS+고도 (두 센서 동시 오염)
- **점진적 스푸핑**: 값을 선형 램프로 서서히 증가시켜 임계값을 우회하려는 시도 (rate 0.1~0.5)
- **FP/FN 피처 분포 분석**: 오탐이 발생한 윈도우에서 어떤 피처가 정상 분포 대비 몇 σ 벗어났는지 정량화

---

## 실험 결과

### 1) 기본 3종 공격 — 혼동행렬 기반 (UAV/UGV 노트북)

아래 수치는 `UAV_anomaly_vae_SEAD.ipynb` / `UGV_anomaly_vae.ipynb` 의 혼동행렬 이미지에서 직접 산출한 값입니다. **경계(전이) 윈도우를 제외하지 않은** 전체 윈도우 기준이라, 공격 시작 직후 윈도우에 정상 데이터가 섞여 있는 구간이 미탐(FN)으로 집계됩니다.

**UAV (전체, TN=186 / FP=10 / FN=240 / TP=600)**

| 공격 유형 | Precision | Recall | F1 |
|---|---|---|---|
| GPS Spoofing | 0.953 | 0.718 | 0.819 |
| Altitude Spoofing | 0.952 | 0.714 | 0.816 |
| Command Injection | 0.952 | 0.711 | 0.814 |
| **전체(All)** | **0.984** | **0.714** | **0.828** |

**UGV (전체, TN=186 / FP=10 / FN=105 / TP=735)**

| 공격 유형 | Precision | Recall | F1 |
|---|---|---|---|
| GPS Spoofing | 0.954 | 0.739 | 0.833 |
| Wheel Slip | 0.963 | 0.918 | 0.940 |
| Cmd Anomaly | 0.964 | 0.968 | 0.966 |
| **전체(All)** | **0.987** | **0.875** | **0.927** |

| UAV 혼동행렬 | UGV 혼동행렬 |
|---|---|
| ![UAV 혼동행렬](docs/images/uav_confusion_matrix.png) | ![UGV 혼동행렬](docs/images/ugv_confusion_matrix.png) |

| UAV 피처별 기여도 (XAI) | UGV 피처별 기여도 (XAI) |
|---|---|
| ![UAV 피처 기여도](docs/images/uav_feature_contribution.png) | ![UGV 피처 기여도](docs/images/ugv_feature_contribution.png) |

### 2) 복합·점진적 공격 — 경계 윈도우 제외 평가 (advanced_attacks.py)

`advanced_attack_results.json` 기준. 공격 시작 후 19개 경계 윈도우를 평가에서 제외하고, 정상 100 + 공격 200 타임스텝(UAV)에 대해 측정한 값입니다. 임계값 = **1.2283** (정상 95th percentile).

| 공격 유형 | TP | FP | FN | Precision | Recall | F1 | 탐지 지연 |
|---|---|---|---|---|---|---|---|
| GPS Spoofing (예선) | 181 | 8 | 0 | 0.958 | 1.000 | 0.978 | 100 ms |
| Altitude Spoofing (예선) | 181 | 3 | 0 | 0.984 | 1.000 | 0.992 | 0 ms |
| Command Injection (예선) | 181 | 3 | 0 | 0.984 | 1.000 | 0.992 | 0 ms |
| 복합 A: GPS+CmdInj | 181 | 5 | 0 | 0.973 | 1.000 | 0.986 | 0 ms |
| 복합 B: GPS+Alt | 181 | 6 | 0 | 0.968 | 1.000 | 0.984 | 0 ms |
| 점진적 GPS (rate=0.3) | 181 | 5 | 0 | 0.973 | 1.000 | 0.986 | 200 ms |
| 점진적 GPS (rate=0.1) | 181 | 6 | 0 | 0.968 | 1.000 | 0.984 | 400 ms |
| 점진적 Alt (rate=0.5) | 181 | 5 | 0 | 0.973 | 1.000 | 0.986 | 300 ms |
| 점진적 Alt (rate=0.2) | 181 | 6 | 0 | 0.968 | 1.000 | 0.984 | 500 ms |

**핵심 관찰**
- 경계 구간을 제외하면 모든 공격에서 **미탐(FN) 0** — 일단 윈도우가 공격 데이터로 채워지면 100% 탐지.
- **점진적 스푸핑일수록 탐지 지연 증가**(rate 0.1 → 400 ms, rate 0.2 → 500 ms). 값을 천천히 올려 임계값을 우회하려는 공격은 탐지가 늦어짐을 정량 확인.
- 복합 공격(두 센서 동시)은 단일 공격 대비 탐지 성능 저하 없음.
- 남은 오탐(FP)은 대부분 `residual_x`, `pitch_rate` 등 분산이 큰 피처에서 정상 분포 2~3σ 지점에 발생.

| 기본/복합 공격 이상 점수 | 점진적 공격 탐지 지연 | FP/FN 피처 분포 |
|---|---|---|
| ![기본 공격](docs/images/result_original_attacks.png) | ![점진적 공격](docs/images/result_gradual_attacks.png) | ![FP/FN 분석](docs/images/result_fpfn_analysis.png) |

![복합 공격](docs/images/result_composite_attacks.png)

> 두 평가(1·2)의 Recall 차이는 **경계 윈도우 포함 여부**에서 비롯됩니다. 노트북은 전이 구간을 포함(보수적), 심화 실험은 제외(정상 상태 성능)한 결과입니다.

---

## 실시간 추론 성능

오프라인 배치 실험만으로는 "온보드에서 제때 돌아가는가"를 답할 수 없어, 기존 VAE와
가중치를 그대로 로드해 CPU 추론 지연시간을 실측했습니다. 전체 수치·방법은
[`benchmarks/README.md`](benchmarks/README.md), 원시 결과는
[`benchmarks/results.json`](benchmarks/results.json).

**측정 환경**: Apple M5 (10코어) / macOS 26.4 arm64 / Python 3.14.3 / PyTorch 2.11.0 CPU /
`torch.set_num_threads(1)` / 워밍업 300회 + 측정 2,000회 / 시드 42.
1회 측정 구간은 `AnomalyDetector._infer()` 와 동일한 **VAE forward + 재구성 MSE** 입니다.

### 단일 윈도우 지연시간 / 처리량

| 플랫폼 | 정밀도 | 배치 | p50 (ms) | p95 (ms) | p99 (ms) | 처리량 (windows/sec) |
|---|---|---|---|---|---|---|
| UAV (input 160) | FP32 | 1 | 0.032 | 0.062 | 0.119 | 27,338 |
| UAV | FP32 | 8 | 0.042 | 0.045 | 0.051 | 190,017 |
| UAV | FP32 | 32 | 0.052 | 0.057 | 0.071 | 607,056 |
| UAV | INT8 | 1 | 0.123 | 0.132 | 0.153 | 8,068 |
| UGV (input 200) | FP32 | 1 | 0.033 | 0.036 | 0.041 | 29,593 |
| UGV | FP32 | 8 | 0.043 | 0.047 | 0.051 | 182,876 |
| UGV | FP32 | 32 | 0.053 | 0.056 | 0.061 | 599,949 |
| UGV | INT8 | 1 | 0.123 | 0.130 | 0.148 | 8,044 |

![지연시간 분포](docs/images/latency_distribution.png)
![배치별 처리량](docs/images/latency_batch_throughput.png)

### 센서 주기 대비 실시간성 판단

`sensor_publisher.py` 의 발행 주기는 `rate_hz` 기본값 **10 Hz**, 즉 샘플당 예산 **100 ms** 입니다.

| 플랫폼 | 센서 주기 | p99 지연 (FP32) | 예산 사용률 | 여유 | 지속 가능 최대 주기 |
|---|---|---|---|---|---|
| UAV | 100 ms | 0.119 ms | 0.12 % | 839× | 8,393 Hz |
| UGV | 100 ms | 0.041 ms | 0.04 % | 2,431× | 24,313 Hz |

**실시간 처리 가능.** 최악(p99)에서도 주기의 0.12 % 만 사용하며, 단일 스레드로 UAV·UGV를
동시에 돌려도 여유가 3자리수 배입니다. Docker(linux/arm64) 컨테이너에서 재측정했을 때도
p99 는 1.91 ms(UAV) / 2.86 ms(UGV)로 예산의 3 % 미만이었습니다.
다만 **추론 지연 ≠ 탐지 지연**입니다. 실제 반응 시간은 슬라이딩 윈도우가 공격 데이터로
채워지는 시간(최대 20 샘플 = 2 s @10 Hz)과 위 점진적 스푸핑 실험의 탐지 지연(최대 500 ms)이
지배하며, ROS2/DDS 전송 지연은 포함되지 않았습니다.

### 경량화 (동적 양자화, qint8)

`torch.ao.quantization.quantize_dynamic(model, {nn.Linear}, qint8)` 적용 결과.

| 항목 | UAV | UGV |
|---|---|---|
| 모델 크기 (FP32 → INT8) | 244 KB → 71 KB (**-70.9 %**) | 284 KB → 81 KB (**-71.4 %**) |
| p50 지연 (batch=1, 호스트) | 0.032 → 0.123 ms (**3.8× 느려짐**) | 0.033 → 0.123 ms (**3.7× 느려짐**) |
| p50 지연 (batch=1, 컨테이너) | 0.163 → 0.100 ms (**1.6× 빨라짐**) | 0.178 → 0.114 ms (**1.6× 빨라짐**) |
| 정상 점수 p95 (임계값) | 1.2301 → 1.2293 | 1.1884 → 1.1884 |
| FP32-INT8 점수 상관 | Pearson r = 1.0000 | Pearson r = 1.0000 |
| 기존 임계값에서의 F1 | 0.9891 → 0.9891 (5개 공격 전부) | 0.9856 → 0.9856 (Wheel Slip / Cmd Anomaly) |

- **탐지 성능은 그대로**입니다. 재구성 오차 분포가 유지되므로 임계값 재보정 없이 양자화
  모델을 투입할 수 있습니다.
- **지연시간 이득은 환경에 따라 부호가 바뀝니다.** 행렬이 작아(최대 200×128) 양자화
  커널 오버헤드가 연산 절감을 넘어서는 경우가 있습니다. 이 규모에서 양자화의 근거는
  속도가 아니라 **메모리/플래시 71 % 절감**이며, 실제 타깃 보드에서 재확인이 필요합니다.

---

## 설명가능성 (SHAP)

"왜 이 경보가 울렸는가"를 운용자가 확인할 수 있어야 오탐 판단과 사후 검증이 가능합니다.
재구성 오차를 출력하는 함수를 `shap.KernelExplainer` 로 분해해, 어느 센서가 이상 점수를
만들었는지 정량화했습니다. 상세: [`explainability/README.md`](explainability/README.md).

- 설명 대상: `윈도우 → 재구성 오차(anomaly score)` 스칼라 함수 (모델 출력이 아님)
- 입력 20 타임스텝 × N 피처의 SHAP 값을 타임스텝 축으로 합산 → 센서 단위 기여도
- 배경 분포 `shap.kmeans(정상 윈도우, 30)`, `nsamples=8192`, 조건별 상위 10개 윈도우

| 플랫폼 | 조건 | SHAP 상위 3 | 기존 재구성오차 상위 3 | Spearman ρ |
|---|---|---|---|---|
| UAV | GPS Spoofing | residual_x, residual_y, gps_vel_x | 동일 | 1.00 |
| UAV | Altitude Spoofing | baro_alt, imu_ay, gps_vel_y | baro_alt, gps_vel_y, imu_ay | 0.88 |
| UAV | Command Injection | pitch_rate, gps_vel_y, imu_ay | 동일 | 0.93 |
| UGV | GPS Spoofing | residual_x, gps_vel_x, residual_y | residual_y, cmd_vel_x, gps_vel_y | **-0.08** |
| UGV | Wheel Slip | wheel_vel_l, wheel_vel_r, residual_x | 동일 | 1.00 |
| UGV | Command Anomaly | wheel_vel_l, wheel_vel_r, cmd_vel_x | 동일 | 0.94 |

![UAV SHAP 기여도](docs/images/shap_attack_contribution_uav.png)
![UGV SHAP 기여도](docs/images/shap_attack_contribution_ugv.png)

| UAV SHAP summary | UGV SHAP summary |
|---|---|
| ![UAV SHAP summary](docs/images/shap_summary_uav.png) | ![UGV SHAP summary](docs/images/shap_summary_ugv.png) |

**공격 유형별 서명이 분리**되며, `anomaly_detector.py` 의 `FEATURE_TO_ATTACK` 매핑
(residual/gps → GPS_SPOOFING, baro_alt → ALTITUDE_SPOOFING, pitch_rate·휠 →
COMMAND_INJECTION)이 SHAP 기준으로도 근거가 있음을 확인했습니다.

**기존 기여도 이미지와의 대조 (정직한 기록)**

- **일치**: UAV GPS Spoofing / Altitude Spoofing, UGV Wheel Slip / Command Anomaly —
  상위 피처가 그대로 재현됩니다.
- **부분 불일치 (원인 규명)**: UAV Command Injection 은 기존 이미지에서 `imu_ax` 가 1위,
  SHAP 에서는 `pitch_rate` 가 1위입니다. 노트북 생성기는 `pitch_rate` + `imu_ax/ay` 를 함께
  교란하고, 본 분석이 쓴 `advanced_attacks.atk_cmd_injection()` 은 `pitch_rate` 만 교란하기
  때문입니다. 설명이 갈린 게 아니라 **설명 대상 데이터가 다릅니다.**
- **불일치 (기존 결과가 신뢰 불가)**: UGV GPS Spoofing 의 기존 패널은 가로축이 1e14 이고
  휠 속도를 1위로 지목합니다. ① 노트북이 정상(n=1000)과 공격(n=300) 시계열을 섞어 쓰면서
  정규화 기준이 어긋났고(`generate_normal` 이 `linspace(0,10,n)` 의 gradient 로 속도를
  만들기 때문), ② 점수가 극단적으로 튀는 구간에서는 재구성 오차가 10개 피처 전체에
  거의 균일하게 퍼져 "최대 오차 피처" 휴리스틱이 무의미해지기 때문입니다. 같은 윈도우에
  대해 SHAP 은 실제 주입 지점(`residual_x`, `gps_vel_x`, `residual_y`)을 정확히 지목했습니다.

→ 중간 강도 이상까지는 온보드의 가벼운 휴리스틱으로 충분하지만, **점수가 임계값의 수백 배를
넘는 강한 공격에서는 휴리스틱의 피처 지목이 무너집니다.** 온보드는 휴리스틱을 유지하고
지상국에서 SHAP 으로 원인 센서를 확정하는 2단계 설명 구성을 권장합니다.

---

## 재현

```bash
# 로컬 (CPU)
make setup        # requirements.txt 설치
make benchmark    # 지연시간/처리량/양자화 벤치마크 → benchmarks/results.json
make explain      # SHAP 설명가능성 분석 → explainability/shap_results.json
make attacks      # 기존 복합·점진적 공격 실험 재실행
make reproduce    # benchmark + explain 전체

# Docker (CPU 전용 이미지, ROS2 불필요)
make docker-build           # = docker build -t haegeum-addon:cpu .
make docker-run             # 컨테이너에서 make reproduce
```

- 난수 시드는 `utils/seed.py` 의 `set_seed(42)` 로 python/numpy/torch 를 함께 고정합니다.
- `benchmarks/` 와 `explainability/` 는 ROS2 없이 동작합니다. `utils/load_model.py` 가
  rclpy 가 없을 때만 최소 스텁을 주입해 **`anomaly_detector.py` 원본을 수정하지 않고**
  동일한 `VAE` 클래스와 `vae_*.pth` 가중치를 로드합니다.
- 지연시간은 하드웨어·부하에 따라 달라지므로, 재측정 시 `results.json` 의
  `meta.environment` 와 함께 읽어야 합니다.

---

## 저장소 구조

```
haegeum-addon/
├── anomaly_detector.py        # L1 VAE 이상탐지 ROS2 노드 (GPS/IMU + 레이더 융합)
├── target_fusion.py           # L2 다중 소스(레이더/VAE/카메라) 가중 융합 노드
├── red_agent.py               # 4단계 자동 공격 캠페인 (RECON→EW→SWARM→PAYLOAD)
├── advanced_attacks.py        # 복합·점진적 공격 탐지 심화 실험 (standalone)
├── sensor_publisher.py        # 정상 센서 패턴 발행 노드 (테스트용)
├── ugv_agent.py               # UGV 상태 발행 / 임무 수신 노드
├── ugv_patrol.py              # UGV 순찰 웨이포인트 관리
├── UGV_anomaly_vae.ipynb      # UGV VAE 학습·평가 노트북 (Colab)
├── UAV_anomaly_vae_SEAD.ipynb # UAV VAE 학습·평가 노트북 (Colab)
├── advanced_attack_results.json  # 심화 실험 정량 결과
├── benchmarks/
│   ├── latency_benchmark.py   # CPU 추론 지연시간·처리량·양자화 벤치마크
│   ├── results.json           # 벤치마크 실측 원시 결과
│   └── README.md              # 측정 방법·환경·결과 표
├── explainability/
│   ├── shap_analysis.py       # 재구성 오차에 대한 SHAP 기여도 분석
│   ├── shap_results.json      # SHAP 실측 원시 결과
│   └── README.md              # 방산 체계에서의 필요성·기존 결과 대조
├── utils/
│   ├── seed.py                # 난수 시드 고정 + 측정 환경 기록
│   ├── load_model.py          # 기존 anomaly_detector.VAE·가중치 로더 (ROS2 스텁)
│   └── ugv_data.py            # UGV 합성 센서/공격 데이터 생성기
├── Dockerfile                 # CPU 재현 이미지 (ROS2 불필요)
├── Makefile                   # setup / benchmark / explain / reproduce
├── requirements.txt           # 버전 고정 의존성
├── vae_uav.pth                # UAV VAE 가중치 (input=160)
├── vae_ugv.pth                # UGV VAE 가중치 (input=200)
└── docs/images/               # 결과 그래프·혼동행렬·아키텍처 다이어그램
```

---

## 실행 방법

### 심화 실험 (ROS2 불필요, 바로 실행 가능)

```bash
pip install torch numpy matplotlib
python advanced_attacks.py
# → docs 결과 PNG 3종 + advanced_attack_results.json 재생성
```

### ROS2 노드 (haegeum 통합 환경)

```bash
# 의존성
pip install torch numpy   # + ROS2 rclpy 환경

# 방어: anomaly_detector (UGV / UAV)
ros2 run haegeum_addon anomaly_detector --ros-args -p platform:=ugv -p robot_id:=ugv_01
ros2 run haegeum_addon anomaly_detector --ros-args -p platform:=uav -p robot_id:=uav_01

# 융합
ros2 run haegeum_addon target_fusion

# 공격 시뮬레이션
ros2 run haegeum_addon red_agent --ros-args -p swarm_size:=6

# 정상 센서 발행 (테스트)
ros2 run haegeum_addon sensor_publisher --ros-args -p platform:=ugv
```

> `haegeum_interfaces` 커스텀 메시지가 없으면 각 노드는 콘솔 출력 시뮬레이션 모드로 동작합니다.

### VAE 재학습

`UGV_anomaly_vae.ipynb` / `UAV_anomaly_vae_SEAD.ipynb` 를 Google Colab에서 실행하면 합성 데이터로 VAE를 학습하고 `vae_*.pth` 를 생성합니다.

---

## 기술 스택

- **딥러닝**: PyTorch (VAE, reparameterization trick)
- **로보틱스 미들웨어**: ROS2 (rclpy, Float32MultiArray, 커스텀 msg)
- **수치/시각화**: NumPy, Matplotlib
- **XAI**: 피처별 재구성 오차 기여도 분석, SHAP (KernelExplainer)
- **경량화/프로파일링**: PyTorch 동적 양자화(qint8), 지연시간 분위수(p50/p95/p99) 측정
- **재현성**: Docker (CPU), Makefile, 버전 고정 requirements, 시드 고정 유틸
- **위협 모델링**: MITRE ATT&CK for ICS, STANAG 4586 시나리오 매핑
- **평가**: Precision / Recall / F1, 탐지 지연(ms), 혼동행렬, FP/FN 피처 분포

---

## 한계 및 향후 과제

- **합성 데이터 기반**: 정상/공격 데이터를 시뮬레이션으로 생성했습니다. 실제 GPS 스푸핑 공개 데이터셋(예: Mendeley DOI 10.17632/z7dj3yyzt8.3)으로 교체·검증이 필요합니다.
- **경계 구간 탐지 지연**: 슬라이딩 윈도우 특성상 공격 시작 직후 윈도우에는 정상 데이터가 섞여 탐지가 지연됩니다. 다중 윈도우/멀티스케일 탐지로 개선 여지가 있습니다.
- **점진적 스푸핑 취약성**: 값을 아주 천천히 올리는 공격에서 지연이 커집니다(최대 500 ms 관측). 온라인 임계값 적응이나 변화율(rate) 기반 보조 탐지가 필요합니다.
- **오탐 관리**: 분산이 큰 피처(`residual`, `pitch_rate`)에서 소수의 오탐이 남습니다. 피처별 정규화·robust threshold로 저감 가능.
- **레이더 융합 미검증**: 레이더/카메라 융합 로직은 구현되어 있으나 정량 평가는 향후 과제입니다.
- **지연시간 측정 범위**: 추론 커널만 측정했습니다. ROS2/DDS 전송 지연, 실제 온보드 SoC
  (Jetson 계열 등)에서의 재측정, 다중 노드 동시 구동 시의 간섭은 포함되지 않았습니다.
- **강한 공격에서의 설명 붕괴**: 점수가 임계값의 수백 배를 넘는 구간에서 기존 피처별
  재구성 오차 휴리스틱의 원인 지목이 무너지는 것을 SHAP 대조로 확인했습니다
  (`explainability/README.md`). 온보드 경량 설명 기법의 개선이 필요합니다.

---

## 참고

- MITRE ATT&CK for ICS — https://attack.mitre.org/matrices/ics/
- STANAG 4586 — UAV 지상통제 스테이션 인터페이스 표준
- 본 저장소는 DAH 2026 팀 **해금**의 통합 시스템 중 **L1 방어 + Red Agent 모듈** 부분입니다.
