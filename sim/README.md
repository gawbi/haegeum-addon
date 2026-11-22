# sim/ — MuJoCo 기반 UGV 물리 시뮬레이션 + 센서 모델 + GPS 스푸핑 주입

## 1. 이게 무엇인가

프로젝트의 VAE 이상탐지기가 쓰는 **UGV 10개 피처**를 난수가 아니라 **물리 시뮬레이션**에서 생성하는 최소 구성의 데모 환경이다.

- 물리 엔진: MuJoCo 3.16 (500 Hz, timestep 0.002 s)
- 차량: 차동구동(differential-drive) UGV — 섀시 + 구동륜 2개 + 전방 캐스터 1개
- 임무: 20 × 20 m 모서리 깎은 사각 루프 위 웨이포인트 8개를 무한 순환, 순항 약 1.45 m/s
- 센서 모델: GPS 위치/속도, IMU 가속도, 바퀴 엔코더, 컴파스(헤딩) — 각각 가우시안 잡음 추가
- 공격 주입: **측정 계층**에서 GPS를 변조하고, 그 변조된 값으로 **제어기가 자기 위치를 추정**하므로 차량이 물리적으로 잘못된 경로로 주행한다
- 출력: 10 Hz 피처 CSV + ground truth, 궤적/피처 PNG, 탑뷰+추적캠 MP4 데모 영상

피처 순서(프로젝트 공통, `sensor_publisher.py`와 동일):

```
gps_vel_x, gps_vel_y, imu_ax, imu_ay, residual_x, residual_y,
wheel_vel_l, wheel_vel_r, cmd_vel_x, cmd_vel_w
```

확장 피처 2개(`FEATURES_EXT = FEATURES + [pos_resid_x, pos_resid_y]`, 10개 뒤에 추가 → 12차원). 10개만 쓰면 위치 수준 GPS 스푸핑을 거의 탐지하지 못하는 문제를 보완한다 (4절 수식, `docs_feature_comparison.md` 비교 결과).

## 2. 왜 물리 기반 데이터가 필요한가

현재 저장소의 "센서 데이터"(`advanced_attacks.py`의 `gen_normal` 등)는 `uniform(1,3)` 같은 **독립 난수**다. 여기에는 다음이 없다.

| 항목 | 난수 합성 데이터 | 물리 시뮬레이션 데이터 |
|---|---|---|
| 피처 간 인과 관계 | 없음 (각 열 독립) | 있음: cmd_vel → 바퀴 속도 → 가속도 → 속도 → 위치 |
| 시간 상관 | 없음 (샘플 i.i.d.) | 있음: 가속/감속/회전 구간이 연속적으로 이어짐 |
| 주행 패턴 | 없음 | 직진·코너링·감속이 루프 주기로 반복 |
| 공격의 물리적 결과 | 없음 (숫자만 바뀜) | 있음: 스푸핑된 GPS로 조향 → 실제 궤적 이탈 |
| 잔차(residual)의 의미 | `speed - imu` 산술 | GPS 속도 vs IMU dead-reckoning 추정의 실제 불일치 |

VAE는 "정상 패턴의 결합 분포"를 학습하는 모델이므로, 피처 간 상관과 시간 구조가 없는 난수로는 **무엇을 정상으로 배웠는지 설명할 수 없다**. 물리 기반 데이터에서는 "코너링 중 imu_ay 상승 + cmd_vel_w 포화 + 좌우 바퀴 속도 차"처럼 정상 패턴이 실제로 존재하고, 공격은 그 패턴을 깨뜨린다. 또한 공격이 **차량 거동을 실제로 바꾸는지**를 영상으로 보여줄 수 있다.

## 3. 아키텍처

```
┌──────────────────────┐   500 Hz   ┌──────────────────────┐
│  MuJoCo 물리 엔진     │ ─────────▶ │  진실 상태            │
│  (ugv_model.xml)      │            │  pos, vel, quat,     │
│  섀시/바퀴/캐스터/지면 │            │  가속도계, 바퀴 qvel  │
└──────────▲───────────┘            └──────────┬───────────┘
           │                                   │ 10 Hz (50 스텝마다)
           │                                   ▼
           │                        ┌──────────────────────┐
           │                        │  센서 모델            │
           │                        │  + 가우시안 잡음       │
           │                        │  gps_x/y, gps_vx/vy, │
           │                        │  imu_ax/ay, yaw,     │
           │                        │  wheel_vel_l/r       │
           │                        └──────────┬───────────┘
           │                                   ▼
           │                        ┌──────────────────────┐
           │                        │  공격 주입 (측정 계층) │  attack(t, meas) -> meas
           │                        │  gps_spoof_bias      │
           │                        │  gps_spoof_gradual   │
           │                        │  gps_freeze          │
           │                        └──────────┬───────────┘
           │                                   ▼
           │                        ┌──────────────────────┐      ┌─────────────────────┐
           │                        │  웨이포인트 제어기     │      │  잔차 계산            │
           │                        │  (변조된 GPS 위치 +   │      │  residual = gps_vel  │
           │                        │   컴파스 헤딩만 사용)  │      │   − IMU 적분 추정     │
           │                        └──────────┬───────────┘      └──────────┬──────────┘
           │                                   │ (cmd_vel_x, cmd_vel_w)       │
           │   바퀴 목표 각속도 (차동 믹싱)       ▼                             ▼
           └────────────────────── 모터 속도 서보 ──────▶  10개 피처 + GT + label → CSV
```

핵심은 제어기가 **ground truth를 전혀 보지 않는다**는 점이다. 제어기 입력은 (스푸핑 가능한) GPS 위치와 컴파스 헤딩뿐이므로, GPS 위치에 +4 m 편향을 주면 제어기는 "4 m 앞서 있다"고 믿고 조향하여 **실제 궤적이 −4 m 평행 이동**한다 (`out/trajectory_gps_spoof_bias.png`).

## 4. 센서 모델 수식

10 Hz, `dt = 0.1 s`. `N(0, σ)`는 독립 가우시안 잡음.

| 측정 | 수식 | 기본 σ |
|---|---|---|
| gps_x, gps_y | 진실 위치 + N(0, σ_gps_pos) | 0.10 m |
| gps_vel_x, gps_vel_y | 진실 월드좌표 속도 + N(0, σ_gps_vel) | 0.05 m/s |
| imu_ax, imu_ay | 섀시 IMU 가속도계 body-frame x/y (1차 LPF α=0.3로 접촉 진동 억제) + N(0, σ_imu). 수평 지면에서 중력은 z축에만 실리므로 x/y는 그대로 운동 가속도 | 0.05 m/s² |
| yaw (컴파스) | 진실 yaw + N(0, σ_yaw) | 0.01 rad |
| wheel_vel_l, wheel_vel_r | 힌지 조인트 각속도 × 바퀴 반지름(0.20 m) + N(0, σ_wheel) | 0.02 m/s |
| cmd_vel_x, cmd_vel_w | 제어기 출력 (잡음 없음) | — |

잔차 (상보 필터식 dead-reckoning, `UGVSim.residual`):

```
a_world = R(yaw) · [imu_ax, imu_ay]              # body → world 회전
v_est   = 0.98 · (v_est + a_world · dt) + 0.02 · gps_vel
residual = gps_vel − v_est
```

IMU 적분(98 %)에 GPS를 2 %씩 섞어 드리프트를 천천히 보정한다 (시정수 ≈ 5 s). GPS 속도가 갑자기 바뀌면(스푸핑/동결) v_est가 따라가는 데 수 초가 걸리므로 그 동안 residual에 불일치가 드러난다.

위치 잔차 (바퀴 오도메트리 dead-reckoning, `UGVSim.pos_residual`, 확장 피처):

```
v_odo     = (wheel_vel_l + wheel_vel_r) / 2                      # 잡음 포함 측정값
odom_pos  = (1 − α) · (odom_pos + R(yaw) · [v_odo, 0] · dt) + α · gps_pos     # α = 0.01 (10 Hz → 시정수 1/α = 100샘플 ≈ 10 s)
pos_resid = gps_pos − odom_pos                                   # (pos_resid_x, pos_resid_y)
```

`odom_pos`는 첫 GPS 측정으로 초기화하고, 이후 바퀴 속도·컴파스 헤딩만으로 적분하되 (스푸핑될 수 있는) GPS 위치에 1 %씩 앵커링해 정상 주행에서는 잡음 수준(σ ≈ 0.14 m)으로 유지한다. GPS 위치가 바퀴 적분과 어긋나면 잔차가 즉시 그 차이만큼 뛰고, 앵커 때문에 ≈ 10 s 시정수로 다시 수렴한다. 공격별 거동(α = 0.01, dt = 0.1 s): 고정 편향 b → b로 점프 후 e^(−t/10 s) 감쇠; 점진 드리프트 rate → rate·dt/α = rate × 10 s 에 수렴 (0.25 m/s → 2.5 m); 동결 → v·dt/α ≈ 15 m 까지 성장. 즉 **시정수(10 s)보다 훨씬 느린 편향은 이 피처에도 숨을 수 있다.**

제어기 (`UGVSim.controller`):

```
pos  = (gps_x, gps_y)                       # 변조될 수 있는 값
err  = wrap(atan2(wp − pos) − yaw)
cmd_vel_w = clip(1.8 · err, ±1.2 rad/s)
cmd_vel_x = 1.5 · max(0.3, cos err)   m/s     # 큰 헤딩 오차에서 감속
|wp − pos| < 1.5 m 이면 다음 웨이포인트
```

차동 믹싱 → 모터: `ω_l = (v − w·L/2)/r`, `ω_r = (v + w·L/2)/r` (L = 0.68 m, r = 0.20 m). MuJoCo `velocity` 액추에이터가 `τ = kv·(ω_target − ω)` (|τ| ≤ 12 N·m)로 토크를 생성한다 — 실제 ESC의 속도 서보에 해당.

## 5. 공격 (측정 계층)

| 이름 | 동작 | 물리적 결과 |
|---|---|---|
| `gps_spoof_bias(start_t, bias_xy, vel_bias_xy=None)` | 위치에 상수 bias (기본 (4, 0) m), 속도에 상수 vel_bias (기본 bias 방향 0.3 m/s) | 실제 궤적이 −bias 방향으로 평행 이동 |
| `gps_spoof_gradual(start_t, rate_xy, max_offset=12)` | 위치가 rate (기본 (0.25, 0) m/s)로 drift, 속도에 rate가 더해짐 | 궤적이 서서히 밀려남 (60 s 데이터에서 최대 10 m) |
| `gps_freeze(start_t)` | 위치를 start_t 값으로 고정, 속도 0 보고 | 제어기가 같은 방향으로 계속 조향 → 루프 이탈 후 직진 (40 s 동안 ~60 m) |

주의할 점: 순수한 **위치** 편향은 10개 피처(모두 속도/가속도/명령 기반)에는 약하게만 나타난다. 피처에 보이는 것은 (a) 속도 편향 vel_bias, (b) 그로 인한 residual 과도응답, (c) 재조향 시 cmd_vel_w 변화뿐이다. 이는 시뮬레이션의 한계가 아니라 **현재 피처 집합의 탐지 한계**를 드러내는 결과다. 이를 보완한 것이 4절의 위치 잔차 `pos_resid_x/y`(GPS 위치 − 바퀴 오도메트리 위치)이며, 10개 vs 12개 피처의 VAE 탐지 성능 비교는 `docs_feature_comparison.md` / `out/feature_comparison.png` 참조.

## 6. 실행 방법

```bash
cd sim
uv venv .venv -p 3.14
uv pip install -p .venv/bin/python -r requirements.txt

.venv/bin/python ugv_sim.py            # 60 s 정상 주행 스모크 테스트 (통계 출력)
.venv/bin/python generate_datasets.py  # CSV + PNG 생성 (~15 s)
.venv/bin/python make_demo_video.py    # MP4 2개 렌더링 (~1 분)
.venv/bin/python make_demo_video.py --quick   # 짧은 버전 (5 s / 20 s)

# VAE 재학습·평가 (torch 필요 — 시스템 python3 사용)
python3 train_eval_sim_vae.py --feats 10   # 기존 10개 피처 (베이스라인) → out/*_f10.*
python3 train_eval_sim_vae.py --feats 12   # 10개 + pos_resid_x/y          → out/*_f12.*
python3 plot_feature_comparison.py         # 두 결과 비교 그림 → out/feature_comparison.png

# 연구 요약 모션그래픽 영상 (82 s, 1920×1080, ~30 s 렌더) — 위 결과 파일이 모두 있어야 함
python3 export_motion_data.py              # 제로데이 점수 시계열 → out/zeroday_scores.npz (torch 필요)
.venv/bin/python make_motion_video.py      # → out/motion_summary.mp4
.venv/bin/python make_motion_video.py --preview 13 44 73   # 지정 시각(초)만 PNG 미리보기
```

Python API:

```python
from ugv_sim import UGVSim, gps_spoof_bias
df = UGVSim(seed=42).run(60.0, attack=gps_spoof_bias(20.0, bias_xy=(4.0, 0.0)))
X = df[FEATURES].values       # (600, 10) → VAE 입력 (기존)
X12 = df[FEATURES_EXT].values # (600, 12) → 10개 + pos_resid_x/y
```

## 7. 출력 파일 (`sim/out/`)

| 파일 | 내용 |
|---|---|
| `normal_train.csv` | 정상 600 s, seed 42, 6000행 — VAE 학습용 |
| `normal_test.csv` | 정상 120 s, seed 123, 1200행 |
| `gps_spoof_bias.csv` / `gps_spoof_gradual.csv` / `gps_freeze.csv` | 각 60 s, 20 s부터 공격, 600행 (label 1 = 400행) |
| `trajectory_*.png` | 진실 궤적(파랑) vs GPS가 믿는 궤적(빨강) vs 웨이포인트(주황) |
| `features_*.png` | 12개 피처(10개 + pos_resid_x/y) 시계열, 공격 시작점 빨간 점선 |
| `demo_normal.mp4` | 20 s 정상 주행 (탑뷰 ∥ 추적캠, 1280×480, 20 fps) |
| `demo_gps_spoof.mp4` | 40 s, 15 s부터 bias (4, 0) m 스푸핑 — 섀시가 빨갛게 바뀌고 파란 실제 궤적이 빨간 GPS 궤적/주황 마커에서 4 m 벗어남 |
| `trajectory_gps_spoof.png` / `features_gps_spoof.png` | 위 영상(데모 스푸핑 주행)의 궤적 플롯 / 10개 피처 시계열 |
| `sim_vae_results_f10.json` / `_f12.json` | VAE 평가 결과 (임계값, 공격별 P/R/F1/지연/상위 기여 피처) — 10개 / 12개 피처 |
| `sim_vae_scores_f10.png` / `_f12.png`, `sim_vae_scores_f*.npz` | 공격별 이상 점수 시계열 그림 / 원본 배열 |
| `vae_ugv_sim_f10.pth` / `_f12.pth` | 학습된 VAE 가중치 |
| `feature_comparison.png` | 10개 vs 12개 피처 이상 점수 비교 (3 공격 × 2 변형) |
| `zeroday_scores.npz` | 제로데이 GPS 동결 / 전체 센서 동결 점수 시계열 (영상용, `zeroday_results.json`과 같은 보정 경로) |
| `motion_summary.mp4` | 연구 요약 모션그래픽 82 s — 출발점 → 시뮬레이션 구조 → 실제 화면 → 궤적 → 10 vs 12 피처 → 제로데이 → 정리. 모든 수치·곡선은 위 결과 파일에서 읽어 그림 |

CSV 열: `t`, 10개 피처, `pos_resid_x/y`, `true_x/y/z`, `true_vx/vy`, `true_yaw/roll/pitch`, `gps_x/y` (제어기가 믿는 위치), `odom_x/y` (바퀴 오도메트리 위치), `wp_idx`, `label`.

## 8. 한계

- **GPS/RF 물리 없음**: 위성 기하, 의사거리, 수신기 추적 루프, 신호 세기를 모델링하지 않는다. 스푸핑은 "수신기가 출력하는 위치/속도가 바뀐다"는 **측정 계층 효과**만 모사한다.
- **평지·무장애물**: 지면은 평면, 경사·요철·장애물 없음. 따라서 imu_ax/ay에 지형 성분이 없고 roll/pitch ≈ 0.
- **단순화된 차량**: 박스 섀시 + 실린더 바퀴 + 마찰 없는 구 캐스터, 서스펜션 없음, 모터는 이상적 속도 서보. 실제 UGV의 슬립, 모터 동역학, 배터리 효과는 없다.
- **단순 제어기**: 헤딩 P 제어 + 고정 순항 속도. 실제 자율주행 스택(EKF 융합, 경로 계획)이 아니므로 "EKF가 스푸핑을 일부 걸러내는" 효과는 반영되지 않는다.
- **잡음 모델**: 독립 가우시안만. GPS 다중경로, IMU 바이어스 drift, 엔코더 양자화 없음.
- **영상**: 오프스크린 렌더링 결과를 그대로 저장. 장면 내 궤적 점은 시각화용이며 물리에 영향 없음.

## 9. 파일 구성

```
sim/
├── ugv_model.xml        MJCF 모델 (차량, 지면, 마커, 센서, 카메라)
├── ugv_sim.py           UGVSim, 센서 모델(위치 잔차 포함), 공격 클래스, 제어기, 플롯 헬퍼
├── video.py             VideoRecorder (mujoco.Renderer → imageio MP4, 궤적/텍스트 오버레이)
├── generate_datasets.py CSV/PNG 생성
├── make_demo_video.py   데모 MP4 생성
├── train_eval_sim_vae.py  VAE 재학습·평가 (--feats 10|12)
├── plot_feature_comparison.py  10 vs 12 피처 비교 그림
├── docs_feature_comparison.md  위치 잔차 피처 추가 전/후 비교 문서
├── requirements.txt
└── out/                 생성물 (mp4는 .gitignore)
```
