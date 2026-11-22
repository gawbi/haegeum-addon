# 위치 잔차 피처(pos_resid) 추가 전/후 비교 — UGV VAE, MuJoCo 시뮬레이션

작성일 2026-10-22. 재현: `generate_datasets.py` → `train_eval_sim_vae.py --feats 10` / `--feats 12` → `plot_feature_comparison.py`. 아래 숫자는 모두 `out/sim_vae_results_f10.json`, `out/sim_vae_results_f12.json`에서 그대로 옮긴 값이다.

## 1. 문제

기존 UGV 피처 10개(gps_vel_x/y, imu_ax/ay, residual_x/y, wheel_vel_l/r, cmd_vel_x/w)는 **전부 속도·가속도·명령 기반**이다. GPS **위치**에 상수 편향을 더해도 속도 프로파일은 그대로이고(경로가 평행 이동할 뿐), GPS를 동결시켜도 직진 중이면 "속도 0 + 바퀴 1.5 m/s"라는 모순이 residual 과도응답으로 잠깐 보였다가 IMU 적분 추정이 GPS 속도 0에 수렴하면서 사라진다. 베이스라인 결과(`_f10`)가 이를 그대로 보여준다: bias R = 0.029, gradual R = 0.289, freeze R = 0.000.

## 2. 추가한 피처

바퀴 오도메트리 dead-reckoning 위치와 GPS 위치의 차이. `ugv_sim.py` `UGVSim.pos_residual`, 10 Hz (dt = 0.1 s).

```
v_odo     = (wheel_vel_l + wheel_vel_r) / 2                        # 잡음 포함 측정값
odom_pos  = (1 − α)·(odom_pos + R(yaw)·[v_odo, 0]·dt) + α·gps_pos   # α = 0.01, yaw = 컴파스 측정
pos_resid = gps_pos − odom_pos                                      # → pos_resid_x, pos_resid_y
```

- `odom_pos`는 첫 GPS 측정으로 초기화. 바퀴 속도·컴파스만으로 적분하면 드리프트가 쌓이므로 (스푸핑될 수 있는) GPS 위치에 1 %씩 앵커링 → 시정수 1/α = 100 샘플 ≈ **10 s**.
- 12차원 벡터 = 기존 10개(순서 불변) + `pos_resid_x`, `pos_resid_y`. CSV에는 `odom_x`, `odom_y`도 함께 기록.
- 공격별 이론 거동: 고정 편향 b → 즉시 b로 점프 후 e^(−t/10 s) 감쇠(정상 상태 0). 점진 드리프트 rate → rate·dt/α = rate × 10 s 에 수렴. 동결 → 차량 속도 v에 대해 v·dt/α ≈ 15 m 까지 성장.

### 정상 주행에서의 크기 (재생성한 데이터)

| 데이터 | pos_resid_x σ | pos_resid_y σ | ‖pos_resid‖ 최대 |
|---|---|---|---|
| normal_train (600 s, seed 42) | 0.138 m | 0.137 m | 0.57 m |
| normal_test (120 s, seed 123) | 0.139 m | 0.138 m | 0.49 m |
| 공격 CSV 3개의 공격 전 구간 | 0.108–0.118 m | 0.109–0.115 m | 0.40–0.44 m |

GPS 위치 잡음(σ 0.10 m)이 그대로 들어가므로 σ ≈ 0.14 m이며 발산하지 않는다.

### 공격 중 거동 (pos_resid_x, 공격 시작 20 s)

| 공격 | t=20.0 | 25 | 30 | 40 | 50 | 59.9 | 이론 |
|---|---|---|---|---|---|---|---|
| gps_spoof_bias (+4 m) | +3.76 | +2.16 | +1.37 | +0.74 | +0.35 | −0.16 | 4 m 점프 → τ 10 s 감쇠 |
| gps_spoof_gradual (0.25 m/s) | −0.13 | +0.76 | +1.47 | +2.38 | +2.53 | +2.20 | 0.25 × 10 = 2.5 m 수렴 |
| gps_freeze | −0.07 | +5.77 | +9.31 | +12.77 | +14.02 | +14.47 | ≈ 15 m 로 성장 |

## 3. 전/후 결과

조건은 두 변형 모두 동일: normal_train 6000행 학습(seed 42, 60 epoch, WIN 20), 임계값 = normal_test 재구성 오차 95 백분위, 공격 시작 후 19샘플 경계 윈도우 제외. 지연은 공격 시작 후 최초 임계값 초과까지.

| 공격 | 변형 | P | R | F1 | 지연 | 최초 탐지 윈도우 상위 기여 피처 |
|---|---|---|---|---|---|---|
| gps_spoof_bias | 10 피처 | 0.282 | 0.029 | 0.052 | 24.7 s | wheel_vel_r, imu_ay, imu_ax |
| gps_spoof_bias | **12 피처** | **0.909** | **0.732** | **0.811** | **0.0 s** | pos_resid_x, pos_resid_y, wheel_vel_r |
| gps_spoof_gradual | 10 피처 | 0.786 | 0.289 | 0.422 | 7.7 s | imu_ax, wheel_vel_r, imu_ay |
| gps_spoof_gradual | **12 피처** | **0.931** | **1.000** | **0.965** | **1.7 s** | pos_resid_x, pos_resid_y, residual_x |
| gps_freeze | 10 피처 | 0.000 | 0.000 | 0.000 | 미탐 (※) | residual_x, imu_ax, imu_ay |
| gps_freeze | **12 피처** | **0.927** | **1.000** | **0.962** | **0.0 s** | residual_x, pos_resid_y, imu_ay |

임계값: 10 피처 0.2090, 12 피처 0.2847 (피처가 늘어 정상 재구성 오차 자체가 커짐). 그림: `out/feature_comparison.png` (왼쪽 10 피처, 오른쪽 12 피처).

※ 10 피처 freeze는 JSON에 `delay_ms = 0`으로 기록되지만 R = 0이다. 동결 순간 gps_vel이 0으로 떨어지며 residual 과도응답이 점수를 2 s 가량 임계값 위로 올리지만, 그 구간은 평가에서 제외하는 경계 윈도우 안이고 이후 점수는 다시 임계값 아래로 떨어져 끝까지 올라오지 않는다. 실질적으로 미탐이다.

## 4. 해석

- **위치 수준 공격 3종 모두 탐지 가능해졌다.** 12 피처에서 최초 탐지 윈도우의 상위 기여 피처가 bias/gradual에서는 pos_resid_x/y, freeze에서는 residual_x + pos_resid_y로, 새 피처가 실제로 결정적 역할을 한다.
- **bias의 R = 0.73은 앵커 시정수의 직접적 결과다.** 잔차가 4 m에서 10 s 시정수로 감쇠해 공격 시작 ≈ 28 s 뒤(t ≈ 48 s, 잔차 ≈ 0.4 m)에 점수가 처음 임계값 아래로 내려가고, 50–60 s 구간은 16 %만 임계값 위에 머문다 (20–50 s 구간은 94 %). 즉 상수 편향은 "발생 순간"만 잡히고 **지속 상태는 탐지되지 않는다**. 운용에서는 탐지를 래치(latch)하거나 시정수를 늘려야 한다.
- **시정수보다 충분히 느린 편향은 숨는다.** 수렴값이 rate × 10 s 이므로 rate 가 0.02 m/s 이하(수렴값 ≈ 0.2 m, 정상 σ 0.14 m 수준)면 pos_resid로도 구분하기 어렵다. 0.25 m/s 드리프트는 2.5 m에 수렴해 쉽게 잡혔을 뿐이다. 반대로 α를 줄이면(시정수 ↑) 느린 공격은 더 잘 보이지만 정상 오도메트리 드리프트(바퀴 슬립, 컴파스 오차)가 누적돼 FP가 늘어나는 트레이드오프가 있다.
- **FP는 그대로다.** 공격 전 구간 FP가 두 변형 모두 27–30개(약 15 %)로, 95 백분위 임계값과 코너링 구간의 재구성 오차 때문이다. Precision 개선은 TP가 늘어난 효과이지 정상 구간 오탐이 줄어든 것이 아니다.
- **시뮬레이션 한계.** 평지·무슬립·이상적 속도 서보 환경이라 오도메트리가 실제보다 훨씬 정확하다. 실차에서는 바퀴 슬립과 컴파스 자기 간섭으로 정상 pos_resid σ가 커져 임계값이 올라가고, 작은 편향에 대한 민감도는 여기보다 낮을 것이다. GPS/RF 물리(의사거리, 신호 세기)는 모델링하지 않았으므로 "수신기 출력이 바뀐다"는 효과만 평가한 것이다.
- **데이터는 공격 CSV 3개(각 60 s, 단일 seed)뿐이다.** 공격 종류·강도·시작 시점·seed를 바꾼 반복 실험 없이 얻은 단일 사례 수치이므로, 수치 자체보다 "10 피처 → 12 피처로 위치 공격이 보이게 된다"는 정성적 결론이 요점이다.

## 5. 변경 파일

- `ugv_sim.py`: `POS_RESID_FEATURES`, `FEATURES_EXT`, `ODOM_ALPHA`, `UGVSim.pos_residual`, CSV 열 `pos_resid_x/y`, `odom_x/y`; `plot_features`가 12개 패널.
- `generate_datasets.py`: pos_resid 통계 출력 (데이터 seed·길이 불변, 기존 10개 열 값 동일 — `_f10` 결과가 이전 베이스라인과 일치함으로 확인).
- `train_eval_sim_vae.py`: `--feats 10|12`, 출력 접미사 `_f10`/`_f12`, 점수 시계열 `.npz` 저장.
- `plot_feature_comparison.py` (신규): `out/feature_comparison.png`.
