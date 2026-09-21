# 설명가능성 (SHAP) — VAE 이상탐지 판단 근거 분해

`explainability/shap_analysis.py` 실행 결과입니다. 원시 수치는
[`shap_results.json`](shap_results.json) 에 저장됩니다.

```bash
make explain   # = python explainability/shap_analysis.py --n-explain 10 --nsamples 8192 --background 30
```

## 왜 방산 체계에 필요한가

L1 이상탐지가 경보를 올리면 상위 계층은 임무를 중단하거나 기동을 제한하는 방어 행동을
취합니다. 그런데 VAE 는 "재구성 오차가 임계값을 넘었다"는 **스칼라 하나**만 내놓기 때문에,
운용자 입장에서는 경보가 진짜 GPS 스푸핑인지, 단순 노면 충격이나 센서 드리프트로 생긴
오탐인지 화면만 보고는 구분할 수 없습니다. 유·무인 복합체계에서 이 구분이 안 되면 두
방향 모두로 위험합니다. 오탐을 믿고 임무를 중단하면 가용성이 무너지고, 반대로 경보를
습관적으로 무시하게 되면(alert fatigue) 진짜 전자전 공격을 흘려보냅니다. 경보와 함께
**"어느 센서가 이 판단을 만들었는가"** 가 제시되어야 운용자가 해당 센서의 원시값과
교차 검증(GPS↔IMU residual, 휠 오도메트리↔IMU)을 즉시 확인하고 몇 초 안에 조치를
결정할 수 있습니다.

두 번째 이유는 사후 검증과 책임 추적입니다. 방산 AI 에 요구되는 신뢰성(Trustworthy AI)
요건은 판단 결과뿐 아니라 **판단 근거가 로그로 남아 재구성 가능**할 것을 요구합니다.
`anomaly_detector._publish_alert()` 는 이미 `top_feature` 와 `attack_type` 을 이벤트에
실어 보내는데, 그 `top_feature` 는 "피처별 재구성 오차 최댓값"이라는 휴리스틱입니다.
본 분석은 그 휴리스틱이 **SHAP 이라는 독립적이고 이론적 근거(Shapley value)가 있는
귀속 방법과 실제로 일치하는지**를 검증합니다. 일치한다면 현재의 가벼운 휴리스틱을 온보드에
그대로 쓰고, SHAP 은 지상국/사후분석 단계에서 감사 도구로 운영하면 된다는 결론이 서고,
불일치한다면 그 구간이 바로 설명이 깨지는 취약 지점입니다.

## 방법

- **설명 대상 함수**: 모델 출력이 아니라 `윈도우 → 재구성 오차(anomaly score)` 스칼라 함수.
  이상 판정의 근거가 곧 재구성 오차이기 때문입니다.
- **설명기**: `shap.KernelExplainer` (모델 비종속). 배경 분포는 정상 윈도우
  `shap.kmeans(..., 30)`, 섭동 샘플 `nsamples=8192`, 조건별 상위 10개 윈도우 설명.
- **입력 차원 → 피처 귀속**: 입력은 20 타임스텝 × N 피처(UAV 160 / UGV 200)이며,
  SHAP 값을 **타임스텝 축으로 합산**해 센서 단위 기여도로 환산합니다.
- **결정성**: 실제 추론 경로는 재파라미터화 샘플링(`z = mu + std·eps`)이라 같은 입력에도
  점수가 흔들립니다. 수만 번의 섭동 평가를 하는 SHAP 에는 치명적이라 설명용 함수는
  `z = mu` 로 고정했습니다. 두 경로의 정상 윈도우 점수 상관은 **UAV r = 0.9287,
  UGV r = 1.0000** 으로, 결정적 경로가 실제 추론 경로를 잘 대표합니다.

## 결과 — 공격 유형별 피처 기여도

![UAV SHAP 기여도](../docs/images/shap_attack_contribution_uav.png)
![UGV SHAP 기여도](../docs/images/shap_attack_contribution_ugv.png)

| 플랫폼 | 조건 | 평균 점수 | SHAP 상위 3 | 기존 재구성오차 상위 3 | Spearman ρ | top-1 일치 |
|---|---|---|---|---|---|---|
| UAV | 정상 | 0.99 | residual_x, baro_alt, pitch_rate | gps_vel_y, imu_ay, residual_x | 0.57 | ✗ |
| UAV | GPS Spoofing | 5,446 | **residual_x, residual_y, gps_vel_x** | residual_x, residual_y, gps_vel_x | **1.00** | ✓ |
| UAV | Altitude Spoofing | 2,733 | **baro_alt**, imu_ay, gps_vel_y | baro_alt, gps_vel_y, imu_ay | 0.88 | ✓ |
| UAV | Command Injection | 593 | **pitch_rate**, gps_vel_y, imu_ay | pitch_rate, gps_vel_y, imu_ay | 0.93 | ✓ |
| UGV | 정상 | 0.99 | cmd_vel_x, gps_vel_y, wheel_vel_l | cmd_vel_x, gps_vel_y, wheel_vel_l | 0.94 | ✓ |
| UGV | GPS Spoofing | 473,505 | **residual_x, gps_vel_x, residual_y** | residual_y, cmd_vel_x, gps_vel_y | **-0.08** | ✗ |
| UGV | Wheel Slip | 1,367 | **wheel_vel_l, wheel_vel_r**, residual_x | wheel_vel_l, wheel_vel_r, residual_x | **1.00** | ✓ |
| UGV | Command Anomaly | 505 | **wheel_vel_l, wheel_vel_r**, cmd_vel_x | wheel_vel_l, wheel_vel_r, cmd_vel_x | 0.94 | ✓ |

![UAV SHAP summary](../docs/images/shap_summary_uav.png)
![UGV SHAP summary](../docs/images/shap_summary_ugv.png)

**공격 유형별 서명이 분리됩니다.** GPS 스푸핑은 GPS-IMU residual 계열, 고도 조작은
`baro_alt` 단독, 명령 주입은 `pitch_rate`(UAV) / 휠 속도(UGV)로, `anomaly_detector.py` 의
`FEATURE_TO_ATTACK` 매핑이 가정한 대로 나타납니다. 즉 현재의 공격 유형 분류 로직은
SHAP 기준으로도 근거가 있습니다.

## 기존 기여도 이미지와의 대조 (검증 포인트)

기존 `docs/images/uav_feature_contribution.png`, `ugv_feature_contribution.png` (노트북의
피처별 재구성 오차 XAI)와 대조한 결과입니다. **일치·불일치를 그대로 적습니다.**

| 기존 이미지 패널 | 기존 이미지 상위 피처 | 본 SHAP 결과 | 판정 |
|---|---|---|---|
| UAV / GPS Spoofing | residual_x, residual_y, gps_vel_x | residual_x, residual_y, gps_vel_x | **일치** |
| UAV / Altitude Spoofing | baro_alt (사실상 단독) | baro_alt | **일치** |
| UAV / Command Injection | imu_ax, pitch_rate | pitch_rate (imu 계열은 하위) | **부분 불일치 — 원인 규명됨** |
| UGV / Wheel Slip | wheel_vel_l, wheel_vel_r | wheel_vel_l, wheel_vel_r | **일치** |
| UGV / Command Anomaly | wheel_vel_l, wheel_vel_r | wheel_vel_l, wheel_vel_r | **일치** |
| UGV / GPS Spoofing | wheel_vel_r, gps_vel_x, cmd_vel_w (스케일 1e14) | residual_x, gps_vel_x, residual_y | **불일치 — 기존 결과가 신뢰 불가** |

### 불일치 1 — UAV Command Injection: 데이터 생성기가 다름

노트북의 `generate_command_injection()` 은 `pitch_rate` **와 함께** `imu_ax`, `imu_ay` 를
동시에 교란합니다. 반면 본 분석이 쓴 `advanced_attacks.atk_cmd_injection()` 은
`pitch_rate` 열만 교란합니다. 따라서 SHAP 이 `pitch_rate` 를 최상위로 지목한 것은
**주입된 교란과 정확히 일치**하며, 기존 이미지에서 `imu_ax` 가 1위였던 것도 그 데이터에서는
맞는 설명입니다. 모델의 설명이 갈린 것이 아니라 설명 대상 데이터가 다릅니다.

### 불일치 2 — UGV GPS Spoofing: 기존 기여도 자체가 붕괴

기존 이미지의 UGV / GPS Spoofing 패널은 가로축 스케일이 **1e14** 이고 피처 순위가
물리적으로 말이 되지 않습니다(휠 속도가 GPS 스푸핑의 1위). 두 가지 원인을 확인했습니다.

1. **정규화 기준 불일치.** 노트북은 정상 데이터를 `generate_normal(1000)`, 공격 데이터를
   `generate_*(300)` 으로 만들어 섞어 씁니다. 그런데 `generate_normal` 은
   `t = linspace(0, 10, n)` 의 `np.gradient` 로 속도를 만들기 때문에 n 이 달라지면 속도
   스케일 자체가 3배 이상 달라집니다. 정상(n=1000) 기준 mean/std 로 공격(n=300)을
   정규화하면 공격과 무관한 구간까지 전부 분포를 벗어납니다. → 본 저장소의
   `utils/ugv_data.py` 는 길이를 n=1000 으로 통일해 이 문제를 제거했습니다.
2. **극단적 OOD 에서의 포화.** 길이를 맞춘 뒤에도 UGV GPS 스푸핑 윈도우는 평균 점수가
   47만에 달하는 극단적 분포 이탈이라, 디코더 출력이 발산하면서 **재구성 오차가 10개
   피처 전체에 거의 균일하게(2.7~6.4 × 10¹³) 퍼집니다.** 이 상태에서 "최대 오차 피처"를
   고르는 휴리스틱은 사실상 난수 뽑기가 되고(ρ = -0.08), 실제로 원인과 무관한 `cmd_vel_x`
   가 2위로 올라옵니다. 같은 윈도우에 대해 SHAP 은 `residual_x`, `gps_vel_x`, `residual_y`
   를 지목해 **주입된 교란(gps_vel_x/y, residual_x/y)과 정확히 일치**했습니다.

**시사점:** 정상 범위~중간 강도 이상에서는 기존의 가벼운 휴리스틱(피처별 재구성 오차)으로
충분합니다(ρ 0.88~1.00). 그러나 **점수가 극단적으로 튀는 강한 공격에서는 휴리스틱의
피처 지목이 무너지며, 하필 그 구간이 운용자에게 근거 제시가 가장 중요한 구간**입니다.
온보드에는 휴리스틱을 유지하되, 점수가 임계값의 수백 배를 넘는 경보에 대해서는
지상국에서 SHAP 재분석을 돌려 원인 센서를 확정하는 2단계 구성이 타당합니다.

## 한계

- 합성 데이터 기반이라는 저장소 전체의 한계를 그대로 가집니다.
- `KernelExplainer` 는 근사기입니다. 조건별 10개 윈도우 × nsamples 8192 로 측정했고,
  `nsamples` 를 줄이면(예: 256) 순위가 불안정해지는 것을 확인했습니다. 재현 시
  `--nsamples` 를 낮추지 마십시오.
- SHAP 값은 타임스텝 축으로 합산했으므로 "언제"가 아니라 "어느 센서"만 설명합니다.
  시점별 설명은 `docs/images/*_timestep_scores.png` 의 시계열 점수와 함께 읽어야 합니다.
- 본 분석은 온보드 실시간 경로가 아니라 **오프라인 감사 도구**입니다
  (조건당 1.4~2.1 초 소요, `shap_results.json` 의 `elapsed_sec` 참조).
