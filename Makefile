# haegeum-addon — 재현 타깃
# 사용: make setup / make benchmark / make explain / make reproduce
PYTHON ?= python3
PIP    ?= $(PYTHON) -m pip
IMAGE  ?= haegeum-addon:cpu

# 벤치마크/SHAP 파라미터 (기본값 = README·results.json 에 기록된 실측 설정)
PYTEST_ARGS ?= -q
ITERS      ?= 2000
WARMUP     ?= 300
THREADS    ?= 1
N_EXPLAIN  ?= 10
NSAMPLES   ?= 8192
BACKGROUND ?= 30

.PHONY: help setup test benchmark explain figures attacks zeroday threshold sim-data sim-video sim-vae sim-recovery sim-v2 sim-motion reproduce docker-build docker-run docker-test clean

help:
	@echo "make setup        — requirements.txt 설치"
	@echo "make test         — pytest 단위 테스트 (ROS2 불필요)"
	@echo "make benchmark    — CPU 추론 지연시간/처리량/양자화 벤치마크"
	@echo "make explain      — SHAP 기반 설명가능성 분석"
	@echo "make figures      — UGV 피처별 기여도 그림 재생성"
	@echo "make attacks      — 기존 복합·점진적 공격 실험 재실행"
	@echo "make zeroday      — 제로데이 공격 5종(freeze/replay/drift/전체 freeze) 실험"
	@echo "make sim-data     — MuJoCo UGV 시뮬레이션 데이터셋 생성 (sim/.venv 필요)"
	@echo "make sim-video    — MuJoCo 정상/GPS 스푸핑 데모 영상 렌더링"
	@echo "make sim-vae      — 시뮬레이션 데이터로 UGV VAE 재학습·평가"
	@echo "make sim-recovery — 탐지 후 보정(오도메트리 전환) 폐루프 실험 (sim-vae 선행)"
	@echo "make sim-v2       — 공격 9종 탐지, 진단, 센서별 보정, VAE 복원 보정 실험"
	@echo "make sim-motion   — 연구 요약 모션그래픽 영상 렌더링 (sim/out/motion_summary.mp4)"
	@echo "make threshold    — ROC 기반 임계값 선택 근거 산출"
	@echo "make reproduce    — test + benchmark + explain + figures + threshold 전체 재현"
	@echo "make docker-build — CPU 재현 이미지 빌드"
	@echo "make docker-run   — 컨테이너에서 make reproduce 실행"
	@echo "make docker-test  — 컨테이너에서 make test 실행"

setup:
	$(PIP) install -r requirements.txt

# ROS2 없이 동작 (utils/load_model.py 가 rclpy 스텁을 주입)
test:
	$(PYTHON) -m pytest tests/ $(PYTEST_ARGS)

benchmark:
	$(PYTHON) benchmarks/latency_benchmark.py \
		--iters $(ITERS) --warmup $(WARMUP) --threads $(THREADS)

explain:
	$(PYTHON) explainability/shap_analysis.py \
		--n-explain $(N_EXPLAIN) --nsamples $(NSAMPLES) --background $(BACKGROUND)

figures:
	$(PYTHON) explainability/regen_ugv_feature_contribution.py \
		--nsamples $(NSAMPLES) --background $(BACKGROUND)

attacks:
	$(PYTHON) advanced_attacks.py

zeroday:
	$(PYTHON) zeroday_attacks.py

SIM_PY ?= sim/.venv/bin/python
sim-data:
	cd sim && ../$(SIM_PY) generate_datasets.py

sim-video:
	cd sim && ../$(SIM_PY) make_demo_video.py

sim-vae:
	cd sim && $(PYTHON) train_eval_sim_vae.py

sim-recovery:
	cd sim && $(PYTHON) export_vae_numpy.py && ../$(SIM_PY) recovery_eval.py

sim-v2:
	cd sim && ../$(SIM_PY) gen_v2.py && $(PYTHON) train_v2.py && ../$(SIM_PY) eval_v2.py && ../$(SIM_PY) eval_combo.py

sim-motion:
	cd sim && $(PYTHON) export_motion_data.py && ../$(SIM_PY) make_motion_video.py

threshold:
	$(PYTHON) analysis/threshold_selection.py

reproduce: test benchmark explain figures threshold
	@echo "재현 완료 → benchmarks/results.json, explainability/shap_results.json, analysis/threshold_results.json, docs/images/"

docker-build:
	docker build -t $(IMAGE) .

docker-run:
	docker run --rm -v "$(CURDIR)":/app $(IMAGE) make reproduce

docker-test:
	docker run --rm -v "$(CURDIR)":/app $(IMAGE) make test

clean:
	rm -rf __pycache__ */__pycache__ .pytest_cache
