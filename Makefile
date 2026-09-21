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

.PHONY: help setup test benchmark explain figures attacks threshold reproduce docker-build docker-run docker-test clean

help:
	@echo "make setup        — requirements.txt 설치"
	@echo "make test         — pytest 단위 테스트 (ROS2 불필요)"
	@echo "make benchmark    — CPU 추론 지연시간/처리량/양자화 벤치마크"
	@echo "make explain      — SHAP 기반 설명가능성 분석"
	@echo "make figures      — UGV 피처별 기여도 그림 재생성"
	@echo "make attacks      — 기존 복합·점진적 공격 실험 재실행"
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
