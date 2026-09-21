# haegeum-addon — CPU 전용 재현 이미지
#   벤치마크(benchmarks/latency_benchmark.py)와
#   설명가능성 분석(explainability/shap_analysis.py)을 그대로 재현합니다.
#   ROS2 런타임은 포함하지 않습니다 (두 스크립트는 rclpy 없이 동작).
#
#   build : docker build -t haegeum-addon:cpu .
#   run   : docker run --rm -v "$PWD/out:/out" haegeum-addon:cpu make reproduce
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    MPLBACKEND=Agg \
    PIP_NO_CACHE_DIR=1

# make: Makefile 타깃 실행용 / libgomp1: torch OpenMP 런타임
RUN apt-get update && apt-get install -y --no-install-recommends \
        make libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# CPU 전용 PyTorch 휠 인덱스 사용 (CUDA 런타임 미포함 → 이미지 경량)
COPY requirements.txt .
RUN pip install --upgrade pip \
    && pip install --extra-index-url https://download.pytorch.org/whl/cpu \
         -r requirements.txt

COPY . .

CMD ["make", "reproduce"]
