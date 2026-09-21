#!/usr/bin/env python3
"""
난수 시드 고정 유틸 — 벤치마크/설명가능성 분석 재현성 확보용.

기존 실험 코드(advanced_attacks.py, 노트북)의 동작은 건드리지 않고,
새로 추가된 benchmarks/ · explainability/ 스크립트에서만 사용합니다.
"""

import os
import random

DEFAULT_SEED = 42


def set_seed(seed: int = DEFAULT_SEED, deterministic: bool = True) -> int:
    """python / numpy / torch 전역 시드를 한 번에 고정하고 사용한 시드를 반환."""
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)

    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass

    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass

    return seed


def env_info() -> dict:
    """측정 환경 기록용 메타데이터 (칩셋/OS/torch/스레드 수)."""
    import platform
    import subprocess

    info = {
        'python': platform.python_version(),
        'platform': platform.platform(),
        'machine': platform.machine(),
        'processor': platform.processor() or platform.machine(),
    }

    if platform.system() == 'Darwin':
        try:
            info['cpu_brand'] = subprocess.check_output(
                ['sysctl', '-n', 'machdep.cpu.brand_string'], text=True
            ).strip()
        except Exception:
            pass

    try:
        import torch
        info['torch'] = torch.__version__
        info['torch_num_threads'] = torch.get_num_threads()
        info['cuda_available'] = torch.cuda.is_available()
    except ImportError:
        pass

    try:
        import numpy as np
        info['numpy'] = np.__version__
    except ImportError:
        pass

    info['cpu_count'] = os.cpu_count()
    return info
