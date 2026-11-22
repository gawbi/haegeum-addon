#!/usr/bin/env python3
"""
모션그래픽 영상(make_motion_video.py)에 쓸 제로데이 이상 점수 시계열을 내보낸다.

torch 가 필요하므로 시스템 Python 으로 실행한다 (sim/.venv 에는 torch 가 없음):
    cd sim && python3 export_motion_data.py

zeroday_attacks.main() 과 같은 보정·시드 경로를 그대로 따르므로
점수는 zeroday_results.json 의 수치와 일치한다.
출력: sim/out/zeroday_scores.npz
"""
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from advanced_attacks import gen_normal, run_experiment  # noqa: E402
from zeroday_attacks import (SEED, N_TOTAL, ATK_START, calibrate,  # noqa: E402
                             atk_gps_freeze, atk_sensor_freeze_all)


def main():
    det, _, cal_scores = calibrate()
    base = gen_normal(N_TOTAL, seed=123)
    out = {'threshold': det.th, 'normal_min': float(cal_scores.min()),
           'normal_max': float(cal_scores.max()), 'atk_start': ATK_START}
    for key, data in (('gps_freeze', atk_gps_freeze(base, ATK_START)),
                      ('freeze_all', atk_sensor_freeze_all(base, ATK_START))):
        torch.manual_seed(SEED)
        res = run_experiment(det, data, key, ATK_START, N_TOTAL)
        out[f'{key}_scores'] = res['scores']
        out[f'{key}_recall'] = res['Recall']
    path = HERE / 'out' / 'zeroday_scores.npz'
    np.savez(path, **out)
    print(f'[저장] {path}  threshold={det.th:.4f}  normal_min={out["normal_min"]:.4f}  '
          f'recall freeze={out["gps_freeze_recall"]} freeze_all={out["freeze_all_recall"]}')


if __name__ == '__main__':
    main()
