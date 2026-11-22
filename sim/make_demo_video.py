#!/usr/bin/env python3
"""
교수님 데모용 영상 렌더링
    out/demo_normal.mp4      20 s 정상 주행
    out/demo_gps_spoof.mp4   40 s, 15 s부터 GPS 스푸핑(bias 4 m, +x) → 차량이 루프에서 눈에 띄게 이탈
프레임: 탑뷰(640x480) | 추적캠(640x480) = 1280x480, 20 fps
"""
import os
import sys
import time

import numpy as np

from ugv_sim import OUT_DIR, UGVSim, gps_spoof_bias, plot_features, plot_trajectory

SPOOF_START = 15.0
BIAS = (4.0, 0.0)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    quick = "--quick" in sys.argv

    t0 = time.time()
    p = os.path.join(OUT_DIR, "demo_normal.mp4")
    df = UGVSim(seed=7).run(5.0 if quick else 20.0, record_video=True, video_path=p)
    print(f"{p}: {os.path.getsize(p)/1e6:.1f} MB, {len(df)} samples, {time.time()-t0:.1f}s")

    t0 = time.time()
    p = os.path.join(OUT_DIR, "demo_gps_spoof.mp4")
    atk = gps_spoof_bias(SPOOF_START, bias_xy=BIAS)
    df = UGVSim(seed=7).run(20.0 if quick else 40.0, attack=atk, record_video=True, video_path=p)
    dev = np.hypot(df.gps_x - df.true_x, df.gps_y - df.true_y)
    print(f"{p}: {os.path.getsize(p)/1e6:.1f} MB, {len(df)} samples, {time.time()-t0:.1f}s  "
          f"|gps-true| after spoof: mean={dev[df.label==1].mean():.2f} m")
    plot_trajectory(df, os.path.join(OUT_DIR, "trajectory_gps_spoof.png"),
                    f"Demo: GPS spoof bias {BIAS} m at {SPOOF_START:.0f} s", attack_start=SPOOF_START)
    plot_features(df, os.path.join(OUT_DIR, "features_gps_spoof.png"),
                  f"Demo: GPS spoof bias {BIAS} m", attack_start=SPOOF_START)


if __name__ == "__main__":
    main()
