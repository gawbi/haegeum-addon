#!/usr/bin/env python3
"""
물리 기반 UGV 데이터셋 생성
    out/normal_train.csv        600 s, seed 42   (6000 rows)
    out/normal_test.csv         120 s, seed 123  (1200 rows)
    out/gps_spoof_bias.csv      60 s, 공격 시작 20 s
    out/gps_spoof_gradual.csv   60 s, 공격 시작 20 s
    out/gps_freeze.csv          60 s, 공격 시작 20 s
각 CSV: t, 10개 피처(FEATURES 순서), pos_resid_x/y(확장 피처), ground truth(true_*),
        GPS 믿는 위치(gps_x/y), 오도메트리 위치(odom_x/y), wp_idx, label
"""
import os
import time

import numpy as np

from ugv_sim import (FEATURES, FEATURES_EXT, OUT_DIR, UGVSim, gps_freeze, gps_spoof_bias,
                     gps_spoof_gradual, plot_features, plot_trajectory)

ATTACK_START = 20.0


def summarize(name, df):
    sp = np.hypot(df.true_vx, df.true_vy)
    dev = np.hypot(df.gps_x - df.true_x, df.gps_y - df.true_y)
    print(f"[{name}] rows={len(df)}  t={df.t.iloc[-1]:.1f}s  label1={int(df.label.sum())}  "
          f"speed mean/max={sp.mean():.2f}/{sp.max():.2f} m/s  "
          f"z=({df.true_z.min():.3f},{df.true_z.max():.3f})  |roll|max={np.abs(df.true_roll).max():.3f}  "
          f"|pitch|max={np.abs(df.true_pitch).max():.3f}  |gps-true| max={dev.max():.2f} m  NaN={int(df.isna().sum().sum())}")
    pr = np.hypot(df.pos_resid_x, df.pos_resid_y)
    for lab in sorted(df.label.unique()):
        m = df.label == lab
        print(f"    pos_resid (label={lab}): std x/y={df.pos_resid_x[m].std():.3f}/{df.pos_resid_y[m].std():.3f} m  "
              f"|pos_resid| max={pr[m].max():.2f} m  mean={pr[m].mean():.2f} m")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    t0 = time.time()

    df = UGVSim(seed=42).run(600.0)
    df.to_csv(os.path.join(OUT_DIR, "normal_train.csv"), index=False)
    summarize("normal_train", df)
    plot_trajectory(df, os.path.join(OUT_DIR, "trajectory_normal.png"), "Normal run (600 s, seed 42)")
    plot_features(df.iloc[:1200], os.path.join(OUT_DIR, "features_normal.png"), "Normal run (first 120 s)")

    df = UGVSim(seed=123).run(120.0)
    df.to_csv(os.path.join(OUT_DIR, "normal_test.csv"), index=False)
    summarize("normal_test", df)

    attacks = {
        "gps_spoof_bias": gps_spoof_bias(ATTACK_START, bias_xy=(4.0, 0.0)),
        "gps_spoof_gradual": gps_spoof_gradual(ATTACK_START, rate_xy=(0.25, 0.0)),
        "gps_freeze": gps_freeze(ATTACK_START),
    }
    for i, (name, atk) in enumerate(attacks.items()):
        df = UGVSim(seed=1000 + i).run(60.0, attack=atk)
        df.to_csv(os.path.join(OUT_DIR, f"{name}.csv"), index=False)
        summarize(name, df)
        plot_trajectory(df, os.path.join(OUT_DIR, f"trajectory_{name}.png"),
                        f"{name} (attack at {ATTACK_START:.0f} s)", attack_start=ATTACK_START)
        plot_features(df, os.path.join(OUT_DIR, f"features_{name}.png"), name, attack_start=ATTACK_START)

    print("\nnormal_train feature stats:")
    import pandas as pd
    df = pd.read_csv(os.path.join(OUT_DIR, "normal_train.csv"))
    print(df[FEATURES_EXT].describe().T[["mean", "std", "min", "max"]].round(3).to_string())
    print(f"\ndone in {time.time() - t0:.1f}s -> {OUT_DIR}")


if __name__ == "__main__":
    main()
