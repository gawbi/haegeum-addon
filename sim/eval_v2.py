#!/usr/bin/env python3
"""
v2 평가 (sim/.venv): 공격 9종 × (공격 시작 20/30/40 s × 시드 5) = 공격당 15회

1부 탐지와 진단 (보정 없이 주행 기록 후 판정 — 보정이 없으면 폐루프와 결과가 같다)
   12피처 VAE와 16피처 VAE의 탐지율, 지연, 윈도우 단위 Recall/오탐률
   16피처의 진단 정확도: z-score 방식 vs 복원 마스킹 방식
   정상 주행 150 s × 10회 오경보
2부 보정 (폐루프): 방어 없음 / 탐지 후 무조건 GPS→오도메트리(이전 방식) / 진단 후 센서별 보정
3부 VAE 복원 보정: 공격받은 센서 그룹 피처를 VAE 복원값으로 바꿨을 때 실제 값과의 오차
출력: out/v2_results.json, out/v2_runs/*.csv (그림용 대표 주행)
"""
import json
import os

import numpy as np
import pandas as pd

from defense_v2 import (ATTACKS, FEATURES_EXT, GROUPS, OUT_DIR, WIN, Detector, SimV2, run)

STARTS, SEEDS, LEN = [20.0, 30.0, 40.0], [0, 1, 2, 3, 4], 40.0
K, ARM = SimV2.K_CONSEC, SimV2.ARM_S
FIG = (30.0, 0)           # 그림용 대표 주행 (공격 30 s, 첫 시드)
RUNS = os.path.join(OUT_DIR, "v2_runs"); os.makedirs(RUNS, exist_ok=True)


def seed_of(ai, sd, st):
    return 5000 + 100 * ai + 10 * STARTS.index(st) + sd


def offline_detect(d, df, use_flat):
    X = df[d.feats].to_numpy(); W = d.windows(X); s = d.score(W); t = df.t.values[WIN - 1:]
    flat = np.array([d.flatline(w) for w in W]) if use_flat else np.zeros(len(W), bool)
    over = (s > d.th) | flat
    cnt, det = 0, None
    for i in range(len(W)):
        cnt = cnt + 1 if (over[i] and t[i] >= ARM) else 0
        if (flat[i] and t[i] >= ARM) or cnt >= K:
            det = i; break
    return W, t, over, det, flat


def main():
    d12 = Detector("v2_f12", feats=FEATURES_EXT); d16 = Detector("v2")
    print(f"thresholds f12={d12.th:.4f} f16={d16.th:.4f}")
    names = list(ATTACKS)
    part1, part2, part3 = [], [], []

    # 정상 주행 오경보
    normal = {"f12": 0, "f16": 0, "runs": 10, "len_s": 150}
    for sd in range(10):
        df, _ = run("normal", 150.0, 3000 + sd, length=0.0)
        for tag, d, fl in (("f12", d12, False), ("f16", d16, True)):
            normal[tag] += offline_detect(d, df, fl)[3] is not None
    print("normal false alarms", normal)

    for ai, name in enumerate(names):
        tgt = ATTACKS[name].target
        for st in STARTS:
            for sd in SEEDS:
                seed = seed_of(ai, sd, st)
                df, _ = run(name, st, seed, length=LEN)        # 방어 없음
                rec = {"attack": name, "target": tgt, "start": st, "seed": seed,
                       "xte_none": float(df[df.t >= st].xte.mean())}
                for tag, d, fl in (("f12", d12, False), ("f16", d16, True)):
                    W, t, over, det, flat = offline_detect(d, df, fl)
                    atk_w = t >= st + (WIN - 1) * 0.1; pre_w = (t >= ARM) & (t < st)
                    rec[f"{tag}_win_recall"] = float(over[atk_w].mean())
                    rec[f"{tag}_win_fpr"] = float(over[pre_w].mean())
                    rec[f"{tag}_det_t"] = float(t[det]) if det is not None else None
                    if tag == "f16" and det is not None and t[det] >= st:
                        if flat[det]:
                            rec["diag_z"] = rec["diag_zshared"] = rec["diag_mask"] = "전체"
                        else:
                            rec["diag_z"] = d.diagnose_z(W[det])[0]; rec["diag_mask"] = d.diagnose_mask(W[det])[0]
                            rec["diag_zshared"] = d.diagnose_z_shared(W[det])[0]
                            if (st, sd) == FIG:
                                rec["z_feat"] = {f: round(float(v), 2) for f, v in zip(d.feats, d.feature_z(W[det]))}
                part1.append(rec)
                if (st, sd) == FIG:
                    df.to_csv(os.path.join(RUNS, f"{name}_none.csv"), index=False)

                # 3부: VAE 복원 보정 (공격 대상 그룹 피처를 복원값으로 대체)
                if tgt in ("GPS", "IMU", "바퀴") and (st, sd) in [(30.0, s) for s in SEEDS]:
                    clean, _ = run("normal", st + LEN, seed, length=0.0) if tgt != "GPS" else (None, None)
                    fs = {"GPS": ["gps_vel_x", "gps_vel_y"], "IMU": ["imu_ax", "imu_ay"], "바퀴": ["wheel_vel_l", "wheel_vel_r"]}[tgt]
                    X = df[d16.feats].to_numpy(); W = d16.windows(X); tt = df.t.values[WIN - 1:]
                    sel = np.where(tt >= st + 2.0)[0]
                    cols = [d16.idx[f] for f in fs]
                    corr = np.array([d16.impute(W[i], GROUPS[tgt])[0][-1, cols] for i in sel])
                    att = X[WIN - 1:][sel][:, cols]
                    if tgt == "GPS":
                        ref = df[["true_vx", "true_vy"]].to_numpy()[WIN - 1:][sel]
                    else:
                        ref = clean[fs].to_numpy()[WIN - 1:][sel]
                    part3.append({"attack": name, "seed": seed, "feats": fs,
                                  "rmse_attacked": float(np.sqrt(((att - ref) ** 2).mean())),
                                  "rmse_corrected": float(np.sqrt(((corr - ref) ** 2).mean()))})
                    if sd == 0:
                        pd.DataFrame({"t": tt[sel], "ref_x": ref[:, 0], "att_x": att[:, 0], "cor_x": corr[:, 0]}).to_csv(
                            os.path.join(RUNS, f"impute_{name}.csv"), index=False)

                # 2부: 보정 (폐루프)
                for pol in ("gps_only", "diag"):
                    dfr, sim = run(name, st, seed, detector=d16, policy=pol, length=LEN)
                    part2.append({"attack": name, "start": st, "seed": seed, "policy": pol,
                                  "det_t": sim.detect_t, "diag": sim.diag_group,
                                  "xte": float(dfr[dfr.t >= st].xte.mean()), "xte_max": float(dfr[dfr.t >= st].xte.max())})
                    if (st, sd) == FIG:
                        dfr.to_csv(os.path.join(RUNS, f"{name}_{pol}.csv"), index=False)
        print("done", name, flush=True)

    p1, p2 = pd.DataFrame(part1), pd.DataFrame(part2)
    summ = []
    for name in names:
        g = p1[p1.attack == name]; h = p2[p2.attack == name]
        r = {"attack": name, "target": ATTACKS[name].target, "n": len(g), "xte_none": round(g.xte_none.mean(), 3)}
        for tag in ("f12", "f16"):
            dt = g[f"{tag}_det_t"]
            ok = dt.notna() & (dt >= g.start)
            r[f"{tag}_detected"] = int(ok.sum()); r[f"{tag}_pre"] = int((dt.notna() & (dt < g.start)).sum())
            r[f"{tag}_delay"] = round(float((dt[ok] - g.start[ok]).mean()), 2) if ok.any() else None
            r[f"{tag}_win_recall"] = round(g[f"{tag}_win_recall"].mean(), 3)
        r["diag_z_acc"] = round(float((g.diag_z == r["target"]).mean()), 3) if "diag_z" in g else None
        r["diag_zshared_acc"] = round(float((g.diag_zshared == r["target"]).mean()), 3) if "diag_zshared" in g else None
        r["diag_mask_acc"] = round(float((g.diag_mask == r["target"]).mean()), 3) if "diag_mask" in g else None
        for pol in ("gps_only", "diag"):
            hp = h[h.policy == pol]
            r[f"xte_{pol}"] = round(hp.xte.mean(), 3)
            r[f"xte_max_{pol}"] = round(hp.xte_max.mean(), 3)
        r["diag_closed_loop"] = h[h.policy == "diag"].diag.value_counts().to_dict()
        summ.append(r)
        print(r)
    # 진단 혼동행렬 (z, mask)
    labels = ["GPS", "IMU", "컴파스", "바퀴", "명령", "전체"]
    conf = {m: [[int(((p1.target == a) & (p1[f"diag_{m}"] == b)).sum()) for b in labels] for a in labels] for m in ("z", "zshared", "mask")}
    imp = pd.DataFrame(part3).groupby("attack")[["rmse_attacked", "rmse_corrected"]].mean().round(3).reset_index().to_dict("records")
    print("impute", imp)
    with open(os.path.join(OUT_DIR, "v2_results.json"), "w", encoding="utf-8") as f:
        json.dump({"thresholds": {"f12": d12.th, "f16": d16.th}, "K": K, "ARM_S": ARM, "STARTS": STARTS, "SEEDS": SEEDS,
                   "normal_false_alarm": normal, "summary": summ, "confusion_labels": labels, "confusion": conf,
                   "impute": imp, "part1": part1, "part2": part2, "part3": part3}, f, ensure_ascii=False, indent=1, default=str)
    print("saved v2_results.json")


if __name__ == "__main__":
    main()
