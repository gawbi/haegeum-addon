#!/usr/bin/env python3
"""
탐지 후 보정(recovery) 폐루프 실험 — sim/.venv 로 실행 (mujoco 필요, torch 불필요)

탐지만 하는 모델에서 한 걸음 더: VAE가 GPS 스푸핑을 탐지하면 제어기가 GPS를 버리고
바퀴 오도메트리 + 컴파스 dead-reckoning(DR) 위치로 주행을 이어간다.

    매 0.1 s: 센서 → (공격) → 피처 12개 기록 → 최근 2 s 윈도우를 VAE로 채점
    점수 > 임계값이 K초 연속이면 '탐지' (시동 직후 ARM_S 초는 무장 전)
    탐지 시: 오도메트리 위치를 ROLLBACK 샘플 전(공격 오염 전) 값으로 되돌린 뒤
             그 이후 바퀴 이동량을 다시 적분 → 이후 제어기는 이 DR 위치만 사용

비교 조건 (공격 3종 + 정상 주행):
    none : 방어 없음 (기존 시스템)
    f10  : 기존 10피처 VAE 탐지 + 보정
    f12  : 위치 잔차 포함 12피처 VAE 탐지 + 보정
평가: 실제 차량 위치가 웨이포인트 루프(경로)에서 벗어난 거리(cross-track error), 공격 구간 평균/최대

선행: train_eval_sim_vae.py --feats 10/12 → export_vae_numpy.py (시스템 python3)
평가: 공격 시작 시각 3개 × 시드 5개 반복. 공격 전에 탐지가 울린 실행은 '오경보'로 따로 센다.
출력: out/recovery_results.json, out/recovery_trajectories.png, out/recovery_error.png
"""
from __future__ import annotations

import json
import math
import os
from collections import deque

import numpy as np
import pandas as pd

from ugv_sim import (FEATURES, OUT_DIR, POS_RESID_FEATURES, WAYPOINTS, UGVSim,
                     gps_freeze, gps_spoof_bias, gps_spoof_gradual)

WIN = 20            # 2 s 윈도우 (train_eval_sim_vae.py 와 동일)
K_CONSEC = 20       # 2 s 연속 초과 시 탐지 (1 s 는 정상 주행 오경보가 잦음 — sweep() 결과 참조)
ARM_S = 10.0        # 시동 직후 정지→가속 구간은 학습 데이터에 거의 없어 오경보 → 10 s 후 무장
ROLLBACK = 50       # 5 s 전 오도메트리로 되돌림 (공격 시작~탐지 확정 지연 1.8~3.3 s 를 덮음)
STARTS = [20.0, 30.0, 40.0]   # 공격 시작 시각 (코너 위치에 따른 우연을 피하려고 여러 개)
SEEDS = [0, 1, 2, 3, 4]
ATTACK_LEN = 40.0             # 공격 후 주행 시간
NORMAL_LEN = 100.0            # 정상 주행 길이 (오경보 확인)
FIG_START = 30.0              # 그림에 쓰는 대표 실행


# ── numpy VAE (z = mu, 결정적 추론) ─────────────────────────────
class NumpyVAE:
    def __init__(self, n_feat: int):
        self.feats = FEATURES + (POS_RESID_FEATURES if n_feat == 12 else [])
        self.W = dict(np.load(os.path.join(OUT_DIR, f"vae_ugv_sim_f{n_feat}_np.npz")))
        tr = pd.read_csv(os.path.join(OUT_DIR, "normal_train.csv"))[self.feats].to_numpy()
        self.mean, self.std = tr.mean(0), tr.std(0) + 1e-8
        te = pd.read_csv(os.path.join(OUT_DIR, "normal_test.csv"))[self.feats].to_numpy()
        idx = np.arange(WIN)[None] + np.arange(len(te) - WIN + 1)[:, None]
        self.th = float(np.percentile(self.score(te[idx]), 95))   # 정상 검증셋 95p (기존 규칙과 동일)

    def score(self, win_raw: np.ndarray) -> np.ndarray:
        """win_raw: (N, WIN, F) 원시 피처 → (N,) 재구성 MSE"""
        W, r = self.W, (lambda a: np.maximum(a, 0))
        x = ((win_raw - self.mean) / self.std).reshape(len(win_raw), -1)
        h = r(r(x @ W["encoder_0_weight"].T + W["encoder_0_bias"]) @ W["encoder_2_weight"].T + W["encoder_2_bias"])
        z = h @ W["fc_mu_weight"].T + W["fc_mu_bias"]
        d = r(r(z @ W["decoder_0_weight"].T + W["decoder_0_bias"]) @ W["decoder_2_weight"].T + W["decoder_2_bias"])
        d = d @ W["decoder_4_weight"].T + W["decoder_4_bias"]
        return ((d - x) ** 2).mean(1)


# ── 탐지 + 보정 시뮬레이터 ───────────────────────────────────────
class GuardedUGVSim(UGVSim):
    def __init__(self, detector: NumpyVAE | None = None, **kw):
        self.detector = detector
        super().__init__(**kw)

    def reset(self):
        super().reset()
        self.buf = deque(maxlen=WIN)              # 최근 WIN개 피처 행
        self.odom_hist = deque(maxlen=ROLLBACK + 1)  # 오도메트리 위치 이력
        self.step_hist = deque(maxlen=ROLLBACK + 1)  # 샘플별 바퀴 이동량 (world frame)
        self.over = 0
        self.dr_pos = None                        # None = GPS 모드, 값 = 보정(DR) 모드
        self.detect_t = None
        self.last_score = 0.0

    def controller(self, meas):
        v_odo = 0.5 * (meas["wheel_vel_l"] + meas["wheel_vel_r"])
        step = np.array([math.cos(meas["yaw"]), math.sin(meas["yaw"])]) * v_odo * self.dt_ctrl
        self.step_hist.append(step)
        self.odom_hist.append(self.odom_pos.copy())
        if self.dr_pos is None:
            return super().controller(meas)
        self.dr_pos = self.dr_pos + step          # 보정 모드: GPS 대신 바퀴 적분 위치로 제어
        return super().controller(dict(meas, gps_x=self.dr_pos[0], gps_y=self.dr_pos[1]))

    def control_step(self, attack=None):
        row = super().control_step(attack)
        if self.detector is not None:
            self.buf.append([row[f] for f in self.detector.feats])
            if len(self.buf) == WIN:
                self.last_score = float(self.detector.score(np.array(self.buf)[None])[0])
                self.over = self.over + 1 if (self.last_score > self.detector.th and row["t"] >= ARM_S) else 0
                if self.over >= K_CONSEC and self.dr_pos is None:
                    # 오염 전 오도메트리로 되돌린 뒤, 그 이후 바퀴 이동량을 다시 더한다
                    self.dr_pos = self.odom_hist[0] + np.sum(list(self.step_hist)[1:], axis=0)
                    self.detect_t = row["t"]
        row["score"] = self.last_score
        row["mode_dr"] = int(self.dr_pos is not None)
        return row


# ── 평가: 경로(루프)에서 벗어난 거리 ─────────────────────────────
def cross_track(xy: np.ndarray) -> np.ndarray:
    a = WAYPOINTS
    b = np.roll(WAYPOINTS, -1, axis=0)
    d = np.full(len(xy), np.inf)
    for p, q in zip(a, b):
        u = np.clip(((xy - p) @ (q - p)) / ((q - p) @ (q - p)), 0, 1)
        d = np.minimum(d, np.linalg.norm(xy - (p + u[:, None] * (q - p)), axis=1))
    return d


def make_attack(name, start):
    return {"normal": None,
            "gps_spoof_bias": gps_spoof_bias(start, bias_xy=(4.0, 0.0)),
            "gps_spoof_gradual": gps_spoof_gradual(start, rate_xy=(0.25, 0.0)),
            "gps_freeze": gps_freeze(start)}[name]


def run_one(det, seed, atk, start):
    """start 초에 공격 시작, 이후 ATTACK_LEN 초 주행. 반환: df, 요약 dict"""
    df = GuardedUGVSim(detector=det, seed=seed).run(start + ATTACK_LEN, attack=make_attack(atk, start))
    df["xte"] = cross_track(df[["true_x", "true_y"]].to_numpy())
    dt = df.loc[df.mode_dr == 1, "t"]
    det_t = float(dt.iloc[0]) if len(dt) else None
    w = df[df.t >= start]
    return df, {"attack": atk, "seed": seed, "start": start, "detect_t": det_t,
                "false_trigger": det_t is not None and det_t < start,
                "xte_mean": float(w.xte.mean()), "xte_max": float(w.xte.max())}


def sweep(dets, ks=(10, 20, 30), n_normal=10, normal_len=150.0):
    """연속 조건 K에 따른 오경보/탐지/경로 이탈 trade-off"""
    global K_CONSEC
    k0, out = K_CONSEC, []
    for k in ks:
        K_CONSEC = k
        for cond in ("f10", "f12"):
            det = dets[cond]
            fa = sum(run_one(det, 3000 + s, "normal", normal_len)[1]["detect_t"] is not None for s in range(n_normal))
            row = {"K": k, "cond": cond, "normal_false_runs": fa, "normal_runs": n_normal}
            for ai, atk in enumerate(["gps_spoof_bias", "gps_spoof_gradual", "gps_freeze"]):
                rs = [run_one(det, 2000 + 100 * ai + s, atk, st)[1] for st in STARTS for s in SEEDS]
                ok = [r for r in rs if r["detect_t"] is not None and not r["false_trigger"]]
                row[atk] = {"detected": len(ok), "runs": len(rs), "pre_attack_trigger": sum(r["false_trigger"] for r in rs),
                            "xte_mean": round(float(np.mean([r["xte_mean"] for r in rs])), 3)}
            out.append(row)
            print("sweep", row)
    K_CONSEC = k0
    return out


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    dets = {"none": None, "f10": NumpyVAE(10), "f12": NumpyVAE(12)}
    print(f"thresholds: f10={dets['f10'].th:.4f}  f12={dets['f12'].th:.4f}")
    attacks = ["gps_spoof_bias", "gps_spoof_gradual", "gps_freeze"]
    rows, runs = [], {}
    # 정상 주행: 오경보로 보정 모드에 들어가는지 (SEEDS개 × NORMAL_LEN 초)
    for cond, det in dets.items():
        for sd in SEEDS:
            df, r = run_one(det, 3000 + sd, "normal", NORMAL_LEN)   # start=NORMAL_LEN → 공격 없음
            rows.append(dict(r, cond=cond, attack="normal", false_trigger=r["detect_t"] is not None))
    # 공격: 시작 시각 STARTS × 시드 SEEDS
    for ai, atk in enumerate(attacks):
        for cond, det in dets.items():
            for st in STARTS:
                for sd in SEEDS:
                    df, r = run_one(det, 2000 + 100 * ai + sd, atk, st)
                    rows.append(dict(r, cond=cond))
                    if st == FIG_START and sd == SEEDS[0]:
                        runs[(atk, cond)] = df
    res = pd.DataFrame(rows)

    summary = []
    for (atk, cond), g in res.groupby(["attack", "cond"], sort=False):
        ok = g[~g.false_trigger & g.detect_t.notna()]
        summary.append({
            "attack": atk, "cond": cond, "n_runs": len(g),
            "false_trigger_runs": int(g.false_trigger.sum()),
            "detected_after_attack": int(len(ok)),
            "delay_s_mean": round(float((ok.detect_t - ok.start).mean()), 2) if len(ok) else None,
            "xte_mean": round(float(g.xte_mean.mean()), 3),
            "xte_max_mean": round(float(g.xte_max.mean()), 3),
            "xte_mean_excl_false": round(float(g[~g.false_trigger].xte_mean.mean()), 3) if (~g.false_trigger).any() else None,
        })
    for s in summary:
        print(s)
    sw = sweep(dets)
    with open(os.path.join(OUT_DIR, "recovery_results.json"), "w", encoding="utf-8") as f:
        json.dump({"sweep": sw, "K_CONSEC": K_CONSEC, "ARM_S": ARM_S, "ROLLBACK": ROLLBACK, "STARTS": STARTS,
                   "SEEDS": SEEDS, "ATTACK_LEN": ATTACK_LEN, "NORMAL_LEN": NORMAL_LEN,
                   "thresholds": {k: v.th for k, v in dets.items() if v},
                   "summary": summary, "runs": res.to_dict("records")}, f, ensure_ascii=False, indent=2)

    # 그림 1: 공격 3종 × 조건 궤적 (대표 실행: 공격 FIG_START 초, 첫 시드)
    colors = {"none": "#c0392b", "f10": "#e0a100", "f12": "#1f5fbf"}
    labels = {"none": "no defense", "f10": "10-feature VAE + recovery", "f12": "12-feature VAE + recovery"}
    loop = np.vstack([WAYPOINTS, WAYPOINTS[:1]])
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.2))
    for ax, atk in zip(axes, attacks):
        ax.plot(loop[:, 0], loop[:, 1], "--", color="#9a9a9a", lw=1.2, label="planned loop")
        for cond in dets:
            d = runs[(atk, cond)]
            d = d[d.t >= FIG_START - 5]
            ax.plot(d.true_x, d.true_y, color=colors[cond], lw=1.8, label=labels[cond])
            if d.mode_dr.any():
                j = int(np.argmax(d.mode_dr.values))
                ax.scatter(d.true_x.iloc[j], d.true_y.iloc[j], marker="*", s=180, color=colors[cond], edgecolor="k", zorder=5)
        ax.set_aspect("equal"); ax.grid(alpha=0.3); ax.set_title(atk); ax.set_xlabel("x [m]")
    axes[0].set_ylabel("y [m]"); axes[0].legend(loc="lower left", fontsize=8)
    fig.suptitle(f"True vehicle path, attack at {FIG_START:.0f} s (star = detection, switch to odometry)")
    fig.tight_layout(); fig.savefig(os.path.join(OUT_DIR, "recovery_trajectories.png"), dpi=140); plt.close(fig)

    # 그림 2: 시간에 따른 경로 이탈 거리 (대표 실행)
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    for ax, atk in zip(axes, attacks):
        for cond in dets:
            d = runs[(atk, cond)]
            ax.plot(d.t, d.xte, color=colors[cond], lw=1.5, label=labels[cond])
        ax.axvline(FIG_START, color="k", ls=":", lw=1)
        ax.set_ylabel("off-path [m]"); ax.set_title(atk, loc="left", fontsize=10); ax.grid(alpha=0.3)
    axes[0].legend(fontsize=8, loc="upper left"); axes[-1].set_xlabel(f"time [s]  (attack starts at {FIG_START:.0f} s)")
    fig.tight_layout(); fig.savefig(os.path.join(OUT_DIR, "recovery_error.png"), dpi=140); plt.close(fig)
    print("saved recovery_results.json, recovery_trajectories.png, recovery_error.png")


if __name__ == "__main__":
    main()
