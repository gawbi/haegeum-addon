#!/usr/bin/env python3
"""
탐지 → 진단 → 보정 (v2) — 공격 9종으로 확장한 UGV 시뮬레이터. sim/.venv 로 실행.

ugv_sim.UGVSim 을 상속하고 기존 파일은 건드리지 않는다 (이전 결과 재현 유지).

추가된 것
  센서     : 자이로 yaw rate (gyro_z)
  피처 16개: 기존 12개 + gyro_z, yaw_resid(컴파스 − 자이로 적분 헤딩),
             w_resid(자이로 − 직전 회전 명령), v_resid(바퀴 평균 속도 − 직전 속도 명령)
  공격 9종 : GPS 오프셋/점진 이동/고정/재전송, 컴파스 오프셋, 바퀴 엔코더 배율,
             IMU 음향 주입(사인파), 구동 명령 주입(액추에이터), 센서 전체 고정
  보정     : 진단된 센서별로 다르게 대응
             GPS     → 바퀴 오도메트리 dead-reckoning 위치로 주행 (5 s 롤백)
             컴파스  → 자이로 적분 헤딩으로 주행 (5 s 롤백)
             명령    → 외란 관측기로 주입량 추정 후 상쇄
             전체 고정 → 안전 정지
             IMU, 바퀴 → 제어에 쓰지 않는 센서라 격리(기록)만
"""
from __future__ import annotations

import math
import os
from collections import deque

import numpy as np
import pandas as pd

from ugv_sim import FEATURES_EXT, OUT_DIR, WAYPOINTS, UGVSim

FEATURES_V2 = FEATURES_EXT + ["gyro_z", "yaw_resid", "w_resid", "v_resid"]
WIN = 20
HEAD_ALPHA = 0.02          # 컴파스로 자이로 헤딩을 붙잡는 상보 필터 계수 (시정수 ≈ 5 s)
SIGMA_GYRO = 0.01          # rad/s

# 진단용 센서 그룹: 그 센서가 직접 만들거나 계산에 들어가는 피처
GROUPS = {
    "GPS":     ["gps_vel_x", "gps_vel_y", "residual_x", "residual_y", "pos_resid_x", "pos_resid_y"],
    "IMU":     ["imu_ax", "imu_ay", "residual_x", "residual_y"],
    "컴파스":  ["yaw_resid", "residual_x", "residual_y", "pos_resid_x", "pos_resid_y"],
    "바퀴":    ["wheel_vel_l", "wheel_vel_r", "v_resid", "pos_resid_x", "pos_resid_y"],
    "명령":    ["cmd_vel_x", "cmd_vel_w", "w_resid", "v_resid"],
}
# 각 센서만 만드는 고유 피처 (공유 피처는 여러 센서가 원인일 수 있어 진단에서 뺀다: 구조적 잔차 아이디어)
EXCLUSIVE = {
    "GPS": ["gps_vel_x", "gps_vel_y"], "IMU": ["imu_ax", "imu_ay"], "컴파스": ["yaw_resid"],
    "바퀴": ["wheel_vel_l", "wheel_vel_r", "v_resid"], "명령": ["w_resid"],
}
RAW_SENSOR_FEATS = ["gps_vel_x", "gps_vel_y", "imu_ax", "imu_ay", "wheel_vel_l", "wheel_vel_r", "gyro_z"]


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


# ─────────────────────────────────────────────────────────────
# 공격 9종
# ─────────────────────────────────────────────────────────────
class Atk:
    name, target, actuator = "none", None, False

    def __init__(self, start):
        self.start_t = float(start)

    def active(self, t):
        return t >= self.start_t

    def __call__(self, t, m):
        return self.apply(t, m) if self.active(t) else m

    def apply(self, t, m):          # 측정 공격이 아니면 그대로
        return m

    def apply_cmd(self, t, cmd):    # 구동 명령 공격이 아니면 그대로
        return cmd


class GpsBias(Atk):
    name, target = "gps_bias", "GPS"

    def apply(self, t, m):
        m["gps_x"] += 4.0; m["gps_vx"] += 0.3
        return m


class GpsGradual(Atk):
    name, target = "gps_gradual", "GPS"

    def apply(self, t, m):
        off = min(0.25 * (t - self.start_t), 12.0)
        m["gps_x"] += off
        if off < 12.0:
            m["gps_vx"] += 0.25
        return m


class GpsFreeze(Atk):
    name, target = "gps_freeze", "GPS"

    def __init__(self, start):
        super().__init__(start); self.v = None

    def apply(self, t, m):
        if self.v is None:
            self.v = (m["gps_x"], m["gps_y"])
        m["gps_x"], m["gps_y"] = self.v; m["gps_vx"] = m["gps_vy"] = 0.0
        return m


class GpsReplay(Atk):
    """10 s 전에 녹화한 GPS를 다시 보냄 (녹화는 공격 전부터 계속)"""
    name, target = "gps_replay", "GPS"

    def __init__(self, start, lag=100):
        super().__init__(start); self.buf = deque(maxlen=lag + 1)

    def __call__(self, t, m):
        self.buf.append((m["gps_x"], m["gps_y"], m["gps_vx"], m["gps_vy"]))
        if self.active(t) and len(self.buf) == self.buf.maxlen:
            m["gps_x"], m["gps_y"], m["gps_vx"], m["gps_vy"] = self.buf[0]
        return m


class CompassBias(Atk):
    name, target = "compass_bias", "컴파스"

    def apply(self, t, m):
        m["yaw"] = wrap(m["yaw"] + math.radians(25)); return m


class WheelScale(Atk):
    name, target = "wheel_scale", "바퀴"

    def apply(self, t, m):
        m["wheel_vel_l"] *= 1.3; m["wheel_vel_r"] *= 1.3; return m


class ImuInject(Atk):
    """MEMS 가속도계 음향 공진 주입을 흉내 낸 사인파 (진폭 1.5 m/s², 1.3 Hz)"""
    name, target = "imu_inject", "IMU"

    def apply(self, t, m):
        ph = 2 * math.pi * 1.3 * (t - self.start_t)
        m["imu_ax"] += 1.5 * math.sin(ph); m["imu_ay"] += 1.5 * math.cos(ph); return m


class CmdInject(Atk):
    """제어기 출력과 모터 사이에서 회전 명령에 +0.3 rad/s를 더함 (액추에이터 계층)"""
    name, target, actuator = "cmd_inject", "명령", True

    def apply_cmd(self, t, cmd):
        if self.active(t):
            cmd = cmd.copy(); cmd[1] += 0.3
        return cmd


class FreezeAll(Atk):
    name, target = "freeze_all", "전체"

    def __init__(self, start):
        super().__init__(start); self.v = None

    def apply(self, t, m):
        if self.v is None:
            self.v = dict(m)
        return dict(self.v)


class Combo(Atk):
    """여러 공격을 같은 시각에 동시에 건다 (복합 공격)"""
    def __init__(self, start, parts):
        super().__init__(start)
        self.parts = [ATTACKS[p](start) for p in parts]
        self.name = "+".join(parts)
        self.target = "+".join(a.target for a in self.parts)
        self.actuator = any(a.actuator for a in self.parts)

    def __call__(self, t, m):
        for a in self.parts:
            m = a(t, m)
        return m

    def apply_cmd(self, t, cmd):
        for a in self.parts:
            cmd = a.apply_cmd(t, cmd)
        return cmd


ATTACKS = {c.name: c for c in (GpsBias, GpsGradual, GpsFreeze, GpsReplay, CompassBias,
                               WheelScale, ImuInject, CmdInject, FreezeAll)}
ATTACK_KO = {"gps_bias": "GPS 위치 오프셋", "gps_gradual": "GPS 점진 이동", "gps_freeze": "GPS 고정",
             "gps_replay": "GPS 재전송", "compass_bias": "컴파스 오프셋", "wheel_scale": "바퀴 엔코더 배율",
             "imu_inject": "IMU 음향 주입", "cmd_inject": "구동 명령 주입", "freeze_all": "센서 전체 고정"}
COMBOS = {"gps_bias+compass_bias": "GPS 오프셋 + 컴파스 오프셋",
          "gps_freeze+wheel_scale": "GPS 고정 + 바퀴 엔코더 배율",
          "compass_bias+cmd_inject": "컴파스 오프셋 + 구동 명령 주입"}
ATTACK_KO.update(COMBOS)
MULTI_REL, MULTI_ZMIN = 0.2, 5.0    # 다중 진단: 최대 그룹 점수의 20% 이상이면서 z 5 이상인 센서를 모두 공격 대상으로


# ─────────────────────────────────────────────────────────────
# numpy VAE (z = mu) + 진단
# ─────────────────────────────────────────────────────────────
class Detector:
    def __init__(self, tag="v2", feats=FEATURES_V2):
        self.feats = list(feats)
        self.W = dict(np.load(os.path.join(OUT_DIR, f"vae_{tag}_np.npz")))
        st = np.load(os.path.join(OUT_DIR, f"vae_{tag}_norm.npz"))
        self.mean, self.std = st["mean"], st["std"]
        te = pd.read_csv(os.path.join(OUT_DIR, "normal_test_v2.csv"))
        wins = np.concatenate([self.windows(g[self.feats].to_numpy()) for _, g in te.groupby("run")])
        s, fe = self.score(wins, per_feature=True)
        self.th = float(np.percentile(s, 95))
        self.fe_mu, self.fe_sd = fe.mean(0), fe.std(0) + 1e-9   # 피처별 정상 오차 분포 (z-score 진단용)
        self.idx = {f: i for i, f in enumerate(self.feats)}

    @staticmethod
    def windows(x):
        idx = np.arange(WIN)[None] + np.arange(len(x) - WIN + 1)[:, None]
        return x[idx]

    def recon(self, xn):
        """xn: (N, WIN*F) 정규화 입력 → 재구성"""
        W, r = self.W, (lambda a: np.maximum(a, 0))
        h = r(r(xn @ W["encoder_0_weight"].T + W["encoder_0_bias"]) @ W["encoder_2_weight"].T + W["encoder_2_bias"])
        z = h @ W["fc_mu_weight"].T + W["fc_mu_bias"]
        d = r(r(z @ W["decoder_0_weight"].T + W["decoder_0_bias"]) @ W["decoder_2_weight"].T + W["decoder_2_bias"])
        return d @ W["decoder_4_weight"].T + W["decoder_4_bias"]

    def score(self, win_raw, per_feature=False):
        n = len(win_raw)
        xn = ((win_raw - self.mean) / self.std).reshape(n, -1)
        err = (self.recon(xn) - xn) ** 2
        if not per_feature:
            return err.mean(1)
        return err.mean(1), err.reshape(n, WIN, -1).mean(1)

    def feature_z(self, win_raw):
        _, fe = self.score(win_raw[None], per_feature=True)
        return (fe[0] - self.fe_mu) / self.fe_sd

    # 진단 1 (기본): 피처별 재구성 오차의 z-score를 센서 고유 피처끼리 평균, 가장 큰 센서
    def diagnose_z(self, win_raw):
        z = self.feature_z(win_raw)
        g = {k: float(np.mean([z[self.idx[f]] for f in fs])) for k, fs in EXCLUSIVE.items()}
        return max(g, key=g.get), g

    # 진단 1-c (복합 공격용): 조건을 넘는 센서를 모두 고른다
    def diagnose_multi(self, win_raw):
        _, g = self.diagnose_z(win_raw)
        mx = max(g.values())
        sel = [k for k in EXCLUSIVE if g[k] >= MULTI_REL * mx and g[k] >= MULTI_ZMIN]
        return "+".join(sel), g

    # 진단 1-b (비교): 공유 피처까지 포함한 그룹 평균 z-score
    def diagnose_z_shared(self, win_raw):
        z = self.feature_z(win_raw)
        g = {k: float(np.mean([z[self.idx[f]] for f in fs])) for k, fs in GROUPS.items()}
        return max(g, key=g.get), g

    # 진단 2 + VAE 복원 보정: 그룹 피처를 VAE 복원값으로 반복 대체했을 때 점수가 가장 많이 내려가는 센서
    def impute(self, win_raw, feats, iters=8):
        xn = (win_raw - self.mean) / self.std
        cols = [self.idx[f] for f in feats]
        x = xn.copy(); x[:, cols] = 0.0          # 정상 평균으로 시작
        for _ in range(iters):
            r = self.recon(x.reshape(1, -1)).reshape(WIN, -1)
            x[:, cols] = r[:, cols]
        s = float(((self.recon(x.reshape(1, -1)) - x.reshape(1, -1)) ** 2).mean())
        return x * self.std + self.mean, s

    def diagnose_mask(self, win_raw):
        g = {k: self.impute(win_raw, fs)[1] for k, fs in GROUPS.items()}
        return min(g, key=g.get), g

    def flatline(self, win_raw):
        """값 변화 없음 검사: 원시 센서 피처가 2초 동안 모두 그대로면 센서 버스 고정으로 판단"""
        cols = [self.idx[f] for f in RAW_SENSOR_FEATS]
        return bool(np.all(win_raw[:, cols].std(0) < 1e-9))


# ─────────────────────────────────────────────────────────────
# 시뮬레이터 v2
# ─────────────────────────────────────────────────────────────
class SimV2(UGVSim):
    K_CONSEC, ARM_S, ROLLBACK = 20, 10.0, 50

    def __init__(self, detector: Detector | None = None, recover=True, diag="z", policy="diag", **kw):
        """policy: diag = 진단된 센서별 보정, gps_only = 탐지되면 무조건 GPS 대신 오도메트리 (이전 방식)
        diag: z = 가장 의심되는 센서 하나, multi = 조건을 넘는 센서 모두 (비교용),
              seq = 순차 진단 (하나를 보정한 뒤 그 센서 피처를 VAE 복원값으로 가리고도 이상이 계속되면 다음 센서), mask = 비교용"""
        self.detector, self.recover, self.diag, self.policy = detector, recover, diag, policy
        super().__init__(**kw)

    def reset(self):
        super().reset()
        self.h_est = None
        self.prev_sent = np.zeros(2)
        self.buf = deque(maxlen=WIN)
        self.over = 0
        self.modes = set()          # 보정 중인 센서: GPS / 컴파스 / 명령 / 전체 / IMU / 바퀴
        self.detect_t = None
        self.diag_group = None      # 진단 결과 (recover=False 여도 기록)
        self.diag_scores = None
        self.score_now = 0.0
        self.over2 = 0
        self.diag_times = []        # (시각, 추가된 센서)
        self.hist = deque(maxlen=self.ROLLBACK + 1)   # (오도메트리 위치, 헤딩 추정, 바퀴 속도, 명령 속도, 컴파스, 자이로)
        self.dr_pos = None; self.h_safe = None; self.d_hat = 0.0

    def measure(self, ts):
        m = super().measure(ts)
        m["gyro_z"] = float(self._sensor("imu_gyro")[2]) + self.rng.normal(0, SIGMA_GYRO)
        return m

    def _enter(self, groups):
        """5 s 전(공격 오염 전) 상태로 되돌린 뒤 믿을 수 있는 센서로만 지금까지 다시 적분"""
        self.modes = set(groups)
        H = list(self.hist)
        pos, h = H[0][0].copy(), H[0][1]
        for _, _, v_wh, v_cmd, yaw, gyro in H[1:]:
            h = wrap(h + gyro * self.dt_ctrl)
            hd = h if "컴파스" in self.modes else yaw
            v = v_cmd if "바퀴" in self.modes else v_wh
            pos = pos + np.array([math.cos(hd), math.sin(hd)]) * v * self.dt_ctrl
        if "GPS" in self.modes:
            self.dr_pos = pos
        if "컴파스" in self.modes:
            self.h_safe = h

    def control_step(self, attack=None):
        t = round(self.t, 3)
        dt = self.dt_ctrl
        ts = self.true_state()
        m = self.measure(ts)
        if attack is not None:
            m = attack(t, dict(m))
        res = self.residual(m)
        pres = self.pos_residual(m)
        # 자이로 + 컴파스 상보 필터 헤딩
        if self.h_est is None:
            self.h_est = m["yaw"]
        pred = wrap(self.h_est + m["gyro_z"] * dt)
        yaw_resid = wrap(m["yaw"] - pred)
        self.h_est = wrap(pred + HEAD_ALPHA * yaw_resid)
        w_resid = m["gyro_z"] - self.prev_sent[1]
        v_resid = 0.5 * (m["wheel_vel_l"] + m["wheel_vel_r"]) - self.prev_sent[0]

        # 롤백용 이력
        v_wh = 0.5 * (m["wheel_vel_l"] + m["wheel_vel_r"])
        self.hist.append((self.odom_pos.copy(), self.h_est, v_wh, self.prev_sent[0], m["yaw"], m["gyro_z"]))

        # 보정 모드에 따른 제어 입력
        mc = dict(m)
        if "컴파스" in self.modes:
            self.h_safe = wrap(self.h_safe + m["gyro_z"] * dt)
            mc["yaw"] = self.h_safe
        if "GPS" in self.modes:
            hd = mc["yaw"]
            v = self.prev_sent[0] if "바퀴" in self.modes else v_wh   # 바퀴도 공격받으면 명령 속도로 추정
            self.dr_pos = self.dr_pos + np.array([math.cos(hd), math.sin(hd)]) * v * dt
            mc["gps_x"], mc["gps_y"] = self.dr_pos
        cmd_int = self.controller(mc)
        sent = cmd_int.copy()
        if "명령" in self.modes:
            self.d_hat = 0.7 * self.d_hat + 0.3 * w_resid       # 외란 관측기: 자이로 − 보낸 명령
            sent[1] = float(np.clip(cmd_int[1] - self.d_hat, -1.6, 1.6))
        if "전체" in self.modes:
            sent = np.zeros(2)                                    # 안전 정지
        act = attack.apply_cmd(t, sent) if attack is not None else sent
        self.apply_cmd(act)
        self.cmd = cmd_int
        self.prev_sent = sent.copy()

        row = {"t": t,
               "gps_vel_x": m["gps_vx"], "gps_vel_y": m["gps_vy"], "imu_ax": m["imu_ax"], "imu_ay": m["imu_ay"],
               "residual_x": res[0], "residual_y": res[1], "wheel_vel_l": m["wheel_vel_l"], "wheel_vel_r": m["wheel_vel_r"],
               "cmd_vel_x": sent[0], "cmd_vel_w": sent[1], "pos_resid_x": pres[0], "pos_resid_y": pres[1],
               "gyro_z": m["gyro_z"], "yaw_resid": yaw_resid, "w_resid": w_resid, "v_resid": v_resid,
               "true_x": ts["x"], "true_y": ts["y"], "true_yaw": ts["yaw"], "true_vx": ts["vx"], "true_vy": ts["vy"],
               "gps_x": m["gps_x"], "gps_y": m["gps_y"],
               "label": int(attack is not None and attack.active(t))}

        # 탐지 → 진단 → 보정 모드 진입
        if self.detector is not None:
            self.buf.append([row[f] for f in self.detector.feats])
            # 순차 진단: 이미 보정 중인 센서의 피처를 VAE 복원값으로 가린 뒤에도 이상이면 다음 센서를 찾는다
            if (self.diag == "seq" and self.recover and self.modes and "전체" not in self.modes
                    and len(self.buf) == WIN and len(self.modes) < len(EXCLUSIVE)):
                w = np.array(self.buf)
                handled = sorted({f for g in self.modes for f in GROUPS[g]})
                wi, s2 = self.detector.impute(w, handled)
                self.over2 = self.over2 + 1 if s2 > self.detector.th else 0
                if self.over2 >= self.K_CONSEC:
                    z = self.detector.feature_z(wi)
                    cand = {k: float(np.mean([z[self.detector.idx[f]] for f in fs]))
                            for k, fs in EXCLUSIVE.items() if k not in self.modes}
                    g2 = max(cand, key=cand.get)
                    self._enter(sorted(self.modes) + [g2])
                    self.diag_group += "+" + g2
                    self.diag_times.append((t, g2))
                    self.over2 = 0
            if len(self.buf) == WIN and t >= self.ARM_S and self.diag_group is None:
                w = np.array(self.buf)
                self.score_now = float(self.detector.score(w[None])[0])
                flat = self.detector.flatline(w)
                self.over = self.over + 1 if (self.score_now > self.detector.th or flat) else 0
                if flat or self.over >= self.K_CONSEC:
                    if self.policy == "gps_only":
                        grp, self.diag_scores = "GPS", None
                    elif flat:
                        grp, self.diag_scores = "전체", None
                    elif self.diag in ("z", "seq"):
                        grp, self.diag_scores = self.detector.diagnose_z(w)
                    elif self.diag == "multi":
                        grp, self.diag_scores = self.detector.diagnose_multi(w)
                    else:
                        grp, self.diag_scores = self.detector.diagnose_mask(w)
                    self.detect_t, self.diag_group = t, grp
                    self.diag_times.append((t, grp))
                    if self.recover:
                        self._enter(grp.split("+"))
        row["score"] = self.score_now
        row["mode"] = "+".join(sorted(self.modes))
        row["diag"] = self.diag_group or ""
        return row

def cross_track(xy):
    a, b = WAYPOINTS, np.roll(WAYPOINTS, -1, axis=0)
    d = np.full(len(xy), np.inf)
    for p, q in zip(a, b):
        u = np.clip(((xy - p) @ (q - p)) / ((q - p) @ (q - p)), 0, 1)
        d = np.minimum(d, np.linalg.norm(xy - (p + u[:, None] * (q - p)), axis=1))
    return d


def make_attack(name, start):
    if name == "normal":
        return None
    if "+" in name:
        return Combo(start, name.split("+"))
    return ATTACKS[name](start)


def run(attack_name, start, seed, detector=None, recover=True, diag="z", policy="diag", length=40.0):
    atk = make_attack(attack_name, start)
    sim = SimV2(detector=detector, recover=recover, diag=diag, policy=policy, seed=seed)
    df = sim.run(start + length, attack=atk)
    df["xte"] = cross_track(df[["true_x", "true_y"]].to_numpy())
    return df, sim
