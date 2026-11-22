#!/usr/bin/env python3
"""
MuJoCo 기반 차동구동 UGV 시뮬레이터 + 센서 모델 + 측정 계층 공격 주입

구조:
    물리 엔진(500 Hz) → 센서 모델(10 Hz, 잡음 추가) → 공격 주입(측정값 변조)
        → 웨이포인트 제어기(변조된 GPS로 자기 위치 추정) → 바퀴 속도 명령 → 물리 엔진
    10 Hz마다 프로젝트의 UGV 10개 피처 + 위치 잔차 2개(확장 피처)를 기록한다.

UGV 피처 순서 (프로젝트 공통):
    gps_vel_x, gps_vel_y, imu_ax, imu_ay, residual_x, residual_y,
    wheel_vel_l, wheel_vel_r, cmd_vel_x, cmd_vel_w
확장 피처 (10개 뒤에 추가, FEATURES_EXT = FEATURES + 아래 2개):
    pos_resid_x, pos_resid_y   = GPS 위치 − 바퀴 오도메트리 dead-reckoning 위치

핵심: 공격은 "측정값"에만 가해지며, 제어기는 변조된 GPS 위치로 자기 위치를 추정하므로
      GPS 스푸핑 시 차량이 물리적으로 잘못된 경로로 주행한다 (= 궤적 편차가 실제로 발생).
"""
from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from typing import Callable, Optional

import mujoco
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_XML = os.path.join(HERE, "ugv_model.xml")
OUT_DIR = os.path.join(HERE, "out")

FEATURES = [
    "gps_vel_x", "gps_vel_y", "imu_ax", "imu_ay", "residual_x", "residual_y",
    "wheel_vel_l", "wheel_vel_r", "cmd_vel_x", "cmd_vel_w",
]
# 위치 수준 잔차 (GPS 위치 − 바퀴 오도메트리 위치). 기존 10개 뒤에 붙여 12차원 벡터를 만든다.
POS_RESID_FEATURES = ["pos_resid_x", "pos_resid_y"]
FEATURES_EXT = FEATURES + POS_RESID_FEATURES
ODOM_ALPHA = 0.01   # 오도메트리를 GPS 위치에 앵커링하는 상보 필터 계수 (10 Hz → 시정수 ≈ 10 s)

# 20 x 20 m 모서리 깎은 사각 루프 (8점). ugv_model.xml의 마커 위치와 동일해야 함.
WAYPOINTS = np.array([
    [10.0, -5.0], [10.0, 5.0], [5.0, 10.0], [-5.0, 10.0],
    [-10.0, 5.0], [-10.0, -5.0], [-5.0, -10.0], [5.0, -10.0],
])

PHYSICS_HZ = 500      # timestep 0.002
CONTROL_HZ = 10       # 센서/제어 샘플링
STEPS_PER_CTRL = PHYSICS_HZ // CONTROL_HZ   # 50


# ─────────────────────────────────────────────────────────────
# 공격 (측정 계층) — attack(t, meas) -> meas
#   meas 키: gps_x, gps_y, gps_vx, gps_vy, imu_ax, imu_ay, yaw, wheel_vel_l, wheel_vel_r
# ─────────────────────────────────────────────────────────────
class Attack:
    """start_t 이후 측정 dict를 변조하는 callable. label 계산을 위해 start_t를 노출한다."""
    name = "none"

    def __init__(self, start_t: float):
        self.start_t = float(start_t)

    def active(self, t: float) -> bool:
        return t >= self.start_t

    def __call__(self, t: float, meas: dict) -> dict:
        if not self.active(t):
            return meas
        return self.apply(t, meas)

    def apply(self, t: float, meas: dict) -> dict:  # pragma: no cover
        return meas


class gps_spoof_bias(Attack):
    """start_t 이후 GPS 위치에 상수 bias_xy[m], GPS 속도에 상수 vel_bias_xy[m/s]를 더한다.
    위치 bias → 제어기가 자기 위치를 bias만큼 잘못 알고 → 실제 궤적이 -bias 방향으로 평행 이동.
    vel_bias 기본값: bias 방향으로 0.3 m/s (스푸핑 신호가 도플러 속도에도 영향을 준다고 가정)."""
    name = "gps_spoof_bias"

    def __init__(self, start_t: float, bias_xy=(4.0, 0.0), vel_bias_xy=None):
        super().__init__(start_t)
        self.bias = np.asarray(bias_xy, dtype=float)
        if vel_bias_xy is None:
            n = np.linalg.norm(self.bias)
            vel_bias_xy = 0.3 * self.bias / n if n > 0 else np.zeros(2)
        self.vel_bias = np.asarray(vel_bias_xy, dtype=float)

    def apply(self, t, meas):
        meas["gps_x"] += self.bias[0]
        meas["gps_y"] += self.bias[1]
        meas["gps_vx"] += self.vel_bias[0]
        meas["gps_vy"] += self.vel_bias[1]
        return meas


class gps_spoof_gradual(Attack):
    """start_t 이후 GPS 위치가 rate_xy[m/s]로 서서히 drift. 속도 측정에는 rate_xy가 그대로 더해진다
    (위치 drift의 시간 미분 = 속도 bias → 수신기 입장에서 일관된 스푸핑)."""
    name = "gps_spoof_gradual"

    def __init__(self, start_t: float, rate_xy=(0.25, 0.0), max_offset: float = 12.0):
        super().__init__(start_t)
        self.rate = np.asarray(rate_xy, dtype=float)
        self.max_offset = float(max_offset)

    def offset(self, t):
        off = self.rate * (t - self.start_t)
        n = np.linalg.norm(off)
        if n > self.max_offset:
            off = off * (self.max_offset / n)
        return off

    def apply(self, t, meas):
        off = self.offset(t)
        meas["gps_x"] += off[0]
        meas["gps_y"] += off[1]
        if np.linalg.norm(off) < self.max_offset - 1e-9:
            meas["gps_vx"] += self.rate[0]
            meas["gps_vy"] += self.rate[1]
        return meas


class gps_freeze(Attack):
    """start_t 시점의 GPS 위치를 고정(재생 공격). 속도는 0으로 보고된다.
    제어기는 '아직 같은 자리'라고 믿고 같은 방향으로 계속 조향 → 루프를 이탈."""
    name = "gps_freeze"

    def __init__(self, start_t: float):
        super().__init__(start_t)
        self._frozen = None

    def apply(self, t, meas):
        if self._frozen is None:
            self._frozen = (meas["gps_x"], meas["gps_y"])
        meas["gps_x"], meas["gps_y"] = self._frozen
        meas["gps_vx"] = 0.0
        meas["gps_vy"] = 0.0
        return meas


# ─────────────────────────────────────────────────────────────
# 센서 모델 파라미터
# ─────────────────────────────────────────────────────────────
@dataclass
class SensorNoise:
    sigma_gps_pos: float = 0.10   # m
    sigma_gps_vel: float = 0.05   # m/s
    sigma_imu: float = 0.05       # m/s^2
    sigma_wheel: float = 0.02     # m/s
    sigma_yaw: float = 0.01       # rad (자력계/컴파스)
    imu_lpf_alpha: float = 0.3    # 가속도계 1차 저역통과 (접촉 진동 억제)


@dataclass
class ControllerParams:
    v_cruise: float = 1.5         # m/s 순항 속도
    k_heading: float = 1.8        # 헤딩 오차 P 이득 → yaw rate
    w_max: float = 1.2            # rad/s
    wp_reach_radius: float = 1.5  # m 이내 도달 시 다음 웨이포인트
    wheel_radius: float = 0.20    # m (xml과 동일)
    track_width: float = 0.68     # m (바퀴 간 거리, xml과 동일)


# ─────────────────────────────────────────────────────────────
# 시뮬레이터
# ─────────────────────────────────────────────────────────────
class UGVSim:
    def __init__(self, xml_path: str = DEFAULT_XML, seed: int = 0,
                 noise: Optional[SensorNoise] = None,
                 ctrl: Optional[ControllerParams] = None,
                 waypoints: Optional[np.ndarray] = None):
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        self.rng = np.random.default_rng(seed)
        self.noise = noise or SensorNoise()
        self.cp = ctrl or ControllerParams()
        self.waypoints = WAYPOINTS if waypoints is None else np.asarray(waypoints, float)
        self.dt_ctrl = 1.0 / CONTROL_HZ

        # 센서 주소
        self._sadr = {}
        for i in range(self.model.nsensor):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_SENSOR, i)
            self._sadr[name] = (self.model.sensor_adr[i], self.model.sensor_dim[i])
        self.chassis_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "chassis")
        self.reset()

    # ── 상태 초기화 ───────────────────────────────────────────
    def reset(self):
        mujoco.mj_resetData(self.model, self.data)
        # 첫 웨이포인트(wp7 → wp0 방향) 근처에서 +x 방향을 보고 출발
        self.data.qpos[0:3] = [0.0, -10.0, 0.20]
        self.data.qpos[3:7] = [1, 0, 0, 0]
        mujoco.mj_forward(self.model, self.data)
        self.wp_idx = 0
        self.v_est = np.zeros(2)        # IMU 적분 속도 추정 (world frame)
        self.imu_filt = np.zeros(2)     # 가속도계 LPF 상태
        self.odom_pos = None            # 바퀴 오도메트리 dead-reckoning 위치 (첫 GPS 측정으로 초기화)
        self.cmd = np.zeros(2)          # (cmd_vel_x, cmd_vel_w)
        self.t = 0.0

    def _sensor(self, name):
        adr, dim = self._sadr[name]
        return self.data.sensordata[adr:adr + dim].copy()

    # ── 진실 상태 ─────────────────────────────────────────────
    def true_state(self) -> dict:
        pos = self._sensor("gt_pos")
        vel = self._sensor("gt_vel")
        q = self._sensor("gt_quat")   # w x y z
        w, x, y, z = q
        yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
        roll = math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
        sinp = max(-1.0, min(1.0, 2 * (w * y - z * x)))
        pitch = math.asin(sinp)
        return dict(x=pos[0], y=pos[1], z=pos[2], vx=vel[0], vy=vel[1], vz=vel[2],
                    yaw=yaw, roll=roll, pitch=pitch)

    # ── 센서 모델: 진실 상태 + 잡음 → 측정 dict ───────────────
    def measure(self, ts: dict) -> dict:
        n = self.noise
        rng = self.rng
        acc = self._sensor("imu_acc")   # body frame, 중력 포함 (수평 섀시: z≈+9.81, x/y≈운동 가속도)
        # 가속도계 x/y: 수평 지면에서는 중력 성분이 거의 0 → 그대로 사용 (1차 LPF로 접촉 진동 억제)
        a_raw = acc[0:2]
        self.imu_filt = (1 - n.imu_lpf_alpha) * self.imu_filt + n.imu_lpf_alpha * a_raw
        imu_ax = self.imu_filt[0] + rng.normal(0, n.sigma_imu)
        imu_ay = self.imu_filt[1] + rng.normal(0, n.sigma_imu)

        wl = float(self._sensor("wheel_l_vel")[0]) * self.cp.wheel_radius + rng.normal(0, n.sigma_wheel)
        wr = float(self._sensor("wheel_r_vel")[0]) * self.cp.wheel_radius + rng.normal(0, n.sigma_wheel)

        return dict(
            gps_x=ts["x"] + rng.normal(0, n.sigma_gps_pos),
            gps_y=ts["y"] + rng.normal(0, n.sigma_gps_pos),
            gps_vx=ts["vx"] + rng.normal(0, n.sigma_gps_vel),
            gps_vy=ts["vy"] + rng.normal(0, n.sigma_gps_vel),
            imu_ax=imu_ax, imu_ay=imu_ay,
            yaw=ts["yaw"] + rng.normal(0, n.sigma_yaw),   # 컴파스(자력계) 헤딩
            wheel_vel_l=wl, wheel_vel_r=wr,
        )

    # ── 잔차: GPS 속도 − IMU 적분 속도 추정 ─────────────────────
    def residual(self, meas: dict) -> np.ndarray:
        """v_est = 0.98 * (v_est + a_world * dt) + 0.02 * gps_vel  (상보 필터식 dead-reckoning)
           residual = gps_vel - v_est
        a_world = R(yaw) @ [imu_ax, imu_ay]  (body → world 회전)."""
        c, s = math.cos(meas["yaw"]), math.sin(meas["yaw"])
        a_world = np.array([c * meas["imu_ax"] - s * meas["imu_ay"],
                            s * meas["imu_ax"] + c * meas["imu_ay"]])
        gps_v = np.array([meas["gps_vx"], meas["gps_vy"]])
        self.v_est = 0.98 * (self.v_est + a_world * self.dt_ctrl) + 0.02 * gps_v
        return gps_v - self.v_est

    # ── 위치 잔차: GPS 위치 − 바퀴 오도메트리 dead-reckoning ───────
    def pos_residual(self, meas: dict) -> np.ndarray:
        """v_odo    = (wheel_vel_l + wheel_vel_r) / 2                 (잡음 포함 측정값)
           odom_pos = (1−α) · (odom_pos + R(yaw) · [v_odo, 0] · dt) + α · gps_pos   (α = 0.01)
           pos_resid = gps_pos − odom_pos
        gps_pos 는 (스푸핑될 수 있는) 측정값이므로 정상 시 잔차는 잡음 수준으로 유지되고,
        GPS 위치가 바퀴 적분과 어긋나면 잔차가 뛴 뒤 시정수 1/α 샘플(≈10 s)로 수렴한다."""
        gps_pos = np.array([meas["gps_x"], meas["gps_y"]])
        if self.odom_pos is None:
            self.odom_pos = gps_pos.copy()
            return gps_pos - self.odom_pos
        v_odo = 0.5 * (meas["wheel_vel_l"] + meas["wheel_vel_r"])
        c, s = math.cos(meas["yaw"]), math.sin(meas["yaw"])
        step = np.array([c * v_odo, s * v_odo]) * self.dt_ctrl
        self.odom_pos = (1.0 - ODOM_ALPHA) * (self.odom_pos + step) + ODOM_ALPHA * gps_pos
        return gps_pos - self.odom_pos

    # ── 웨이포인트 제어기: (변조될 수 있는) GPS 위치 + 컴파스 헤딩만 사용 ──
    def controller(self, meas: dict) -> np.ndarray:
        cp = self.cp
        pos = np.array([meas["gps_x"], meas["gps_y"]])
        wp = self.waypoints[self.wp_idx]
        if np.linalg.norm(wp - pos) < cp.wp_reach_radius:
            self.wp_idx = (self.wp_idx + 1) % len(self.waypoints)
            wp = self.waypoints[self.wp_idx]
        desired = math.atan2(wp[1] - pos[1], wp[0] - pos[0])
        err = (desired - meas["yaw"] + math.pi) % (2 * math.pi) - math.pi
        w = float(np.clip(cp.k_heading * err, -cp.w_max, cp.w_max))
        # 큰 헤딩 오차에서는 감속 (최소 30 %)
        v = cp.v_cruise * max(0.3, math.cos(err))
        return np.array([v, w])

    def apply_cmd(self, cmd: np.ndarray):
        """(v, w) → 좌/우 바퀴 목표 각속도 (차동 믹싱). 토크는 모터 속도 서보가 생성."""
        v, w = cmd
        r, L = self.cp.wheel_radius, self.cp.track_width
        self.data.ctrl[0] = (v - w * L / 2) / r
        self.data.ctrl[1] = (v + w * L / 2) / r

    # ── 10 Hz 한 스텝 ─────────────────────────────────────────
    def control_step(self, attack: Optional[Callable] = None) -> dict:
        t = round(self.t, 3)   # 부동소수 누적(19.9999…)으로 공격 시작 샘플이 밀리지 않도록 반올림
        ts = self.true_state()
        meas = self.measure(ts)
        if attack is not None:
            meas = attack(t, dict(meas))
        res = self.residual(meas)
        pres = self.pos_residual(meas)
        self.cmd = self.controller(meas)
        self.apply_cmd(self.cmd)
        row = {
            "t": t,
            "gps_vel_x": meas["gps_vx"], "gps_vel_y": meas["gps_vy"],
            "imu_ax": meas["imu_ax"], "imu_ay": meas["imu_ay"],
            "residual_x": res[0], "residual_y": res[1],
            "wheel_vel_l": meas["wheel_vel_l"], "wheel_vel_r": meas["wheel_vel_r"],
            "cmd_vel_x": self.cmd[0], "cmd_vel_w": self.cmd[1],
            "pos_resid_x": pres[0], "pos_resid_y": pres[1],
            "true_x": ts["x"], "true_y": ts["y"], "true_z": ts["z"],
            "true_vx": ts["vx"], "true_vy": ts["vy"],
            "true_yaw": ts["yaw"], "true_roll": ts["roll"], "true_pitch": ts["pitch"],
            "gps_x": meas["gps_x"], "gps_y": meas["gps_y"],
            "odom_x": self.odom_pos[0], "odom_y": self.odom_pos[1],
            "wp_idx": self.wp_idx,
            "label": int(attack is not None and getattr(attack, "active", lambda _t: False)(t)),
        }
        return row

    def physics_substeps(self, n: int = STEPS_PER_CTRL):
        for _ in range(n):
            mujoco.mj_step(self.model, self.data)
        self.t = self.data.time

    # ── 실행 ─────────────────────────────────────────────────
    def run(self, duration_s: float, attack: Optional[Callable] = None,
            record_video: bool = False, video_path: Optional[str] = None,
            video_fps: int = 20, video_size=(640, 480), progress: bool = False) -> pd.DataFrame:
        self.reset()
        rows = []
        recorder = None
        if record_video:
            from video import VideoRecorder   # 지연 import (렌더링 의존성)
            video_path = video_path or os.path.join(OUT_DIR, "demo.mp4")
            recorder = VideoRecorder(self, video_path, fps=video_fps, size=video_size, attack=attack)
        n_ctrl = int(round(duration_s * CONTROL_HZ))
        steps_per_frame = PHYSICS_HZ // video_fps if recorder else None
        for k in range(n_ctrl):
            row = self.control_step(attack)
            rows.append(row)
            if recorder:
                recorder.on_sample(row)
                for _ in range(STEPS_PER_CTRL // steps_per_frame):
                    self.physics_substeps(steps_per_frame)
                    recorder.capture(row)
            else:
                self.physics_substeps()
            if not np.isfinite(self.data.qpos).all():
                raise RuntimeError(f"physics diverged (NaN) at t={self.t:.2f}")
            if progress and k % (CONTROL_HZ * 10) == 0:
                print(f"  t={self.t:6.1f}s  pos=({row['true_x']:6.2f},{row['true_y']:6.2f})  wp={self.wp_idx}")
        if recorder:
            recorder.close()
        df = pd.DataFrame(rows)
        return df


# ─────────────────────────────────────────────────────────────
# 헬퍼: 궤적/피처 플롯
# ─────────────────────────────────────────────────────────────
def plot_trajectory(df: pd.DataFrame, path: str, title: str = "", attack_start: Optional[float] = None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 7))
    wp = np.vstack([WAYPOINTS, WAYPOINTS[:1]])
    ax.plot(wp[:, 0], wp[:, 1], "--", color="#9a9a9a", lw=1.2, label="waypoint loop")
    ax.scatter(WAYPOINTS[:, 0], WAYPOINTS[:, 1], s=90, facecolors="none", edgecolors="#e07b00", lw=1.8, label="waypoints", zorder=4)
    ax.plot(df["true_x"], df["true_y"], color="#1f5fbf", lw=2.0, label="true path (physics)")
    ax.plot(df["gps_x"], df["gps_y"], color="#c0392b", lw=1.2, alpha=0.85, label="GPS-believed path (controller input)")
    if attack_start is not None and (df["t"] >= attack_start).any():
        i = int(np.argmax(df["t"].values >= attack_start))
        ax.scatter([df["true_x"].iloc[i]], [df["true_y"].iloc[i]], marker="X", s=140, color="#c0392b", zorder=5, label=f"attack start (t={attack_start:.0f}s)")
    ax.scatter([df["true_x"].iloc[0]], [df["true_y"].iloc[0]], marker="o", s=60, color="#1f5fbf", zorder=5)
    ax.set_aspect("equal")
    ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_features(df: pd.DataFrame, path: str, title: str = "", attack_start: Optional[float] = None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    feats = [f for f in FEATURES_EXT if f in df.columns]
    fig, axes = plt.subplots((len(feats) + 1) // 2, 2, figsize=(12, 2 * ((len(feats) + 1) // 2)), sharex=True)
    for ax, f in zip(axes.flat, feats):
        ax.plot(df["t"], df[f], lw=0.9, color="#1f5fbf")
        ax.set_title(f, fontsize=10, loc="left")
        ax.grid(alpha=0.3)
        if attack_start is not None:
            ax.axvline(attack_start, color="#c0392b", ls="--", lw=1.2)
    for ax in axes[-1]:
        ax.set_xlabel("t [s]")
    fig.suptitle(title + ("   (red dashed = attack start)" if attack_start is not None else ""))
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    import time
    sim = UGVSim(seed=0)
    t0 = time.time()
    df = sim.run(60.0, progress=True)
    print(f"60 s normal run in {time.time() - t0:.2f}s wall; rows={len(df)}")
    sp = np.hypot(df.true_vx, df.true_vy)
    print(f"speed mean={sp.mean():.2f} max={sp.max():.2f}  z range=({df.true_z.min():.3f},{df.true_z.max():.3f})  "
          f"|roll|max={np.abs(df.true_roll).max():.3f} |pitch|max={np.abs(df.true_pitch).max():.3f}")
    print(df[FEATURES_EXT].describe().T[["mean", "std", "min", "max"]])
