#!/usr/bin/env python3
"""
오프스크린 렌더링(mujoco.Renderer) → MP4 (imageio + ffmpeg)

프레임 구성 (가로 결합): [탑뷰 (고정 카메라 "top")] | [추적 카메라 "chase"]
탑뷰에는 실제 궤적(파란 점)과 GPS가 믿는 위치 궤적(빨간 점)을 scene에 구를 추가해 그린다.
상단에 시간 / 공격 상태 / 실제-GPS 위치 편차를 텍스트로 오버레이한다 (Pillow).
"""
from __future__ import annotations

import os

import imageio.v2 as imageio
import mujoco
import numpy as np

try:
    from PIL import Image, ImageDraw, ImageFont
    _HAS_PIL = True
except Exception:  # pragma: no cover
    _HAS_PIL = False


class VideoRecorder:
    def __init__(self, sim, path: str, fps: int = 20, size=(640, 480), attack=None,
                 cameras=("top", "chase"), trail_every_s: float = 0.5):
        self.sim = sim
        self.path = path
        self.fps = fps
        self.w, self.h = size
        self.attack = attack
        self.cameras = list(cameras)
        self.trail_every = trail_every_s
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self.renderer = mujoco.Renderer(sim.model, height=self.h, width=self.w, max_geom=6000)
        self.writer = imageio.get_writer(path, fps=fps, codec="libx264", quality=8,
                                         macro_block_size=16, format="FFMPEG")
        self.true_trail: list[tuple[float, float]] = []
        self.gps_trail: list[tuple[float, float]] = []
        self._last_trail_t = -1e9
        self._last_row = None
        self.n_frames = 0
        if _HAS_PIL:
            try:
                self.font = ImageFont.load_default(size=18)
                self.font_big = ImageFont.load_default(size=24)
            except TypeError:
                self.font = self.font_big = ImageFont.load_default()
        self.chassis_geom = mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_GEOM, "chassis_box")

    # 10 Hz 샘플마다 호출: 궤적 점 추가
    def on_sample(self, row: dict):
        self._last_row = row
        if row["t"] - self._last_trail_t >= self.trail_every - 1e-6:
            self.true_trail.append((row["true_x"], row["true_y"]))
            self.gps_trail.append((row["gps_x"], row["gps_y"]))
            self._last_trail_t = row["t"]

    def _add_trail(self, scene, pts, rgba, radius=0.18, z=0.05):
        for (x, y) in pts:
            if scene.ngeom >= scene.maxgeom:
                break
            g = scene.geoms[scene.ngeom]
            mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_SPHERE,
                                np.array([radius, 0, 0]), np.array([x, y, z]),
                                np.eye(3).flatten(), np.asarray(rgba, dtype=np.float32))
            scene.ngeom += 1

    def _render(self, cam: str, attacked: bool):
        # 공격 중이면 섀시 색을 빨강으로 (시각적 표시만, 물리 무관)
        self.sim.model.geom_rgba[self.chassis_geom] = (0.85, 0.15, 0.15, 1) if attacked else (0.15, 0.45, 0.95, 1)
        self.renderer.update_scene(self.sim.data, camera=cam)
        if cam == "top":
            self._add_trail(self.renderer.scene, self.true_trail, (0.1, 0.4, 1.0, 1.0))
            self._add_trail(self.renderer.scene, self.gps_trail, (1.0, 0.1, 0.1, 0.9), radius=0.12, z=0.3)
        return self.renderer.render().copy()

    def capture(self, row: dict):
        t = self.sim.t
        attacked = bool(self.attack is not None and self.attack.active(round(t, 3)))
        frames = [self._render(c, attacked) for c in self.cameras]
        frame = np.concatenate(frames, axis=1)
        if _HAS_PIL:
            frame = self._overlay(frame, t, attacked, row)
        self.writer.append_data(frame)
        self.n_frames += 1

    def _overlay(self, frame, t, attacked, row):
        img = Image.fromarray(frame)
        d = ImageDraw.Draw(img, "RGBA")
        ts = self.sim.true_state()
        dev = float(np.hypot(row["gps_x"] - ts["x"], row["gps_y"] - ts["y"]))
        status = "GPS SPOOFING ACTIVE" if attacked else "NORMAL"
        color = (220, 40, 40, 255) if attacked else (30, 140, 60, 255)
        d.rectangle([0, 0, img.width, 34], fill=(255, 255, 255, 255))
        d.text((10, 6), f"t = {t:5.1f} s", fill=(0, 0, 0, 255), font=self.font_big)
        d.text((170, 6), status, fill=color, font=self.font_big)
        name = getattr(self.attack, "name", "") if attacked else ""
        d.text((500, 9), f"{name}   |GPS - true| = {dev:4.1f} m   v = {np.hypot(ts['vx'], ts['vy']):.2f} m/s",
               fill=(0, 0, 0, 255), font=self.font)
        d.rectangle([0, img.height - 26, self.w, img.height], fill=(255, 255, 255, 170))
        d.text((8, img.height - 23), "top view   blue = true path   red = GPS-believed path   orange = waypoints",
               fill=(0, 0, 0, 255), font=self.font)
        d.rectangle([self.w, img.height - 26, img.width, img.height], fill=(255, 255, 255, 170))
        d.text((self.w + 8, img.height - 23), "chase camera", fill=(0, 0, 0, 255), font=self.font)
        return np.asarray(img)

    def close(self):
        self.writer.close()
        self.renderer.close()
        self.sim.model.geom_rgba[self.chassis_geom] = (0.15, 0.45, 0.95, 1)
