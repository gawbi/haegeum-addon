#!/usr/bin/env python3
"""
연구 진행 요약 모션그래픽 영상 (약 80초, 1920x1080, 30 fps)

모든 수치·곡선은 결과 파일에서 직접 읽는다 (영상 안에 하드코딩된 결과 수치 없음):
  ../analysis/threshold_results.json      기존 합성 데이터 AUC
  out/gps_spoof_bias.csv                  시뮬레이션 궤적 (실제 / GPS가 믿는 위치)
  out/demo_gps_spoof.mp4                  MuJoCo 실제 렌더링 화면
  out/sim_vae_scores_f10.npz, _f12.npz    10/12 피처 VAE 이상 점수 시계열
  out/sim_vae_results_f10.json, _f12.json 10/12 피처 Recall
  ../zeroday_results.json                 제로데이 5종 Recall·지연
  out/zeroday_scores.npz                  제로데이 점수 시계열 (export_motion_data.py 산출)

실행 (sim/.venv):
  .venv/bin/python make_motion_video.py                  # 전체 렌더 → out/motion_summary.mp4
  .venv/bin/python make_motion_video.py --preview 3 20 40  # 지정 시각(초) PNG 미리보기만
"""
import argparse
import json
import math
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OUT = HERE / 'out'

W, H, FPS = 1920, 1080, 30
SS = 2  # 슈퍼샘플링 배율 (2배로 그린 뒤 축소 → 안티에일리어싱)

# ── 색상 (어두운 배경용, 범주 색은 고정 순서) ─────────────────
BG = (16, 18, 23)
PANEL = (26, 29, 36)
GRID = (52, 56, 66)
TEXT = (245, 246, 248)
TEXT2 = (176, 181, 192)
MUTED = (112, 118, 130)
BLUE = (57, 135, 229)      # 실제 경로 / 10 피처
ORANGE = (235, 104, 52)    # GPS가 믿는 경로 / 공격
AQUA = (27, 175, 122)      # 12 피처 (개선)
RED = (227, 73, 72)        # 미탐 / 경고
YELLOW = (237, 161, 0)     # 웨이포인트

FONT_PATH = '/System/Library/Fonts/AppleSDGothicNeo.ttc'
WEIGHT_IDX = {'light': 8, 'regular': 0, 'medium': 2, 'semibold': 4, 'bold': 6}
_font_cache = {}


def font(size, weight='regular'):
    key = (size, weight)
    if key not in _font_cache:
        _font_cache[key] = ImageFont.truetype(FONT_PATH, int(size * SS), index=WEIGHT_IDX[weight])
    return _font_cache[key]


# ── 이징·타이밍 ──────────────────────────────────────────────
def clamp(x, a=0.0, b=1.0):
    return max(a, min(b, x))


def ease_out(x):
    x = clamp(x)
    return 1 - (1 - x) ** 3


def ease_in_out(x):
    x = clamp(x)
    return 4 * x ** 3 if x < 0.5 else 1 - (-2 * x + 2) ** 3 / 2


def prog(t, start, dur, ease=ease_out):
    return ease((t - start) / dur) if dur > 0 else float(t >= start)


def mix(c, base, a):
    a = clamp(a)
    return tuple(int(round(b + (cc - b) * a)) for cc, b in zip(c, base))


# ── 그리기 래퍼 (좌표는 1080p 기준, 내부에서 SS 배) ────────────
class Canvas:
    def __init__(self):
        self.img = Image.new('RGB', (W * SS, H * SS), BG)
        self.d = ImageDraw.Draw(self.img)
        self.scene_a = 1.0  # 장면 전체 페이드

    def _c(self, color, alpha, base):
        return mix(color, base, alpha * self.scene_a)

    @staticmethod
    def _p(pts):
        return [(x * SS, y * SS) for x, y in pts]

    def text(self, x, y, s, size=32, weight='regular', color=TEXT, alpha=1.0,
             anchor='la', base=BG, spacing=0.35):
        if alpha <= 0.002:
            return
        self.d.text((x * SS, y * SS), s, font=font(size, weight), fill=self._c(color, alpha, base),
                    anchor=anchor, spacing=int(size * SS * spacing))

    def text_width(self, s, size=32, weight='regular'):
        return self.d.textlength(s, font=font(size, weight)) / SS

    def line(self, pts, color, width=3, alpha=1.0, base=BG):
        if alpha <= 0.002 or len(pts) < 2:
            return
        self.d.line(self._p(pts), fill=self._c(color, alpha, base), width=int(width * SS), joint='curve')

    def dashed(self, p0, p1, color, width=2, dash=10, gap=8, alpha=1.0, base=BG):
        (x0, y0), (x1, y1) = p0, p1
        L = math.hypot(x1 - x0, y1 - y0)
        if L == 0:
            return
        ux, uy = (x1 - x0) / L, (y1 - y0) / L
        s = 0.0
        while s < L:
            e = min(s + dash, L)
            self.line([(x0 + ux * s, y0 + uy * s), (x0 + ux * e, y0 + uy * e)], color, width, alpha, base)
            s += dash + gap

    def rect(self, x0, y0, x1, y1, fill=None, outline=None, width=2, radius=0, alpha=1.0, base=BG):
        if alpha <= 0.002:
            return
        box = [x0 * SS, y0 * SS, x1 * SS, y1 * SS]
        f = self._c(fill, alpha, base) if fill else None
        o = self._c(outline, alpha, base) if outline else None
        if radius:
            self.d.rounded_rectangle(box, radius=int(radius * SS), fill=f, outline=o, width=int(width * SS))
        else:
            self.d.rectangle(box, fill=f, outline=o, width=int(width * SS))

    def circle(self, cx, cy, r, fill=None, outline=None, width=2, alpha=1.0, base=BG):
        if alpha <= 0.002 or r <= 0:
            return
        box = [(cx - r) * SS, (cy - r) * SS, (cx + r) * SS, (cy + r) * SS]
        self.d.ellipse(box, fill=self._c(fill, alpha, base) if fill else None,
                       outline=self._c(outline, alpha, base) if outline else None, width=int(width * SS))

    def arrow(self, p0, p1, color, width=3, head=14, alpha=1.0, base=BG):
        self.line([p0, p1], color, width, alpha, base)
        ang = math.atan2(p1[1] - p0[1], p1[0] - p0[0])
        a1, a2 = ang + math.radians(150), ang - math.radians(150)
        pts = [p1, (p1[0] + head * math.cos(a1), p1[1] + head * math.sin(a1)),
               (p1[0] + head * math.cos(a2), p1[1] + head * math.sin(a2))]
        self.d.polygon(self._p(pts), fill=self._c(color, alpha, base))

    def paste(self, im, x, y):
        """1080p 좌표에 RGB 이미지 붙이기 (SS 배로 확대). 장면 페이드는 배경과 블렌딩."""
        im = im.resize((im.width * SS, im.height * SS), Image.BILINEAR)
        if self.scene_a < 0.999:
            bg = Image.new('RGB', im.size, BG)
            im = Image.blend(bg, im, self.scene_a)
        self.img.paste(im, (int(x * SS), int(y * SS)))

    def final(self):
        return self.img.reduce(SS)


# ── 데이터 로드 ──────────────────────────────────────────────
def load_json(p):
    with open(p, encoding='utf-8') as f:
        return json.load(f)


class Data:
    def __init__(self):
        thr = load_json(ROOT / 'analysis' / 'threshold_results.json')
        self.auc_nominal = thr['platforms']['uav']['auc']['nominal_pooled_all_attacks']

        traj = pd.read_csv(OUT / 'gps_spoof_bias.csv')
        self.traj_t = traj['t'].to_numpy()
        self.true_xy = traj[['true_x', 'true_y']].to_numpy()
        self.gps_xy = traj[['gps_x', 'gps_y']].to_numpy()
        self.traj_label = traj['label'].to_numpy()
        err = np.hypot(*(self.gps_xy - self.true_xy).T)
        self.err_smooth = pd.Series(err).rolling(10, min_periods=1).mean().to_numpy()
        self.atk_t = float(self.traj_t[np.argmax(self.traj_label == 1)])

        from ugv_sim import WAYPOINTS  # 시뮬레이션과 동일한 웨이포인트
        self.waypoints = np.asarray(WAYPOINTS)

        self.f10 = np.load(OUT / 'sim_vae_scores_f10.npz')
        self.f12 = np.load(OUT / 'sim_vae_scores_f12.npz')
        r10 = {r['attack']: r for r in load_json(OUT / 'sim_vae_results_f10.json')['results']}
        r12 = {r['attack']: r for r in load_json(OUT / 'sim_vae_results_f12.json')['results']}
        self.sim_attacks = [('gps_spoof_bias', 'GPS 고정 편향'),
                            ('gps_spoof_gradual', 'GPS 점진 드리프트'),
                            ('gps_freeze', 'GPS 동결')]
        self.recall10 = [r10[k]['Recall'] for k, _ in self.sim_attacks]
        self.recall12 = [r12[k]['Recall'] for k, _ in self.sim_attacks]

        zd = load_json(ROOT / 'zeroday_results.json')
        names = {'GPS_Freeze': 'GPS 동결', 'GPS_Replay': 'GPS 재전송',
                 'Drift rate=0.02': 'GPS 드리프트 (빠름)', 'Drift rate=0.005': 'GPS 드리프트 (느림)',
                 'FreezeAll': '전체 센서 동결'}
        self.zeroday = []
        for r in zd['results']:
            label = next(v for k, v in names.items() if k in r['attack'])
            self.zeroday.append((label, r['Recall'], r['delay_ms']))
        zs = np.load(OUT / 'zeroday_scores.npz')
        self.zd_th = float(zs['threshold'])
        self.zd_normal_min = float(zs['normal_min'])
        self.zd_atk_start = int(zs['atk_start'])
        self.zd_freeze_all = zs['freeze_all_scores']


class VideoClip:
    """MuJoCo 데모 영상을 순차 스트리밍 (전체를 메모리에 올리지 않음)."""

    def __init__(self, path, size):
        self.path, self.size = path, size
        self.reader = None
        self.idx = -1
        self.frame = None

    def get(self, idx):
        if self.reader is None or idx < self.idx:
            if self.reader is not None:
                self.reader.close()
            self.reader = imageio.get_reader(str(self.path), 'ffmpeg')
            self.it = iter(self.reader)
            self.idx = -1
        while self.idx < idx:
            try:
                raw = next(self.it)
            except StopIteration:
                break
            self.idx += 1
            if self.idx >= idx:
                self.frame = Image.fromarray(raw).resize(self.size, Image.LANCZOS)
        return self.frame


# ── 공통 요소 ────────────────────────────────────────────────
CHAPTERS = []  # (start, end, label) — 장면 정의 뒤 채움
TOTAL = 0.0


def chrome(cv, t):
    """하단 진행 바 + 장 제목 (장면 페이드와 무관하게 항상 표시)."""
    a_save, cv.scene_a = cv.scene_a, 1.0
    y = H - 46
    cv.line([(80, y), (W - 80, y)], GRID, 2)
    cv.line([(80, y), (80 + (W - 160) * clamp(t / TOTAL), y)], TEXT2, 2)
    for s, e, label in CHAPTERS:
        x = 80 + (W - 160) * s / TOTAL
        active = s <= t < e
        cv.circle(x, y, 5 if active else 3, fill=TEXT if active else MUTED)
        if active:
            cv.text(x + 12, y - 12, label, 18, 'medium', TEXT2, anchor='lb')
    cv.scene_a = a_save


def heading(cv, t, kicker, title, t0=0.0):
    cv.text(120, 92, kicker, 22, 'semibold', ORANGE, alpha=prog(t, t0, 0.5), anchor='ls')
    a = prog(t, t0 + 0.1, 0.6)
    cv.text(120, 150 - 16 * (1 - a), title, 52, 'bold', TEXT, alpha=a, anchor='ls')


def caption(cv, t, s, t0, y=H - 120, color=TEXT2, size=30):
    a = prog(t, t0, 0.6)
    cv.text(W / 2, y + 10 * (1 - a), s, size, 'medium', color, alpha=a, anchor='ms')


# ── 장면 1: 타이틀 ───────────────────────────────────────────
def scene_title(cv, t, D, dur):
    cx, cy = W / 2, H / 2
    a0 = prog(t, 0.2, 0.6)
    cv.text(cx, cy - 150, '진로탐색학점제  |  2026-2  중간 진행', 26, 'medium', ORANGE, alpha=a0, anchor='ms')
    a1 = prog(t, 0.5, 0.8)
    cv.text(cx, cy - 50 + 24 * (1 - a1), '물리 시뮬레이션으로 다시 검증한', 64, 'semibold', TEXT2, alpha=a1, anchor='ms')
    a2 = prog(t, 0.8, 0.8)
    cv.text(cx, cy + 50 + 24 * (1 - a2), 'VAE 무인체계 이상탐지', 92, 'bold', TEXT, alpha=a2, anchor='ms')
    lw = 560 * prog(t, 1.4, 0.9, ease_in_out)
    cv.line([(cx - lw / 2, cy + 100), (cx + lw / 2, cy + 100)], ORANGE, 4)
    a3 = prog(t, 2.0, 0.7)
    cv.text(cx, cy + 170, '김광빈  ·  AI응용학과        지도교수  공진우', 30, 'regular', TEXT2, alpha=a3, anchor='ms')
    a4 = prog(t, 2.6, 0.7)
    cv.text(cx, cy + 222, '제로데이 공격 · GPS 스푸핑 · MuJoCo UGV 시뮬레이션', 24, 'regular', MUTED, alpha=a4, anchor='ms')


# ── 장면 2: 출발점 — 너무 쉬운 데이터 ─────────────────────────
def scene_problem(cv, t, D, dur):
    heading(cv, t, '01  출발점', '기존 실험은 너무 쉬웠다')
    # 왼쪽: AUC 카운터
    x0, y0 = 120, 230
    cv.rect(x0, y0, x0 + 700, y0 + 520, fill=PANEL, radius=18, alpha=prog(t, 0.3, 0.6))
    cv.text(x0 + 50, y0 + 80, '난수 합성 데이터에서의 탐지 성능', 28, 'medium', TEXT2,
            alpha=prog(t, 0.5, 0.6), anchor='ls', base=PANEL)
    v = D.auc_nominal * prog(t, 0.8, 2.0, ease_in_out)
    cv.text(x0 + 50, y0 + 150, 'AUC', 34, 'semibold', MUTED, alpha=prog(t, 0.8, 0.5), anchor='ls', base=PANEL)
    cv.text(x0 + 46, y0 + 330, f'{v:.4f}', 170, 'bold', TEXT, alpha=prog(t, 0.8, 0.5), anchor='ls', base=PANEL)
    cv.text(x0 + 50, y0 + 410, '정상과 공격이 완벽히 분리됨', 28, 'regular', TEXT2,
            alpha=prog(t, 2.6, 0.6), anchor='ls', base=PANEL)
    a_warn = prog(t, 4.6, 0.6)
    cv.rect(x0 + 50, y0 + 440, x0 + 650, y0 + 490, fill=mix(RED, PANEL, 0.18), radius=10, alpha=a_warn, base=PANEL)
    cv.text(x0 + 70, y0 + 474, '좋은 모델이 아니라 쉬운 데이터였다', 26, 'semibold', RED,
            alpha=a_warn, anchor='ls', base=PANEL)

    # 오른쪽: gen_normal 과 같은 정의의 난수 속도 (매 스텝 독립 균등분포 1~3)
    px0, py0, pw, ph = 900, 230, 900, 520
    cv.rect(px0, py0, px0 + pw, py0 + ph, fill=PANEL, radius=18, alpha=prog(t, 0.5, 0.6))
    cv.text(px0 + 40, py0 + 70, '기존 "정상 주행" 데이터의 속도 (gen_normal 정의)', 26, 'medium', TEXT2,
            alpha=prog(t, 0.8, 0.6), anchor='ls', base=PANEL)
    rng = np.random.default_rng(42)
    sp = rng.uniform(1.0, 3.0, 120)
    gx0, gx1, gy0, gy1 = px0 + 60, px0 + pw - 40, py0 + 120, py0 + 400
    for k, yv in enumerate((1.0, 2.0, 3.0)):
        yy = gy1 - (yv - 0.5) / 3.0 * (gy1 - gy0)
        cv.line([(gx0, yy), (gx1, yy)], GRID, 1, alpha=prog(t, 0.8, 0.5), base=PANEL)
        cv.text(gx0 - 14, yy, f'{yv:.0f}', 18, 'regular', MUTED, alpha=prog(t, 0.8, 0.5), anchor='rm', base=PANEL)
    n = int(len(sp) * prog(t, 1.2, 3.0, ease_in_out))
    pts = [(gx0 + i / (len(sp) - 1) * (gx1 - gx0), gy1 - (sp[i] - 0.5) / 3.0 * (gy1 - gy0)) for i in range(n)]
    cv.line(pts, BLUE, 2.5, base=PANEL)
    cv.text(gx1, gy1 + 34, '타임스텝 (100 ms)', 18, 'regular', MUTED, alpha=prog(t, 1.2, 0.5), anchor='rs', base=PANEL)
    a_note = prog(t, 3.6, 0.6)
    cv.text(px0 + 40, py0 + 460, '매 스텝 독립 난수  →  궤적도, 관성도, 물리도 없음', 28, 'semibold', TEXT,
            alpha=a_note, anchor='ls', base=PANEL)
    caption(cv, t, '"이 결과가 실제 차량에서도 유지될까?"  →  물리 엔진 위에서 다시 확인하기로 함', 5.8)


# ── 장면 3: 시뮬레이션 구조 ──────────────────────────────────
def scene_pipeline(cv, t, D, dur):
    heading(cv, t, '02  방법', 'MuJoCo 물리 시뮬레이션 구축')
    boxes = [('물리 엔진', 'MuJoCo · 500 Hz', BLUE),
             ('센서 모델', 'GPS · IMU · 휠', BLUE),
             ('공격 주입', '측정값만 조작', ORANGE),
             ('제어기', '속은 GPS로 조향', BLUE)]
    bw, bh, gap = 330, 150, 90
    total_w = len(boxes) * bw + (len(boxes) - 1) * gap
    x = (W - total_w) / 2
    y = 400
    centers = []
    for i, (title, sub, col) in enumerate(boxes):
        a = prog(t, 0.6 + i * 0.45, 0.6)
        yy = y + 30 * (1 - a)
        cv.rect(x, yy, x + bw, yy + bh, fill=PANEL, outline=col, width=3, radius=16, alpha=a)
        cv.text(x + bw / 2, yy + 66, title, 38, 'bold', TEXT, alpha=a, anchor='ms', base=PANEL)
        cv.text(x + bw / 2, yy + 112, sub, 24, 'regular', TEXT2, alpha=a, anchor='ms', base=PANEL)
        centers.append((x, x + bw))
        x += bw + gap
    # 순방향 화살표
    for i in range(len(boxes) - 1):
        p = prog(t, 1.0 + i * 0.45, 0.5, ease_in_out)
        x0, x1 = centers[i][1] + 10, centers[i + 1][0] - 10
        if p > 0:
            cv.arrow((x0, y + bh / 2), (x0 + (x1 - x0) * p, y + bh / 2), TEXT2, 3, 14)
    # 되먹임 화살표 (제어기 → 물리 엔진)
    pf = prog(t, 2.8, 1.0, ease_in_out)
    yb = y + bh + 70
    xr = (centers[-1][0] + centers[-1][1]) / 2
    xl = (centers[0][0] + centers[0][1]) / 2
    if pf > 0:
        segs = [(xr, y + bh + 6), (xr, yb), (xl, yb), (xl, y + bh + 14)]
        L = [math.dist(segs[i], segs[i + 1]) for i in range(3)]
        remain = pf * sum(L)
        pts = [segs[0]]
        for i in range(3):
            if remain <= 0:
                break
            f = min(1.0, remain / L[i])
            pts.append((segs[i][0] + (segs[i + 1][0] - segs[i][0]) * f, segs[i][1] + (segs[i + 1][1] - segs[i][1]) * f))
            remain -= L[i]
        cv.line(pts[:-1] + [pts[-1]], ORANGE, 3)
        if pf >= 0.999:
            cv.arrow((xl, yb), (xl, y + bh + 12), ORANGE, 3, 16)
        cv.text((xl + xr) / 2, yb + 44, '휠 토크  →  차량이 실제로 움직임  (되먹임)', 26, 'medium', ORANGE,
                alpha=prog(t, 3.4, 0.6), anchor='ms')
    # 경로 위를 도는 펄스 점
    if t > 4.0:
        cyc = ((t - 4.0) / 2.4) % 1.0
        loop = [(xl, y + bh / 2), (xr, y + bh / 2), (xr, yb), (xl, yb), (xl, y + bh / 2)]
        L = [math.dist(loop[i], loop[i + 1]) for i in range(4)]
        s = cyc * sum(L)
        for i in range(4):
            if s <= L[i]:
                f = s / L[i]
                px = loop[i][0] + (loop[i + 1][0] - loop[i][0]) * f
                py = loop[i][1] + (loop[i + 1][1] - loop[i][1]) * f
                break
            s -= L[i]
        cv.circle(px, py, 9, fill=TEXT, alpha=prog(t, 4.0, 0.4))
    # 피처 출력
    a_out = prog(t, 4.4, 0.6)
    cv.text(W / 2, 300, '센서 모델 출력 10 Hz  →  피처 벡터  →  VAE 이상탐지', 28, 'medium', AQUA, alpha=a_out, anchor='ms')
    caption(cv, t, '공격은 차를 밀지 않는다.  측정값을 속이면 제어기가 스스로 경로를 벗어난다.', 5.4, color=TEXT, size=32)


# ── 장면 4: 실제 시뮬레이션 화면 ─────────────────────────────
CLIP_SRC_START, CLIP_SPEED, CLIP_FPS = 6.0, 2.0, 20  # 원본 10 s 부터 2 배속
CLIP_SIZE = (1600, 600)


def scene_footage(cv, t, D, dur, clip):
    heading(cv, t, '03  시연', '실제 시뮬레이션 화면  (2배속)')
    src_t = CLIP_SRC_START + CLIP_SPEED * max(0.0, t)
    fr = clip.get(int(src_t * CLIP_FPS))
    x, y = (W - CLIP_SIZE[0]) / 2, 210
    cv.rect(x - 4, y - 4, x + CLIP_SIZE[0] + 4, y + CLIP_SIZE[1] + 4, fill=GRID, radius=6)
    if fr is not None:
        cv.paste(fr, x, y)
    spoof_t = 15.0  # make_demo_video.py: 15 s 부터 GPS 편향 (4, 0) m
    if src_t < spoof_t:
        caption(cv, t, '정상 주행 — 파란 실제 경로와 빨간 "GPS가 믿는 경로"가 겹쳐 웨이포인트를 따라감', 0.3, y=H - 150)
    else:
        t_on = (spoof_t - CLIP_SRC_START) / CLIP_SPEED
        a = prog(t, t_on, 0.4)
        cv.rect(W / 2 - 230, H - 230, W / 2 + 230, H - 186, fill=mix(ORANGE, BG, 0.2), radius=10, alpha=a)
        cv.text(W / 2, H - 197, 'GPS 스푸핑 시작  (+4 m 편향)', 26, 'bold', ORANGE, alpha=a, anchor='ms',
                base=mix(ORANGE, BG, 0.2))
        caption(cv, t, '제어기는 경로를 잘 따라간다고 믿지만, 실제 차량(파랑)은 4 m 옆으로 밀려 달린다', t_on + 0.3, y=H - 130)


# ── 장면 5: 궤적 애니메이션 ──────────────────────────────────
def scene_trajectory(cv, t, D, dur):
    heading(cv, t, '04  결과 1', '속은 GPS를 믿고 달린 궤적')
    # 플롯 영역 (등축)
    px0, py0, size = 180, 205, 600
    xr, yr = (-16.0, 12.0), (-12.5, 12.5)
    scale = size / max(xr[1] - xr[0], yr[1] - yr[0])
    ox = px0 + (size - (xr[1] - xr[0]) * scale) / 2
    oy = py0 + (size - (yr[1] - yr[0]) * scale) / 2

    def m(x, y):
        return ox + (x - xr[0]) * scale, oy + (yr[1] - y) * scale

    cv.rect(px0 - 20, py0 - 20, px0 + size + 20, py0 + size + 20, fill=PANEL, radius=18, alpha=prog(t, 0.2, 0.5))
    wp = D.waypoints
    a_wp = prog(t, 0.5, 0.6)
    for i in range(len(wp)):
        cv.dashed(m(*wp[i]), m(*wp[(i + 1) % len(wp)]), MUTED, 2, 10, 8, alpha=a_wp, base=PANEL)
    for p in wp:
        cv.circle(*m(*p), 10, outline=YELLOW, width=3, alpha=a_wp, base=PANEL)

    p = prog(t, 1.0, 8.5, lambda x: clamp(x))  # 시뮬레이션 시간은 선형으로 재생
    n = max(2, int(len(D.traj_t) * p))
    sim_t = D.traj_t[n - 1]
    cv.line([m(*q) for q in D.gps_xy[:n:2]], ORANGE, 2.5, base=PANEL)
    cv.line([m(*q) for q in D.true_xy[:n]], BLUE, 4, base=PANEL)
    hx, hy = m(*D.true_xy[n - 1])
    cv.circle(hx, hy, 11, fill=BLUE, outline=TEXT, width=3, base=PANEL)
    gx, gy = m(*D.gps_xy[n - 1])
    cv.circle(gx, gy, 8, fill=ORANGE, base=PANEL)
    atk_i = int(np.argmax(D.traj_label == 1))
    if n > atk_i:
        ax_, ay_ = m(*D.true_xy[atk_i])
        k = (t - (1.0 + 8.5 * atk_i / len(D.traj_t)))
        cv.circle(ax_, ay_, 12 + 26 * clamp(k / 0.8), outline=ORANGE, width=3, alpha=1 - clamp(k / 0.8), base=PANEL)
        cv.circle(ax_, ay_, 9, fill=ORANGE, base=PANEL)

    # 범례
    lx, ly = px0 - 10, py0 + size + 62
    cv.line([(lx, ly), (lx + 40, ly)], BLUE, 4, alpha=a_wp)
    cv.text(lx + 52, ly, '실제 경로 (물리)', 22, 'regular', TEXT2, alpha=a_wp, anchor='lm')
    cv.line([(lx + 270, ly), (lx + 310, ly)], ORANGE, 3, alpha=a_wp)
    cv.text(lx + 322, ly, 'GPS가 믿는 경로 (제어기 입력)', 22, 'regular', TEXT2, alpha=a_wp, anchor='lm')

    # 오른쪽 계기판
    rx = 1020
    a = prog(t, 0.6, 0.6)
    cv.text(rx, 260, '시뮬레이션 시각', 26, 'medium', MUTED, alpha=a, anchor='ls')
    cv.text(rx, 340, f'{sim_t:5.1f} s', 72, 'bold', TEXT, alpha=a, anchor='ls')
    attacked = sim_t >= D.atk_t
    cv.text(rx, 420, '상태', 26, 'medium', MUTED, alpha=a, anchor='ls')
    st_col = ORANGE if attacked else AQUA
    st = f'GPS 스푸핑 중  ({D.atk_t:.0f} s~)' if attacked else '정상 주행'
    cv.rect(rx, 440, rx + 420, 496, fill=mix(st_col, BG, 0.18), radius=10, alpha=a)
    cv.text(rx + 20, 478, st, 30, 'bold', st_col, alpha=a, anchor='ls', base=mix(st_col, BG, 0.18))
    cv.text(rx, 580, '|GPS 위치 - 실제 위치|', 26, 'medium', MUTED, alpha=a, anchor='ls')
    cv.text(rx, 680, f'{D.err_smooth[n - 1]:.1f} m', 96, 'bold', ORANGE if attacked else TEXT, alpha=a, anchor='ls')
    caption(cv, t, '"GPS가 믿는 경로"는 계속 웨이포인트 위에 있다  →  제어기는 아무 문제가 없다고 판단', 7.4,
            y=H - 100, color=TEXT)


# ── 장면 6: 10 vs 12 피처 점수 곡선 ──────────────────────────
def score_panel(cv, t, x0, y0, w, h, scores, th, title, col, t0, dur, lo=0.03, hi=100.0, atk_t=20.0):
    a = prog(t, t0 - 0.4, 0.5)
    cv.rect(x0, y0, x0 + w, y0 + h, fill=PANEL, radius=14, alpha=a)
    gx0, gx1, gy0, gy1 = x0 + 90, x0 + w - 40, y0 + 70, y0 + h - 50
    tt = np.arange(len(scores)) * 0.1
    tmax = tt[-1]

    def m(ti, s):
        ly = (math.log10(clamp(s, lo, hi)) - math.log10(lo)) / (math.log10(hi) - math.log10(lo))
        return gx0 + ti / tmax * (gx1 - gx0), gy1 - ly * (gy1 - gy0)

    for v in (0.1, 1, 10, 100):
        if lo <= v <= hi:
            _, yy = m(0, v)
            cv.line([(gx0, yy), (gx1, yy)], GRID, 1, alpha=a, base=PANEL)
            cv.text(gx0 - 12, yy, f'{v:g}', 18, 'regular', MUTED, alpha=a, anchor='rm', base=PANEL)
    ax0, _ = m(atk_t, lo)
    cv.rect(ax0, gy0, gx1, gy1, fill=mix(ORANGE, PANEL, 0.10), alpha=a, base=PANEL)
    cv.text(ax0 + 8, gy0 + 22, '공격 구간', 18, 'medium', ORANGE, alpha=a, anchor='lm', base=mix(ORANGE, PANEL, 0.10))
    _, ty = m(0, th)
    cv.dashed((gx0, ty), (gx1, ty), TEXT2, 2, 8, 6, alpha=a, base=PANEL)
    cv.text(gx1, ty - 10, f'임계값 {th:.3f}', 18, 'regular', TEXT2, alpha=a, anchor='rs', base=PANEL)
    cv.text(x0 + 30, y0 + 44, title, 26, 'semibold', TEXT, alpha=a, anchor='ls', base=PANEL)
    p = prog(t, t0, dur, lambda x: clamp(x))
    start = 19  # 첫 윈도우가 차기 전 구간은 점수 없음
    n = start + int((len(scores) - start) * p)
    pts = [m(tt[i], scores[i]) for i in range(start, n)]
    cv.line(pts, col, 3, base=PANEL)
    cv.text(gx1, gy1 + 34, '시간 (s)', 18, 'regular', MUTED, alpha=a, anchor='rs', base=PANEL)
    return p


def scene_scores(cv, t, D, dur):
    heading(cv, t, '04  결과 2', '같은 공격, 피처만 바꿔 다시 학습한 VAE')
    k = 'gps_spoof_bias'
    w, h = 1680, 330
    score_panel(cv, t, 120, 200, w, h, D.f10[f'{k}_scores'], float(D.f10['threshold']),
                '기존 10개 피처 (속도 · 가속도 · 명령)', BLUE, 0.8, 3.5)
    score_panel(cv, t, 120, 560, w, h, D.f12[f'{k}_scores'], float(D.f12['threshold']),
                '12개 피처 (+ GPS 위치 - 휠 오도메트리 위치 잔차)', AQUA, 2.2, 3.5)
    caption(cv, t, '속도만 보면 경로가 통째로 밀려도 보이지 않는다.  위치를 비교하자 공격 순간 점수가 치솟았다.', 6.2,
            y=H - 96, color=TEXT, size=28)


# ── 장면 7: Recall 막대 비교 ─────────────────────────────────
def scene_recall(cv, t, D, dur):
    heading(cv, t, '04  결과 3', '시뮬레이션 GPS 공격 3종 — 탐지율(Recall)')
    gx0, gx1, gy0, gy1 = 260, 1660, 240, 820
    a = prog(t, 0.3, 0.5)
    for v in (0, 0.25, 0.5, 0.75, 1.0):
        yy = gy1 - v * (gy1 - gy0)
        cv.line([(gx0, yy), (gx1, yy)], GRID, 1, alpha=a)
        cv.text(gx0 - 16, yy, f'{v:.2f}', 20, 'regular', MUTED, alpha=a, anchor='rm')
    gw = (gx1 - gx0) / len(D.sim_attacks)
    bw = 150
    for i, (_, name) in enumerate(D.sim_attacks):
        cx = gx0 + gw * (i + 0.5)
        cv.text(cx, gy1 + 50, name, 30, 'semibold', TEXT, alpha=a, anchor='ms')
        for j, (vals, col, t0) in enumerate(((D.recall10, BLUE, 0.8), (D.recall12, AQUA, 2.0))):
            p = prog(t, t0 + i * 0.25, 1.0)
            v = vals[i] * p
            bx = cx - bw - 6 if j == 0 else cx + 6
            top = gy1 - v * (gy1 - gy0)
            if v > 0.002:
                cv.rect(bx, top, bx + bw, gy1, fill=col, radius=6, alpha=1.0)
                cv.rect(bx, max(top, gy1 - 6), bx + bw, gy1, fill=col)
            else:
                cv.line([(bx, gy1), (bx + bw, gy1)], col, 4, alpha=prog(t, t0 + i * 0.25, 0.3))
            cv.text(bx + bw / 2, top - 14, f'{v:.3f}', 28, 'bold', TEXT, alpha=prog(t, t0 + i * 0.25, 0.4), anchor='ms')
    ly = 190
    cv.rect(gx1 - 520, ly - 16, gx1 - 496, ly + 8, fill=BLUE, radius=4, alpha=a)
    cv.text(gx1 - 486, ly, '기존 10개 피처', 24, 'medium', TEXT2, alpha=a, anchor='lm')
    cv.rect(gx1 - 280, ly - 16, gx1 - 256, ly + 8, fill=AQUA, radius=4, alpha=a)
    cv.text(gx1 - 246, ly, '12개 피처 (+위치 잔차)', 24, 'medium', TEXT2, alpha=a, anchor='lm')
    caption(cv, t, '문제는 모델이 아니라 피처 설계였다 — 난수 데이터에서는 드러나지 않던 결함', 4.2, y=H - 112, color=TEXT)
    cv.text(W / 2, H - 76, '시뮬레이션 결과 · 평지 / 바퀴 미끄러짐 없음 / 단일 seed', 20, 'regular', MUTED,
            alpha=prog(t, 4.6, 0.5), anchor='ms')


# ── 장면 8: 제로데이 ─────────────────────────────────────────
def scene_zeroday(cv, t, D, dur):
    heading(cv, t, '05  제로데이', '학습에 없던 공격 5종 — 못 잡은 것도 있다')
    # 왼쪽: 가로 막대
    x0, y0, lw = 120, 220, 760
    labx = x0 + 250
    bmax = lw - 260
    for i, (name, rec, delay) in enumerate(D.zeroday):
        yy = y0 + 40 + i * 104
        a = prog(t, 0.5 + i * 0.25, 0.5)
        miss = rec == 0
        cv.text(labx - 20, yy + 22, name, 26, 'semibold', RED if miss else TEXT, alpha=a, anchor='rm')
        cv.rect(labx, yy, labx + bmax, yy + 44, fill=PANEL, radius=8, alpha=a)
        p = prog(t, 0.7 + i * 0.25, 0.9)
        if rec > 0:
            cv.rect(labx, yy, labx + bmax * rec * p, yy + 44, fill=AQUA, radius=8, alpha=a, base=PANEL)
            dtxt = f'R {rec * p:.3f}   지연 {delay:,} ms'
            cv.text(labx + 16, yy + 23, dtxt, 22, 'bold', BG, alpha=a * clamp(p * 3), anchor='lm', base=AQUA)
        else:
            cv.text(labx + 16, yy + 23, 'R 0.000   미탐', 22, 'bold', RED, alpha=a, anchor='lm', base=PANEL)
    # 오른쪽: 전체 센서 동결 점수
    px0, py0, pw, ph = 960, 220, 840, 560
    a = prog(t, 3.2, 0.6)
    cv.rect(px0, py0, px0 + pw, py0 + ph, fill=PANEL, radius=16, alpha=a)
    cv.text(px0 + 30, py0 + 48, '전체 센서 동결 — 이상 점수', 26, 'semibold', TEXT, alpha=a, anchor='ls', base=PANEL)
    sc = D.zd_freeze_all
    gx0, gx1, gy0, gy1 = px0 + 80, px0 + pw - 40, py0 + 90, py0 + ph - 60
    lo, hi = 0.3, 1.6

    def m(i, s):
        return gx0 + i / (len(sc) - 1) * (gx1 - gx0), gy1 - (clamp(s, lo, hi) - lo) / (hi - lo) * (gy1 - gy0)

    ax_, _ = m(D.zd_atk_start, lo)
    cv.rect(ax_, gy0, gx1, gy1, fill=mix(ORANGE, PANEL, 0.10), alpha=a, base=PANEL)
    _, ty = m(0, D.zd_th)
    cv.dashed((gx0, ty), (gx1, ty), TEXT2, 2, 8, 6, alpha=a, base=PANEL)
    cv.text(gx0 + 6, ty - 10, f'임계값 {D.zd_th:.3f}', 18, 'regular', TEXT2, alpha=a, anchor='ls', base=PANEL)
    _, my = m(0, D.zd_normal_min)
    cv.dashed((gx0, my), (gx1, my), RED, 2, 4, 6, alpha=a, base=PANEL)
    cv.text(gx0 + 6, my + 24, f'정상 데이터 최솟값 {D.zd_normal_min:.3f}', 18, 'regular', RED, alpha=a, anchor='ls', base=PANEL)
    p = prog(t, 3.6, 2.6, lambda x: clamp(x))
    n = 20 + int((len(sc) - 20) * p)
    cv.line([m(i, sc[i]) for i in range(20, n)], ORANGE, 2.5, base=PANEL)
    cv.text(gx1, gy1 + 36, '타임스텝', 18, 'regular', MUTED, alpha=a, anchor='rs', base=PANEL)
    a2 = prog(t, 6.4, 0.6)
    cv.text(px0 + 30, py0 + ph + 56, '센서가 멈추면 입력이 "너무 규칙적"이라 VAE가 가장 쉽게 복원한다.', 24, 'medium', TEXT,
            alpha=a2, anchor='ls')
    cv.text(px0 + 30, py0 + ph + 94, '점수가 정상보다 낮아져 "높으면 이상" 규칙으로는 구조적으로 볼 수 없다.', 24, 'medium', TEXT,
            alpha=prog(t, 6.9, 0.6), anchor='ls')
    caption(cv, t, 'GPS만 조작하면 GPS–IMU 불일치로 즉시 잡히지만, 모든 센서를 함께 속이면 놓친다', 7.6, y=H - 90, color=TEXT2, size=26)


# ── 장면 9: 정리 ─────────────────────────────────────────────
def scene_summary(cv, t, D, dur):
    heading(cv, t, '06  정리', '이번 단계에서 확인한 것')
    r10, r12 = D.recall10, D.recall12
    items = [
        (ORANGE, '난수 데이터의 AUC 1.0은 착시였다',
         '물리 시뮬레이션에서 기존 피처 설계의 사각지대가 드러남'),
        (AQUA, 'GPS–오도메트리 위치 잔차로 탐지율 회복',
         '  /  '.join(f'{a:.2f} → {b:.2f}' for a, b in zip(r10, r12)) + '   (편향 · 드리프트 · 동결)'),
        (RED, '전체 센서 동결은 아직 미탐',
         '다음 단계: 하한 임계값 · 분산 0 검사 추가, 위치 잔차 시정수 개선'),
    ]
    for i, (col, head, sub) in enumerate(items):
        a = prog(t, 0.6 + i * 0.7, 0.6)
        y = 270 + i * 190
        x = 160 + 40 * (1 - a)
        cv.rect(x, y, x + 10, y + 120, fill=col, radius=4, alpha=a)
        cv.text(x + 44, y + 50, head, 44, 'bold', TEXT, alpha=a, anchor='ls')
        cv.text(x + 44, y + 102, sub, 28, 'regular', TEXT2, alpha=a, anchor='ls')
    a = prog(t, 3.2, 0.7)
    cv.text(W / 2, H - 120, 'github.com/gawbi/haegeum-addon', 26, 'medium', TEXT2, alpha=a, anchor='ms')
    cv.text(W / 2, H - 82, '모든 수치는 저장소의 결과 파일에서 생성 · 시뮬레이션 결과는 실제 차량 검증이 아님', 20, 'regular', MUTED,
            alpha=a, anchor='ms')


# ── 타임라인 ─────────────────────────────────────────────────
SCENES = [
    # (길이 s, 함수, 장 제목)
    (5.5, scene_title, '시작'),
    (9.0, scene_problem, '출발점'),
    (9.0, scene_pipeline, '방법'),
    (11.5, scene_footage, '시연'),
    (11.0, scene_trajectory, '궤적'),
    (9.5, scene_scores, '점수'),
    (8.0, scene_recall, '탐지율'),
    (11.0, scene_zeroday, '제로데이'),
    (7.5, scene_summary, '정리'),
]
FADE = 0.45


def build_timeline():
    global TOTAL
    s = 0.0
    tl = []
    for dur, fn, label in SCENES:
        tl.append((s, dur, fn))
        CHAPTERS.append((s, s + dur, label))
        s += dur
    TOTAL = s
    return tl


def render_frame(t, tl, D, clip):
    cv = Canvas()
    for s, dur, fn in tl:
        if s <= t < s + dur:
            lt = t - s
            cv.scene_a = clamp(min(lt / FADE, (dur - lt) / FADE)) if fn is not scene_title else clamp(min(1.0, (dur - lt) / FADE))
            if fn is scene_footage:
                fn(cv, lt, D, dur, clip)
            else:
                fn(cv, lt, D, dur)
            break
    chrome(cv, t)
    return cv.final()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--preview', nargs='*', type=float, help='지정 시각(초)만 PNG 로 저장')
    ap.add_argument('--out', default=str(OUT / 'motion_summary.mp4'))
    ap.add_argument('--preview-dir', default=str(OUT / 'motion_preview'))
    args = ap.parse_args()

    D = Data()
    tl = build_timeline()
    clip = VideoClip(OUT / 'demo_gps_spoof.mp4', CLIP_SIZE)
    print(f'[타임라인] 총 {TOTAL:.1f} s, 장면 {len(tl)}개')

    if args.preview is not None:
        pdir = Path(args.preview_dir)
        pdir.mkdir(parents=True, exist_ok=True)
        for ts in sorted(args.preview):
            img = render_frame(ts, tl, D, clip)
            p = pdir / f'preview_{ts:05.1f}s.png'
            img.save(p)
            print('[미리보기]', p)
        return

    n_frames = int(round(TOTAL * FPS))
    writer = imageio.get_writer(args.out, fps=FPS, codec='libx264', quality=None, macro_block_size=8,
                                pixelformat='yuv420p', output_params=['-crf', '18', '-preset', 'medium'])
    for i in range(n_frames):
        writer.append_data(np.asarray(render_frame(i / FPS, tl, D, clip)))
        if i % FPS == 0:
            print(f'\r  렌더링 {i / FPS:5.1f} / {TOTAL:.1f} s', end='', flush=True)
    writer.close()
    print(f'\n[저장] {args.out}  ({n_frames} 프레임, {TOTAL:.1f} s)')


if __name__ == '__main__':
    main()
