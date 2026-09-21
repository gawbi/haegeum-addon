#!/usr/bin/env python3
"""
DAH 2026 해금팀 — UGV 순찰 경로 관리
haegeum 레포 src/ugv_agent/ugv_agent/patrol.py 에 배치 예정

uav_agent/patrol.py 와 동일한 구조
"""

import math


class UGVPatrol:
    """
    UGV 순찰 경로 웨이포인트 관리.
    100m × 100m 정사각형 순찰 루트 기본값.
    """

    def __init__(self, waypoints=None):

        if waypoints is None:
            self.waypoints = [
                (0.0,   0.0),
                (50.0,  0.0),
                (50.0, 50.0),
                (0.0,  50.0),
            ]
        else:
            self.waypoints = waypoints

        self.index = 0

    def next_waypoint(self):
        wp = self.waypoints[self.index]
        self.index = (self.index + 1) % len(self.waypoints)
        return wp

    def current_waypoint(self):
        return self.waypoints[self.index]

    def distance_to_next(self, x, y):
        nx, ny = self.waypoints[self.index]
        return math.sqrt((nx - x) ** 2 + (ny - y) ** 2)
