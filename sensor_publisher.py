#!/usr/bin/env python3
"""
DAH 2026 해금팀 — 센서 데이터 퍼블리셔 노드
anomaly_detector.py가 수신할 /sensor_data/{robot_id} 토픽 발행

UAV 8개 피처: gps_vel_x/y, imu_ax/ay, residual_x/y, baro_alt, pitch_rate
UGV 10개 피처: gps_vel_x/y, imu_ax/ay, residual_x/y, wheel_vel_l/r, cmd_vel_x/w

실제 환경에서는 MAVLink / ROS2 센서 토픽에서 읽어오면 됨.
지금은 정상 비행/주행 패턴 시뮬레이션으로 발행.
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
import numpy as np
import time


class SensorPublisher(Node):

    def __init__(self):
        super().__init__('sensor_publisher')

        self.declare_parameter('platform', 'ugv')
        self.declare_parameter('robot_id', 'ugv_01')
        self.declare_parameter('rate_hz',  10.0)

        self.platform  = self.get_parameter('platform').value
        self.robot_id  = self.get_parameter('robot_id').value
        rate           = self.get_parameter('rate_hz').value

        self.publisher = self.create_publisher(
            Float32MultiArray,
            f'/sensor_data/{self.robot_id}',
            10
        )

        self.t = 0.0
        self.dt = 1.0 / rate
        self.create_timer(self.dt, self.publish_sensor)

        self.get_logger().info(
            f'SensorPublisher 시작 | platform={self.platform} | '
            f'robot={self.robot_id} | {rate}Hz'
        )

    def publish_sensor(self):
        self.t += self.dt

        if self.platform == 'uav':
            data = self._uav_sensor()
        else:
            data = self._ugv_sensor()

        msg = Float32MultiArray()
        msg.data = [float(v) for v in data]
        self.publisher.publish(msg)

    def _uav_sensor(self):
        """
        정상 UAV 순항 비행 시뮬레이션
        실제 환경에서는 MAVLink GLOBAL_POSITION_INT, HIGHRES_IMU,
        ALTITUDE, ATTITUDE 메시지에서 읽어오면 됨
        """
        t = self.t
        gps_vel_x  = 5.0  + np.random.normal(0, 0.05)
        gps_vel_y  = np.sin(t * 0.2) * 0.5 + np.random.normal(0, 0.03)
        imu_ax     = gps_vel_x + np.random.normal(0, 0.02)
        imu_ay     = gps_vel_y + np.random.normal(0, 0.02)
        residual_x = gps_vel_x - imu_ax
        residual_y = gps_vel_y - imu_ay
        baro_alt   = 50.0  + np.random.normal(0, 0.1)
        pitch_rate = np.random.normal(0, 0.01)

        return [gps_vel_x, gps_vel_y, imu_ax, imu_ay,
                residual_x, residual_y, baro_alt, pitch_rate]

    def _ugv_sensor(self):
        """
        정상 UGV 직선 주행 시뮬레이션
        실제 환경에서는 /fix, /imu/data, /odom, /cmd_vel 토픽에서 읽어오면 됨
        """
        t = self.t
        gps_vel_x   = 0.5  + np.random.normal(0, 0.01)
        gps_vel_y   = np.sin(t * 0.1) * 0.1 + np.random.normal(0, 0.01)
        imu_ax      = gps_vel_x + np.random.normal(0, 0.005)
        imu_ay      = gps_vel_y + np.random.normal(0, 0.005)
        residual_x  = gps_vel_x - imu_ax
        residual_y  = gps_vel_y - imu_ay
        wheel_vel_l = 0.5  + np.random.normal(0, 0.01)
        wheel_vel_r = 0.5  + np.random.normal(0, 0.01)
        cmd_vel_x   = 0.5  + np.random.normal(0, 0.01)
        cmd_vel_w   = 0.0  + np.random.normal(0, 0.005)

        return [gps_vel_x, gps_vel_y, imu_ax, imu_ay,
                residual_x, residual_y,
                wheel_vel_l, wheel_vel_r, cmd_vel_x, cmd_vel_w]


def main():
    rclpy.init()
    node = SensorPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
