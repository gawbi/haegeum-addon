#!/usr/bin/env python3
"""
DAH 2026 해금팀 — UGV Agent ROS2 노드
위치: L1 물리 실행층 / L2 중대 엣지 클라우드 연동

haegeum 레포 src/ugv_agent/ugv_agent/ 에 배치 예정
uav_agent/uav_agent.py 와 동일한 구조로 작성

토픽:
  발행: /robot_status (RobotStatus) → ai_commander
  구독: /assignment  (Assignment)   ← ai_commander

이상탐지는 anomaly_detector.py 가 별도 노드로 동작
  /target_detected → ai_commander → /assignment → ugv_agent
"""

import rclpy
from rclpy.node import Node

from geometry_msgs.msg import Pose
from haegeum_interfaces.msg import RobotStatus
from haegeum_interfaces.msg import Assignment


class UGVAgent(Node):

    def __init__(self):

        super().__init__("ugv_01")

        self.robot_id = "ugv_01"

        self.state = "PATROL"
        self.current_task = "NONE"

        self.x = 0.0
        self.y = 0.0

        self.publisher = self.create_publisher(
            RobotStatus,
            "/robot_status",
            10
        )

        self.subscription = self.create_subscription(
            Assignment,
            "/assignment",
            self.assignment_callback,
            10
        )

        self.timer = self.create_timer(
            1.0,
            self.publish_status
        )

        self.get_logger().info("UGV Agent Started")

    def publish_status(self):

        msg = RobotStatus()

        msg.robot_id = self.robot_id
        msg.robot_type = "UGV"

        msg.pose = Pose()

        msg.pose.position.x = self.x
        msg.pose.position.y = self.y
        msg.pose.position.z = 0.0

        msg.battery = 100.0

        msg.busy = self.state != "PATROL"

        msg.state = self.state

        msg.current_task = self.current_task

        self.publisher.publish(msg)

    def assignment_callback(self, msg: Assignment):

        if msg.assigned_robot != self.robot_id:
            return

        self.state = "RESPOND"

        self.current_task = msg.target_id

        self.get_logger().info(
            f"Mission Assigned -> {msg.target_id} | type={msg.mission_type} | "
            f"target=({msg.target_position.x:.1f}, {msg.target_position.y:.1f})"
        )


def main():

    rclpy.init()

    node = UGVAgent()

    rclpy.spin(node)

    node.destroy_node()

    rclpy.shutdown()


if __name__ == "__main__":
    main()
