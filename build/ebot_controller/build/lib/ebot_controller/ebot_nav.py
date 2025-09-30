#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
import math
from tf_transformations import euler_from_quaternion

class EbotNavigator(Node):
    def __init__(self):
        super().__init__('ebot_nav')

        # Publisher & Subscribers
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.odom_sub = self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
        self.scan_sub = self.create_subscription(LaserScan, '/scan', self.scan_callback, 10)

        # Robot state
        self.x = 0.0
        self.y = 0.0
        self.yaw = 0.0
        self.scan_ranges = []

        # Waypoints [x, y, yaw]
        self.waypoints = [
            [-1.53, -1.95, 1.57],
            [0.13, 1.24, 0.0],
            [0.38, -3.32, -1.57]
        ]
        self.current_wp_index = 0

        # Tolerances
        self.pos_tol = 0.3
        self.yaw_tol = math.radians(10)  # 10 degrees

        # Controller gains
        self.kp_linear = 0.35
        self.kp_angular = 1.0
        self.max_lin = 0.35
        self.max_ang = 1.2

        # Obstacle detection
        self.obst_stop_distance = 0.4
        self.obst_slow_distance = 0.8

        # Timer for control loop
        self.timer = self.create_timer(0.1, self.control_loop)

    def odom_callback(self, msg):
        self.x = msg.pose.pose.position.x
        self.y = msg.pose.pose.position.y
        quat = msg.pose.pose.orientation
        _, _, self.yaw = euler_from_quaternion([quat.x, quat.y, quat.z, quat.w])

    def scan_callback(self, msg):
        self.scan_ranges = msg.ranges

    def control_loop(self):
        if self.current_wp_index >= len(self.waypoints):
            twist = Twist()
            self.cmd_pub.publish(twist)
            self.get_logger().info("All waypoints reached!")
            return

        target = self.waypoints[self.current_wp_index]
        target_x, target_y, target_yaw = target

        # Compute errors
        dx = target_x - self.x
        dy = target_y - self.y
        distance_error = math.hypot(dx, dy)
        desired_yaw = math.atan2(dy, dx)
        yaw_error = self.normalize_angle(desired_yaw - self.yaw)
        final_yaw_error = self.normalize_angle(target_yaw - self.yaw)

        # LiDAR distances
        if self.scan_ranges:
            front_dist = min(min(self.scan_ranges[0:10]), min(self.scan_ranges[-10:]))
            left_dist = min(self.scan_ranges[60:120])
            right_dist = min(self.scan_ranges[240:300])
        else:
            front_dist = left_dist = right_dist = float('inf')

        twist = Twist()

        # Stop if obstacle too close
        if front_dist < self.obst_stop_distance or left_dist < self.obst_stop_distance or right_dist < self.obst_stop_distance:
            twist.linear.x = 0.0
            twist.angular.z = self.max_ang * 0.5  # small rotation to avoid
            self.cmd_pub.publish(twist)
            return

        # Sequential rotate-then-move
        if distance_error > self.pos_tol:
            if abs(yaw_error) > 0.2:  # rotate on spot
                twist.linear.x = 0.0
                twist.angular.z = max(-self.max_ang, min(self.max_ang, self.kp_angular * yaw_error))
            else:
                # move forward with yaw correction
                lin_vel = self.kp_linear * distance_error
                lin_vel = min(lin_vel, self.max_lin)
                # slow down near front obstacles
                if front_dist < self.obst_slow_distance:
                    lin_vel *= front_dist / self.obst_slow_distance
                twist.linear.x = lin_vel
                twist.angular.z = max(-self.max_ang, min(self.max_ang, self.kp_angular * yaw_error))
        else:
            # Position reached, align final yaw
            if abs(final_yaw_error) > self.yaw_tol:
                twist.linear.x = 0.0
                twist.angular.z = max(-self.max_ang, min(self.max_ang, self.kp_angular * final_yaw_error))
            else:
                self.get_logger().info(f"Reached waypoint {self.current_wp_index + 1}")
                self.current_wp_index += 1
                twist = Twist()  # stop for a moment

        self.cmd_pub.publish(twist)

    @staticmethod
    def normalize_angle(angle):
        while angle > math.pi:
            angle -= 2*math.pi
        while angle < -math.pi:
            angle += 2*math.pi
        return angle

def main(args=None):
    rclpy.init(args=args)
    node = EbotNavigator()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

