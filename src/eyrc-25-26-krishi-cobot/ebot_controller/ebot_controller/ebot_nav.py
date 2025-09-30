#!/usr/bin/env python3
"""
ebot_nav.py
Robust waypoint navigator for eBot Task1 (LiDAR + Odometry based).
No external tf_transformations dependency; computes yaw from quaternion directly.

Waypoints (order):
 P1 : [-1.53, -1.95,  1.57]
 P2 : [ 0.13,  1.24,  0.00]
 P3 : [ 0.38, -3.32, -1.57]

Position tol: 0.3 m
Yaw tol: 10 degrees
"""
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
import math
import math as m
import statistics

def quat_to_yaw(qx, qy, qz, qw):
    # yaw (z-axis rotation) from quaternion
    # safe stable formula
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    return math.atan2(siny_cosp, cosy_cosp)

def clamp(v, lo, hi):
    return max(lo, min(hi, v))

class EBotNavigator(Node):
    def __init__(self):
        super().__init__('ebot_nav')

        # Topics
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.odom_sub = self.create_subscription(Odometry, '/odom', self.odom_cb, 10)
        self.scan_sub = self.create_subscription(LaserScan, '/scan', self.scan_cb, 10)

        # Waypoints
        self.waypoints = [
            (-1.53, -1.95,  1.57),
            ( 0.13,  1.24,  0.00),
            ( 0.38, -3.32, -1.57),
        ]
        self.wp_idx = 0

        # robot state
        self.x = None
        self.y = None
        self.yaw = None
        self.scan = []

        # tolerances
        self.pos_tol = 0.3              # meters
        self.yaw_tol = math.radians(10) # radians

        # controller gains & limits (tune these)
        self.K_lin = 0.6
        self.K_ang = 1.2
        self.max_lin = 0.35
        self.max_ang = 1.2

        # lidar safety distances
        self.front_stop = 0.22    # stop if obstacle closer than this [m]
        self.front_slow = 0.6     # start slowing if obstacle within this [m]
        self.side_stop = 0.20

        # control loop
        self.timer = self.create_timer(0.1, self.control_loop)  # 10 Hz
        self.get_logger().info('EBotNavigator node started.')

    def odom_cb(self, msg: Odometry):
        self.x = msg.pose.pose.position.x
        self.y = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        self.yaw = quat_to_yaw(q.x, q.y, q.z, q.w)

    def scan_cb(self, msg: LaserScan):
        # store raw ranges; handle empty by keeping old
        self.scan = list(msg.ranges)

    def safe_min(self, vals):
        # return min ignoring inf/nan; return large if none valid
        cleaned = [v for v in vals if v is not None and not math.isinf(v) and not math.isnan(v)]
        if not cleaned:
            return float('inf')
        # sometimes keep median to avoid spurious tiny readings
        return min(cleaned)

    def control_loop(self):
        # must have odom and scan to act
        if self.x is None or self.scan is None:
            return

        if self.wp_idx >= len(self.waypoints):
            # finished
            self.publish_stop()
            self.get_logger().info('All waypoints completed.')
            return

        gx, gy, gyaw = self.waypoints[self.wp_idx]

        # compute errors
        dx = gx - self.x
        dy = gy - self.y
        dist = math.hypot(dx, dy)
        desired_theta = math.atan2(dy, dx)
        ang_err = math.atan2(math.sin(desired_theta - self.yaw), math.cos(desired_theta - self.yaw))
        final_yaw_err = math.atan2(math.sin(gyaw - self.yaw), math.cos(gyaw - self.yaw))

        # --- LiDAR sectors robustly computed ---
        N = len(self.scan)
        if N == 0:
            front_min = left_min = right_min = float('inf')
        else:
            mid = N // 2
            # pick windows (indices) relative to mid (front). If LiDAR definition differs,
            # these windows are conservative +/- 30 deg depending on sensor resolution.
            window = max(1, int(0.10 * N))  # small adaptive window (~10% of scan)
            # but we'll also use fixed slice sizes if N large
            slice_half = 15
            # safe indexing:
            def slice_vals(center, half):
                # wrap indices
                vals = []
                for i in range(center-half, center+half+1):
                    vals.append(self.scan[i % N])
                return vals

            front_vals = slice_vals(mid, slice_half)
            left_vals  = slice_vals((mid + N//4) % N, slice_half)
            right_vals = slice_vals((mid - N//4) % N, slice_half)

            front_min = self.safe_min(front_vals)
            left_min  = self.safe_min(left_vals)
            right_min = self.safe_min(right_vals)

        # Debug (reduced verbosity): log occasionally
        if self.get_clock().now().nanoseconds % 10 == 0:
            self.get_logger().debug(f'WP{self.wp_idx+1}: dist={dist:.3f}, ang_err={math.degrees(ang_err):.1f}°, front={front_min:.2f}, left={left_min:.2f}, right={right_min:.2f}')

        # Emergency stop if obstacle too close in front or sides
        if front_min < self.front_stop or left_min < self.side_stop or right_min < self.side_stop:
            # stop and yaw away slightly
            self.get_logger().warn('Obstacle too close! Stopping and rotating slightly.')
            t = Twist()
            t.linear.x = 0.0
            # rotate away from closest side
            if left_min < right_min:
                t.angular.z = -0.5
            else:
                t.angular.z = 0.5
            self.cmd_pub.publish(t)
            return

        # Controller: proportional angular & linear
        # Angular always uses P-control (so robot can correct while moving).
        ang_cmd = clamp(self.K_ang * ang_err, -self.max_ang, self.max_ang)

        # Linear: only move forward when roughly facing the waypoint (loose tolerance),
        # and reduce speed if close to obstacles
        lin_cmd = 0.0
        facing_threshold = math.radians(35)  # allow movement while not perfectly aligned
        if dist > self.pos_tol:
            if abs(ang_err) < facing_threshold:
                # scale linear by distance
                lin_cmd = clamp(self.K_lin * dist, 0.0, self.max_lin)
                # slow if obstacle in front-medium range
                if front_min < self.front_slow:
                    # scale down smoothly
                    factor = max(0.0, (front_min - self.front_stop) / (self.front_slow - self.front_stop))
                    lin_cmd *= factor
            else:
                lin_cmd = 0.0  # do not move forward if angle error large; let angular component correct

        # If close to waypoint position, align to final yaw
        if dist <= self.pos_tol:
            # in position: rotate to final yaw only
            if abs(final_yaw_err) > self.yaw_tol:
                lin_cmd = 0.0
                ang_cmd = clamp(self.K_ang * final_yaw_err, -self.max_ang, self.max_ang)
            else:
                # waypoint complete
                self.get_logger().info(f'Waypoint {self.wp_idx+1} reached (pos & yaw OK).')
                self.wp_idx += 1
                self.publish_stop()
                return

        # Publish cmd
        cmd = Twist()
        cmd.linear.x = lin_cmd
        cmd.angular.z = ang_cmd
        self.cmd_pub.publish(cmd)

    def publish_stop(self):
        t = Twist()
        t.linear.x = 0.0
        t.angular.z = 0.0
        self.cmd_pub.publish(t)

def main(args=None):
    rclpy.init(args=args)
    node = EBotNavigator()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.publish_stop()
        node.destroy_node()
        rclpy.shutdown()

