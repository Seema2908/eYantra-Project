#!/usr/bin/env python3

import PyKDL
import numpy as np
import tf_transformations
import urdf_parser_py
from urdf_parser_py.urdf import URDF
from kdl_parser_py import treeFromUrdfModel
# from urdfpy import Mesh

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, TransformStamped
import os
import trimesh
import fcl
# import tf_transformations
from scipy.spatial.transform import Rotation
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray, Float32, Bool
from tf2_ros import TransformListener, Buffer
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point
from geometry_msgs.msg import PoseStamped
from ament_index_python.packages import get_package_share_directory
import os
# import time

ADJACENT_PAIRS = {
    ("base_link_inertia", "shoulder_link"),
    ("base_link_inertia", "upper_arm_link"),
    ("base_link", "shoulder_link"),
    ("shoulder_link", "upper_arm_link"),
    ("upper_arm_link", "forearm_link"),
    ("forearm_link", "wrist_1_link"),
    ("wrist_1_link", "wrist_2_link"),
    ("wrist_2_link", "wrist_3_link"),
}



PLANE_SIZE = 1.0        # Size of collision planes
PLANE_THICKNESS = 0.01  # thickness of planes
PLANE_ALPHA = 0.3       # Transparency (0.0 = invisible, 1.0 = solid)


class CartesianServoNode(Node):
    def __init__(self):
        super().__init__('cartesian_servo_node')

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        
        self.subscription = self.create_subscription(
            Twist,
            '/delta_twist_cmds',
            self.twist_callback,
            10)
        
        self.joint_sub = self.create_subscription(
            JointState,
            '/joint_states',
            self.joint_callback,
            10)
        self.eef_pub = self.create_publisher(PoseStamped, '/tcp_pose_raw', 10)

        self.cmd_pub = self.create_publisher(
            Float64MultiArray,
            '/forward_velocity_controller/commands',
            10)
        self.publisher_ = self.create_publisher(Float32, '/force_status', 10)
        
        # Add marker publisher for plane visualization
        self.marker_pub = self.create_publisher(
            MarkerArray,
            '/visualization_marker_array',
            10)
        
        self.joint_delta_sub = self.create_subscription(
            Float64MultiArray,
            '/delta_joint_cmds',
            self.delta_joint_sub_callback,
            10)
        

        self.joint_positions = None
        self.twist = np.zeros(6)  # Initialize twist to zero
        self.state = 1.0
        self.show_planes = True  # Toggle for plane visibility
        
        # Initialize boundary_plane_data first
        self.WORKSPACE_LIMITS = {
            'x_min': -10.5, 'x_max': 10.5,    # Left/Right walls
            'y_min': -10.5, 'y_max': 10.5,    # Front/Back walls  
            'z_min': -10.0, 'z_max': 10.0     # Floor/Ceiling
        }

        # self.WORKSPACE_LIMITS = {
        #     'x_min': -0.80, 
        #     'x_max': 0.8,    # Left/Right walls
        #     'y_min': -0.80, 'y_max': 0.8,    # Front/Back walls  
        #     'z_min': -0.00, 'z_max': 0.8     # Floor/Ceiling
        # }

        self.boundary_plane_data = []

        # URDF and KDL setup
        package_share_dir = get_package_share_directory('pymoveit2')
        urdf_path = os.path.join(package_share_dir, 'ur5_arm.urdf')
        base_link = 'base_link'                  # Change to your robot base link
        end_link = 'tool0'     
        self.robot = URDF.from_xml_file(urdf_path)             # Change to your end-effector link
        self.kdl_chain = self.load_kdl_chain(urdf_path, base_link, end_link)
        self.link_geometries = self.load_collision_geometries()
        self.link_names = list(self.link_geometries.keys())
        # note: right now calculate_twist function takes time: 0.0030 seconds,
        #                          so here we set the timer to 0.0050 seconds 
        # so that collision check could occur at fast as possible

        self.create_timer(0.005, self.calculate_twist)
        self.joint_velocities = np.zeros(6)  # Initialize joint velocities to zero
        # Initialize boundary planes AFTER boundary_plane_data is initialized
        self.boundary_planes = self.create_boundary_planes()
        
        # Add this: Create robot collision objects once
        self.robot_collision_objects = {}
        self.create_robot_collision_objects()
        
        # self.publish_plane_markers()
        # Timer for publishing markers
        self.create_timer(1.0, self.publish_plane_markers)

    def delta_joint_sub_callback(self, msg):
        if msg.data is None:
            return
        msg2 = Float64MultiArray()
        msg2.data = list(msg.data)
        self.cmd_pub.publish(msg2)

    def load_collision_geometries(self):
        geoms = {}
        for link in self.robot.links:
            if link.collision and link.collision.geometry:
                geoms[link.name] = link.collision.geometry
        return geoms

    def load_kdl_chain(self, urdf_path, base_link, end_link):
        
        success, kdl_tree = treeFromUrdfModel(self.robot)
        if not success:
            raise RuntimeError("Failed to parse URDF into KDL tree.")
        chain = kdl_tree.getChain(base_link, end_link)
        return chain


    def compute_jacobian(self, chain, joint_positions):
        nj = chain.getNrOfJoints()
        joint_array = PyKDL.JntArray(nj)
        for i in range(nj):
            joint_array[i] = joint_positions[i]

        jacobian = PyKDL.Jacobian(nj)
        solver = PyKDL.ChainJntToJacSolver(chain)
        solver.JntToJac(joint_array, jacobian)
        return jacobian


    def damped_pseudo_inverse(self, J, damping=0.01):
        JT = J.T
        identity = np.eye(J.shape[0])
        return JT @ np.linalg.inv(J @ JT + damping**2 * identity)


    def compute_joint_velocities(self, jacobian, twist_cmd):
        # Convert KDL Jacobian to numpy array
        J = np.array([[jacobian[i, j] for j in range(jacobian.columns())] for i in range(6)])
        singularity_state = self.check_singularity(J)
        # print("Singularity State:", singularity_state)
        # if singularity_state == 3.0:
        #     return None  # signal to halt
        # elif singularity_state == 2.0:
        #     twist_cmd = twist_cmd * (1*10**-3)
        # # elif isinstance(singularity_state, float) and singularity_state < 1.0:
        # #     twist_cmd = twist_cmd * singularity_state  # scale twist
        # else:
        twist_cmd = twist_cmd
        J_pinv = self.damped_pseudo_inverse(J)
        return J_pinv @ twist_cmd



    def check_singularity(self, J, hard_stop_threshold=75.0, lower_threshold=50.0):
    # def check_singularity(self, J, hard_stop_threshold=200.0, lower_threshold=180.0):
        """
        Checks the condition of the Jacobian matrix and returns a scale factor or a halt flag.
        """
        _, singular_vals, _ = np.linalg.svd(J)
        cond_number = np.max(singular_vals) / np.min(singular_vals)
        # print(f"Condition number: {cond_number:.6f}")
        if cond_number > hard_stop_threshold:
            self.state = 3.0  # signal to halt
            return 3.0
        elif cond_number > lower_threshold:
            # scale down
            self.state = 2.0
            return 2.0  # scale factor (0 < scale < 1)
        else:
            # normal operation
            self.state = 1.0
            return 1.0  # normal operation
    def joint_callback(self, msg):
        self.joint_positions = msg.position
        # print("Joint Positions:", self.joint_positions)
        # print("Joint Names:", msg)

    def twist_callback(self, msg):
        self.twist = np.array([msg.linear.x, msg.linear.y, msg.linear.z,
                          msg.angular.x, msg.angular.y, msg.angular.z])
        
    def halting(self):
        self.joint_velocities = np.zeros(len(self.joint_positions))
        vel_msg = Float64MultiArray()
        vel_msg.data = self.joint_velocities.tolist()
        self.cmd_pub.publish(vel_msg)
        publish_force_status = Float32()
        publish_force_status.data = self.state
        self.publisher_.publish(publish_force_status)


    def calculate_twist(self):
        # This function can be used to calculate the twist from joint positions if needed
        # For now, we are using a fixed twist for demonstration
        if self.joint_positions is None:
            self.get_logger().info('Waiting for joint states...')
            return

        # start_time = time.time()
        if self.testing():
            pass
            # self.get_logger().warn("⚠️ Predicted collision detected. Halting motion.")
        #     self.halting()
        #     print(self.joint_velocities)
        #     return
        
        jacobian = self.compute_jacobian(self.kdl_chain, self.joint_positions)
        self.joint_velocities = self.compute_joint_velocities(jacobian, self.twist)
        publish_force_status = Float32()
        publish_force_status.data = self.state
        self.publisher_.publish(publish_force_status)
        
        # if self.joint_velocities is None:
        #     self.get_logger().warn("⚠️ Near singularity detected. Halting motion.")
        #     self.halting()
        #     return
        
        vel_msg = Float64MultiArray()
        vel_msg.data = self.joint_velocities.tolist()
        self.cmd_pub.publish(vel_msg)
        # end_time = time.time()
        # elapsed_time = end_time - start_time
        # print("Elapsed time", elapsed_time)

    
    def resolve_package_uri(self, uri):
        if uri.startswith("package://"):
            package_name, rel_path = uri[len("package://"):].split("/", 1)
            package_path = get_package_share_directory(package_name)
            return os.path.join(package_path, rel_path)
        return uri

    def get_link_pose(self, target_frame, source_frame='base_link' ,tf=False):
            try:
                # now = self.get_clock().now()
                trans: TransformStamped = self.tf_buffer.lookup_transform(
                    source_frame, target_frame, rclpy.time.Time())
                translation = [trans.transform.translation.x,
                            trans.transform.translation.y,
                            trans.transform.translation.z]
                rotation = [trans.transform.rotation.x,
                            trans.transform.rotation.y,
                            trans.transform.rotation.z,
                            trans.transform.rotation.w]
                # Convert quaternion to rotation matrix using scipy
                if tf:
                    return translation, rotation
                r = Rotation.from_quat(rotation)
                matrix = np.eye(4)
                matrix[:3, :3] = r.as_matrix()
                matrix[0:3, 3] = translation
                return matrix
            except Exception as e:
                self.get_logger().warn(f'Failed to get transform for {target_frame}: {str(e)}')
                return None

    def create_fcl_object(self, geometry, pose_matrix):
        if isinstance(geometry, urdf_parser_py.urdf.Mesh):
            mesh_path = self.resolve_package_uri(geometry.filename)
            print(f"Loading mesh from: {mesh_path}")
            mesh = trimesh.load_mesh(mesh_path)
            vertices = mesh.vertices
            triangles = mesh.faces
            model = fcl.BVHModel()
            model.beginModel(len(vertices), len(triangles))
            model.addSubModel(vertices, triangles)
            model.endModel()
            transform = fcl.Transform(pose_matrix[0:3, 0:3], pose_matrix[0:3, 3])
            return fcl.CollisionObject(model, transform)
        return None
            
    def create_robot_collision_objects(self):
        """Create FCL collision objects for robot links once during initialization"""
       
        for link_name in self.link_geometries:
            geometry = self.link_geometries[link_name]
            # Create the collision object with identity transform initially
            identity_pose = np.eye(4)
            obj = self.create_fcl_object(geometry, identity_pose)
            if obj:
                self.robot_collision_objects[link_name] = obj
                self.get_logger().info(f"Created collision object for link: {link_name}")
    
        self.get_logger().info(f"Created {len(self.robot_collision_objects)} robot collision objects")

    def update_collision_object_pose(self, collision_object, pose_matrix):
        """Update the transform of an existing FCL collision object"""
        new_transform = fcl.Transform(pose_matrix[0:3, 0:3], pose_matrix[0:3, 3])
        collision_object.setTransform(new_transform)

    def testing(self):
        if self.joint_positions is None:
            self.get_logger().info('Waiting for joint states...')
            return

        delta_twist_joint_velocity = self.joint_velocities
        current_positions = list(self.joint_positions)
        predicted_positions = [i for i in current_positions]
        delta_t = 0.1#0.4
        
        for i, val in enumerate(delta_twist_joint_velocity):
            predicted_positions[i] = current_positions[i] + delta_t * delta_twist_joint_velocity[i]

        # 1. Get current link poses from TF
        current_poses = {}
        for link in self.link_names:
            tf_pose = self.get_link_pose(link)
            if tf_pose is not None:
                current_poses[link] = tf_pose

        # 2. Compute FK for predicted joint positions
        fk_solver = PyKDL.ChainFkSolverPos_recursive(self.kdl_chain)
        joint_array = PyKDL.JntArray(self.kdl_chain.getNrOfJoints())
        for i in range(min(len(predicted_positions), joint_array.rows())):
            joint_array[i] = predicted_positions[i]

        predicted_global_poses = {}
        for i in range(self.kdl_chain.getNrOfSegments()):
            frame = PyKDL.Frame()
            fk_solver.JntToCart(joint_array, frame, i + 1)
            pose = np.eye(4)
            for r in range(3):
                for c in range(3):
                    pose[r, c] = frame.M[r, c]
                pose[r, 3] = frame.p[r]
            link_name = self.kdl_chain.getSegment(i).getName()
            predicted_global_poses[link_name] = pose
            
            # Add debugging to see link positions
            # if 'wrist' in link_name:
            #     self.get_logger().info(f"Link {link_name} position: {pose[0:3, 3]}")
        # print("predicted_global_poses:", predicted_global_poses)
        tcp_matrix = predicted_global_poses['tool0']
        eef_pose_msg = PoseStamped()
        eef_pose_msg.header.stamp = self.get_clock().now().to_msg()
        eef_pose_msg.header.frame_id = "base_link"
        eef_pose_msg.pose.position.x = tcp_matrix[0, 3]
        eef_pose_msg.pose.position.y = tcp_matrix[1, 3]
        eef_pose_msg.pose.position.z = tcp_matrix[2, 3]
        r = Rotation.from_matrix(tcp_matrix[:3, :3])
        q = r.as_quat()
        eef_pose_msg.pose.orientation.x = q[0]
        eef_pose_msg.pose.orientation.y = q[1]
        eef_pose_msg.pose.orientation.z = q[2]
        eef_pose_msg.pose.orientation.w = q[3]
        self.eef_pub.publish(eef_pose_msg)
        # print("Published predicted TCP pose:", eef_pose_msg.pose)
        # 3. Combine base→current_link and current_link→predicted_link to get predicted global pose
        # predicted_global_poses = {}
        # for link in self.link_names:
            # print("\nProcessing link:", link)
            # print("Current pose:\n", current_poses.get(link, None))
            # print("Predicted relative pose:\n", predicted_relative_poses.get(link, None))
            # if link in current_poses and link in predicted_relative_poses:
                # predicted_global_poses[link] = np.dot(current_poses[link], predicted_relative_poses[link])
            # elif link in current_poses:
            # print("Predicted global pose:\n", predicted_global_poses[link])


        # 4. Update existing collision objects with predicted poses (MODIFIED)
        valid_collision_objects = []
        valid_links = []
        
        for link_name in self.robot_collision_objects:
            # if (link_name in predicted_global_poses and 
            #     link_name in self.robot_collision_objects):
                
                # Update the existing collision object's pose
            pose = predicted_global_poses[link_name]
            self.update_collision_object_pose(self.robot_collision_objects[link_name], pose)
            
            valid_collision_objects.append(self.robot_collision_objects[link_name])
            valid_links.append(link_name)

        # 5. Collision check (using updated objects instead of newly created ones)
        collision_detected = False
        
        # Check robot self-collision
        for i in range(len(valid_collision_objects)):
            for j in range(i + 1, len(valid_collision_objects)):
                l1 = valid_links[i]
                l2 = valid_links[j]
                if (l1, l2) in ADJACENT_PAIRS or (l2, l1) in ADJACENT_PAIRS:
                    continue
                req = fcl.CollisionRequest()
                res = fcl.CollisionResult()
                fcl.collide(valid_collision_objects[i], valid_collision_objects[j], req, res)
                if res.is_collision:
                    self.get_logger().warn(f"Predicted self-collision between {l1} and {l2} in next step. Stopping robot.")
                    collision_detected = True
                    break
            if collision_detected:
                break
        
        # Check collision with boundary planes
        if not collision_detected:
            for i, robot_link in enumerate(valid_collision_objects):
                link_name = valid_links[i]
                
                if 'base' in link_name:
                    continue

                # if 'upper_arm_link' in link_name:
                #     continue
                    
                for j, plane in enumerate(self.boundary_planes):
                    req = fcl.CollisionRequest()
                    res = fcl.CollisionResult()
                    fcl.collide(robot_link, plane, req, res)
                    if res.is_collision:
                        plane_name = "Plane_"+str(j)
                        self.get_logger().warn(f"Predicted collision between {link_name} and {plane_name} in next step. Stopping robot.")
                        collision_detected = True
                        break
                if collision_detected:
                    break
        
        return collision_detected

    def create_boundary_planes(self):
        """Create 6 simple boundary planes around UR5"""
        planes = []
        # BOUNDARY PLANE CONFIGURATION - Modify these values as needed
        # The following values are the offsets
        # self.WORKSPACE_LIMITS = {
        #     'x_min': -0.5, 'x_max': 0.5,    # Left/Right walls
        #     'y_min': -0.5, 'y_max': 0.5,    # Front/Back walls  
        #     'z_min': -0.0, 'z_max': 1.0     # Floor/Ceiling
        # }


        
        # Use class parameters for plane definitions
        plane_configs = [
            ([-1, 0, 0], self.WORKSPACE_LIMITS['x_min']),    #Plane 0  # Left wall 
            ([1, 0, 0], self.WORKSPACE_LIMITS['x_max']),   #Plane 1  # Right wall  
            ([0, -1, 0], self.WORKSPACE_LIMITS['y_min']),    #Plane 2  # Front wall 
            ([0, 1, 0], self.WORKSPACE_LIMITS['y_max']),   #Plane 3  # Back wall 
            ([0, 0, -1], self.WORKSPACE_LIMITS['z_min']),    #Plane 4  # Floor 
            ([0, 0, 1], self.WORKSPACE_LIMITS['z_max'])    #Plane 5  # Ceiling 
        ]
        
        for i, (normal, offset) in enumerate(plane_configs):
            normal = np.array(normal, dtype=float)
            plane = fcl.Plane(normal, abs(offset))
            transform = fcl.Transform()
            plane_obj = fcl.CollisionObject(plane, transform)
            planes.append(plane_obj)
        
        self.get_logger().info(f"Created {len(planes)} boundary planes")
        return planes

    def publish_plane_markers(self):
        """Publish visualization markers for boundary planes"""
        marker_array = MarkerArray()
        
        if not self.show_planes:
            return
        
        # Use class parameters for marker definitions
        planes = [
            # [position, scale, color, name]
            ([self.WORKSPACE_LIMITS['x_min'], 0, PLANE_SIZE/2.0], [PLANE_THICKNESS, PLANE_SIZE, PLANE_SIZE], [1, 0, 0, PLANE_ALPHA], "Left Wall"),
            ([self.WORKSPACE_LIMITS['x_max'], 0, PLANE_SIZE/2.0], [PLANE_THICKNESS, PLANE_SIZE, PLANE_SIZE], [1, 0, 0, PLANE_ALPHA], "Right Wall"),
            ([0, self.WORKSPACE_LIMITS['y_min'], PLANE_SIZE/2.0], [PLANE_SIZE, PLANE_THICKNESS, PLANE_SIZE], [0, 1, 0, PLANE_ALPHA], "Front Wall"),
            ([0, self.WORKSPACE_LIMITS['y_max'], PLANE_SIZE/2.0], [PLANE_SIZE, PLANE_THICKNESS, PLANE_SIZE], [0, 1, 0, PLANE_ALPHA], "Back Wall"),
            ([0, 0, self.WORKSPACE_LIMITS['z_min']],              [PLANE_SIZE, PLANE_SIZE, PLANE_THICKNESS], [0, 0, 1, PLANE_ALPHA], "Floor"),
            ([0, 0, self.WORKSPACE_LIMITS['z_max']],              [PLANE_SIZE, PLANE_SIZE, PLANE_THICKNESS], [0, 0, 1, PLANE_ALPHA], "Ceiling")
        ]
        
        for i, (pos, scale, color, name) in enumerate(planes):
            marker = Marker()
            marker.header.frame_id = "base_link"
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.ns = "boundary_planes"
            marker.id = i
            marker.type = Marker.CUBE
            marker.action = Marker.ADD
            
            # Position
            marker.pose.position.x = float(pos[0])
            marker.pose.position.y = float(pos[1])
            marker.pose.position.z = float(pos[2])
            
            # No rotation needed - keep default orientation
            marker.pose.orientation.w = 1.0
            marker.pose.orientation.x = 0.0
            marker.pose.orientation.y = 0.0
            marker.pose.orientation.z = 0.0
            
            # Scale
            marker.scale.x = float(scale[0])
            marker.scale.y = float(scale[1])
            marker.scale.z = float(scale[2])
            
            # Color
            marker.color.r = float(color[0])
            marker.color.g = float(color[1])
            marker.color.b = float(color[2])
            marker.color.a = float(color[3])
            
            marker_array.markers.append(marker)
        
        self.marker_pub.publish(marker_array)


def main(args=None):
    rclpy.init(args=args)
    node = CartesianServoNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
