#!/usr/bin/env python3

import os
import csv
import math
import time
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.executors import ExternalShutdownException
from std_msgs.msg import Float64
from nav_msgs.msg import Odometry

class DynamicsExcitationNode(Node):

    def __init__(self):
        super().__init__('dynamics_excitation_node', parameter_overrides=[Parameter('use_sim_time', Parameter.Type.BOOL, True)])
        
        # Declare ROS 2 parameters
        self.declare_parameter('csv_path', '')
        self.declare_parameter('rate', 30.0)
        self.declare_parameter('duration', -1.0)  # -1 means automatically use full CSV length
        
        self.rate = float(self.get_parameter('rate').value)
        self.dt = 1.0 / self.rate
        
        # Resolve CSV path
        user_csv_path = self.get_parameter('csv_path').value
        self.csv_path = self.resolve_csv_path(user_csv_path)
        
        self.get_logger().info(f'Loading CSV experiment commands from: {self.csv_path}')
        self.cmds_left, self.cmds_right = self.load_csv_commands(self.csv_path)
        self.total_steps = len(self.cmds_left)
        
        duration_param = float(self.get_parameter('duration').value)
        if duration_param > 0.0:
            max_steps_by_duration = int(duration_param * self.rate)
            self.total_steps = min(self.total_steps, max_steps_by_duration)
            self.duration = duration_param
        else:
            self.duration = self.total_steps * self.dt
            
        self.get_logger().info(f'Starting dynamic excitation node. Loaded {len(self.cmds_left)} command steps.')
        self.get_logger().info(f'Execution target: {self.total_steps} steps ({self.duration:.1f} s) at {self.rate:.1f} Hz.')

        # Publishers
        self.pub_lf = self.create_publisher(Float64, '/wamv/thrusters/left_front/cmd', 10)
        self.pub_lr = self.create_publisher(Float64, '/wamv/thrusters/left_rear/cmd', 10)
        self.pub_rf = self.create_publisher(Float64, '/wamv/thrusters/right_front/cmd', 10)
        self.pub_rr = self.create_publisher(Float64, '/wamv/thrusters/right_rear/cmd', 10)

        # Subscriber
        self.sub_odom = self.create_subscription(
            Odometry,
            '/wamv/sensors/position/ground_truth_odometry',
            self.odom_callback,
            10
        )

        # State storage
        self.current_x = 0.0
        self.current_y = 0.0
        self.current_z = 0.0
        self.current_roll = 0.0
        self.current_pitch = 0.0
        self.current_yaw = 0.0
        self.current_vx = 0.0
        self.current_vy = 0.0
        self.current_vz = 0.0
        self.current_wx = 0.0
        self.current_wy = 0.0
        self.current_wz = 0.0
        self.odom_received = False
        self.last_odom_log = 0.0

        # Data collection for model identification
        self.data_t = []
        self.data_x = []
        self.data_y = []
        self.data_z = []
        self.data_roll = []
        self.data_pitch = []
        self.data_yaw = []
        # Raw odometry twist is retained for diagnostics only. The fields
        # vx, vy, vz, wx, wy and wz saved at the end are reconstructed from
        # pose samples so angular discontinuities can be unwrapped first.
        self.data_vx_odom = []
        self.data_vy_odom = []
        self.data_vz_odom = []
        self.data_wx_odom = []
        self.data_wy_odom = []
        self.data_wz_odom = []
        self.data_u_left = []
        self.data_u_right = []

        self.step_idx = 0
        self.start_time = None
        self.is_finished = False

        # 30Hz Timer loop
        self.timer = self.create_timer(self.dt, self.timer_callback)

    def resolve_csv_path(self, user_path):
        if user_path and os.path.exists(user_path):
            return user_path
        
        script_dir = os.path.dirname(os.path.realpath(__file__))
        candidates = [
            os.path.join(script_dir, '..', 'data', 'DynamicModel', 'experiment_cmd_30Hz_simple.csv'),
            os.path.join(script_dir, 'data', 'DynamicModel', 'experiment_cmd_30Hz_simple.csv'),
            os.path.abspath(os.path.join(script_dir, '..', '..', '..', 'src', 'dynamic_sim_iacquabot_control', 'data', 'DynamicModel', 'experiment_cmd_30Hz_simple.csv'))
        ]
        for cand in candidates:
            cand_norm = os.path.normpath(cand)
            if os.path.exists(cand_norm):
                return cand_norm
        
        return os.path.join(script_dir, '..', 'data', 'DynamicModel', 'experiment_cmd_30Hz_simple.csv')

    def load_csv_commands(self, filepath):
        cmds_left = []
        cmds_right = []
        if not os.path.exists(filepath):
            self.get_logger().error(f'CSV file not found at: {filepath}')
            raise FileNotFoundError(f'CSV file not found: {filepath}')

        with open(filepath, mode='r', newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                cmds_left.append(float(row['cmd_left']))
                cmds_right.append(float(row['cmd_right']))
        
        return cmds_left, cmds_right

    def odom_callback(self, msg):
        self.odom_received = True
        self.current_x = msg.pose.pose.position.x
        self.current_y = msg.pose.pose.position.y
        self.current_z = msg.pose.pose.position.z

        qx = msg.pose.pose.orientation.x
        qy = msg.pose.pose.orientation.y
        qz = msg.pose.pose.orientation.z
        qw = msg.pose.pose.orientation.w
        self.current_roll, self.current_pitch, self.current_yaw = self.euler_from_quaternion(qx, qy, qz, qw)

        self.current_vx = msg.twist.twist.linear.x
        self.current_vy = msg.twist.twist.linear.y
        self.current_vz = msg.twist.twist.linear.z

        self.current_wx = msg.twist.twist.angular.x
        self.current_wy = msg.twist.twist.angular.y
        self.current_wz = msg.twist.twist.angular.z

    def timer_callback(self):
        if self.is_finished:
            return

        if not self.odom_received:
            now_sec = time.time()
            if now_sec - self.last_odom_log > 3.0:
                self.get_logger().info('Waiting for odometry on /wamv/sensors/position/ground_truth_odometry...')
                self.last_odom_log = now_sec
            return

        if self.start_time is None:
            self.start_time = self.get_clock().now().nanoseconds * 1e-9
            self.get_logger().info('Odometry active! Starting CSV excitation sequence at 30Hz...')

        now_sec = self.get_clock().now().nanoseconds * 1e-9
        elapsed_time = now_sec - self.start_time

        if self.step_idx >= self.total_steps or elapsed_time >= (self.duration + 0.5):
            self.stop_and_save()
            return

        u_left = self.cmds_left[self.step_idx]
        u_right = self.cmds_right[self.step_idx]

        # Publish thruster commands
        self.publish_thruster_cmds(u_left, u_right)

        # Save current state and inputs
        self.data_t.append(elapsed_time)
        self.data_x.append(self.current_x)
        self.data_y.append(self.current_y)
        self.data_z.append(self.current_z)
        self.data_roll.append(self.current_roll)
        self.data_pitch.append(self.current_pitch)
        self.data_yaw.append(self.current_yaw)
        self.data_vx_odom.append(self.current_vx)
        self.data_vy_odom.append(self.current_vy)
        self.data_vz_odom.append(self.current_vz)
        self.data_wx_odom.append(self.current_wx)
        self.data_wy_odom.append(self.current_wy)
        self.data_wz_odom.append(self.current_wz)
        self.data_u_left.append(u_left)
        self.data_u_right.append(u_right)

        # Logging every ~5s (150 steps at 30Hz)
        if self.step_idx % 150 == 0:
            self.get_logger().info(
                f'Step: {self.step_idx}/{self.total_steps} ({elapsed_time:.1f}s/{self.duration:.1f}s) | '
                f'u_L: {u_left:.3f}, u_R: {u_right:.3f} | Yaw: {math.degrees(self.current_yaw):.1f}°'
            )

        self.step_idx += 1

    def publish_thruster_cmds(self, u_left, u_right):
        msg_l = Float64()
        msg_l.data = float(u_left)
        msg_r = Float64()
        msg_r.data = float(u_right)

        self.pub_lf.publish(msg_l)
        self.pub_lr.publish(msg_l)
        self.pub_rf.publish(msg_r)
        self.pub_rr.publish(msg_r)

    def reconstruct_velocities_from_pose(self):
        """Differentiate pose after unwrapping angles, then express u,v in body."""
        t = np.asarray(self.data_t, dtype=float)
        if t.size < 3:
            raise ValueError('At least three pose samples are required to reconstruct velocities.')
        if np.any(np.diff(t) <= 0.0):
            raise ValueError('Simulation timestamps must be strictly increasing.')

        x = np.asarray(self.data_x, dtype=float)
        y = np.asarray(self.data_y, dtype=float)
        z = np.asarray(self.data_z, dtype=float)
        roll_unwrapped = np.unwrap(np.asarray(self.data_roll, dtype=float))
        pitch_unwrapped = np.unwrap(np.asarray(self.data_pitch, dtype=float))
        yaw_unwrapped = np.unwrap(np.asarray(self.data_yaw, dtype=float))

        x_dot = np.gradient(x, t, edge_order=2)
        y_dot = np.gradient(y, t, edge_order=2)
        z_dot = np.gradient(z, t, edge_order=2)
        roll_dot = np.gradient(roll_unwrapped, t, edge_order=2)
        pitch_dot = np.gradient(pitch_unwrapped, t, edge_order=2)
        yaw_dot = np.gradient(yaw_unwrapped, t, edge_order=2)

        # x,y are world-frame positions. Rotate their derivative into the
        # body frame so vx=u and vy=v match the Fossen convention.
        cos_yaw = np.cos(yaw_unwrapped)
        sin_yaw = np.sin(yaw_unwrapped)
        u = cos_yaw * x_dot + sin_yaw * y_dot
        v = -sin_yaw * x_dot + cos_yaw * y_dot

        return {
            'vx': u.tolist(), 'vy': v.tolist(), 'vz': z_dot.tolist(),
            'wx': roll_dot.tolist(), 'wy': pitch_dot.tolist(), 'wz': yaw_dot.tolist(),
            'roll_unwrapped': roll_unwrapped.tolist(),
            'pitch_unwrapped': pitch_unwrapped.tolist(),
            'yaw_unwrapped': yaw_unwrapped.tolist(),
            'x_dot_world': x_dot.tolist(), 'y_dot_world': y_dot.tolist()
        }

    def stop_and_save(self):
        if self.is_finished:
            return
        self.is_finished = True

        self.get_logger().info('Experiment completed or stopped. Stopping thrusters...')
        self.publish_thruster_cmds(0.0, 0.0)

        try:
            reconstructed = self.reconstruct_velocities_from_pose()
            self.get_logger().info('Velocities reconstructed from pose with unwrapped Euler angles.')
        except ValueError as exc:
            self.get_logger().error(f'Pose reconstruction failed: {exc}. Saving raw odometry twist instead.')
            reconstructed = {
                'vx': self.data_vx_odom, 'vy': self.data_vy_odom, 'vz': self.data_vz_odom,
                'wx': self.data_wx_odom, 'wy': self.data_wy_odom, 'wz': self.data_wz_odom,
                'roll_unwrapped': self.data_roll, 'pitch_unwrapped': self.data_pitch,
                'yaw_unwrapped': self.data_yaw, 'x_dot_world': self.data_vx_odom,
                'y_dot_world': self.data_vy_odom
            }

        data_dict = {
            't': self.data_t,
            'x': self.data_x,
            'y': self.data_y,
            'z': self.data_z,
            'roll': self.data_roll,
            'pitch': self.data_pitch,
            'yaw': self.data_yaw,
            'vx': reconstructed['vx'], 'vy': reconstructed['vy'], 'vz': reconstructed['vz'],
            'wx': reconstructed['wx'], 'wy': reconstructed['wy'], 'wz': reconstructed['wz'],
            'roll_unwrapped': reconstructed['roll_unwrapped'],
            'pitch_unwrapped': reconstructed['pitch_unwrapped'],
            'yaw_unwrapped': reconstructed['yaw_unwrapped'],
            'x_dot_world': reconstructed['x_dot_world'],
            'y_dot_world': reconstructed['y_dot_world'],
            'vx_odom_raw': self.data_vx_odom, 'vy_odom_raw': self.data_vy_odom,
            'vz_odom_raw': self.data_vz_odom, 'wx_odom_raw': self.data_wx_odom,
            'wy_odom_raw': self.data_wy_odom, 'wz_odom_raw': self.data_wz_odom,
            'u_left': self.data_u_left,
            'u_right': self.data_u_right
        }

        script_dir = os.path.dirname(os.path.realpath(__file__))
        package_dir = os.path.abspath(os.path.join(script_dir, '..'))
        data_dir = os.path.join(package_dir, 'data')

        try:
            os.makedirs(data_dir, exist_ok=True)
        except Exception as e:
            data_dir = os.getcwd()
            self.get_logger().error(f'Could not create/access data directory. Using CWD. Error: {e}')

        timestamp = time.strftime('%Y%m%d_%H%M%S')
        filename_mat = os.path.join(data_dir, f'wamvsim_{timestamp}.mat')
        filename_csv = os.path.join(data_dir, f'wamvsim_{timestamp}.csv')

        try:
            import scipy.io
            scipy.io.savemat(filename_mat, data_dict)
            self.get_logger().info(f'Odometry identification data saved to MATLAB (.mat): {filename_mat}')
        except ImportError:
            self.get_logger().warn('scipy not installed. Skipping .mat save.')

        try:
            with open(filename_csv, mode='w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(data_dict.keys())
                for i in range(len(self.data_t)):
                    row = [data_dict[key][i] for key in data_dict.keys()]
                    writer.writerow(row)
            self.get_logger().info(f'Odometry identification data saved to CSV: {filename_csv}')
        except Exception as e:
            self.get_logger().error(f'Error saving backup CSV: {e}')

        rclpy.shutdown()

    def euler_from_quaternion(self, x, y, z, w):
        t0 = +2.0 * (w * x + y * z)
        t1 = +1.0 - 2.0 * (x * x + y * y)
        roll = math.atan2(t0, t1)

        t2 = +2.0 * (w * y - z * x)
        t2 = +1.0 if t2 > +1.0 else t2
        t2 = -1.0 if t2 < -1.0 else t2
        pitch = math.asin(t2)

        t3 = +2.0 * (w * z + x * y)
        t4 = +1.0 - 2.0 * (y * y + z * z)
        yaw = math.atan2(t3, t4)

        return roll, pitch, yaw


def main(args=None):
    rclpy.init(args=args)
    node = DynamicsExcitationNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        node.get_logger().info('Sequence interrupted by user.')
    finally:
        if rclpy.ok():
            node.stop_and_save()


if __name__ == '__main__':
    main()

