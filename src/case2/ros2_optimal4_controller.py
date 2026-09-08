#!/usr/bin/env python3

import sys
import os
import math
import numpy as np
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry, Path
from geometry_msgs.msg import PoseStamped, Point, Quaternion, TransformStamped
from std_msgs.msg import Float64, Empty, Bool
from std_srvs.srv import Trigger
from visualization_msgs.msg import Marker, MarkerArray
from rclpy.parameter import Parameter
import tf2_ros
from usv_params import m11, m22, m33, X_u, X_uu, Y_v, Y_vv, N_r, N_rr, T_MAX, T_MIN, DT_EXPERIMENT
from main import solve_with_saturation_check
from simulate_mpc import USV_MPC, inverse_thrust, forward_thrust

def quaternion_from_yaw(yaw):
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)
    q = Quaternion()
    q.w = cy
    q.x = 0.0
    q.y = 0.0
    q.z = sy
    return q

def yaw_from_quaternion(q):
    t3 = +2.0 * (q.w * q.z + q.x * q.y)
    t4 = +1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(t3, t4)

class Optimal4CascadeControllerROS2(Node):

    def __init__(self):
        super().__init__('optimal4_cascade_controller', parameter_overrides=[Parameter('use_sim_time', Parameter.Type.BOOL, True)])
        self.declare_parameter('dt_mpc', 0.2)
        self.declare_parameter('N_mpc', 35)
        self.declare_parameter('auto_start', True)
        self.declare_parameter('frame_id', 'map')
        self.declare_parameter('init_tf_frame', 'usv_init_origin')
        self.dt_mpc = float(self.get_parameter('dt_mpc').value)
        self.N_mpc = int(self.get_parameter('N_mpc').value)
        self.auto_start = bool(self.get_parameter('auto_start').value)
        self.frame_id = str(self.get_parameter('frame_id').value)
        self.init_tf_frame = str(self.get_parameter('init_tf_frame').value)
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)
        self.waypoints_local = np.array([[0.0, 0.0], [8.0, 0.0], [14.0, 5.0], [14.0, 13.0], [20.0, 17.0], [28.0, 17.0], [32.0, 10.0], [26.0, 4.0]])
        self.times_initial = np.array([0.0, 7.0, 14.0, 20.0, 27.0, 33.0, 40.0, 48.0])
        self.curr_x = 0.0
        self.curr_y = 0.0
        self.curr_z = 0.0
        self.curr_psi = 0.0
        self.curr_u = 0.0
        self.curr_v = 0.0
        self.curr_r = 0.0
        self.odom_received = False
        self.origin_x = 0.0
        self.origin_y = 0.0
        self.origin_z = 0.0
        self.origin_psi = 0.0
        self.origin_fixed = False
        self.is_active = False
        self.start_time_ros = None
        self.t_sim = None
        self.Z_d_global = None
        self.waypoints_global = None
        self.get_logger().info('Inicializador CasADi MPC Opti Solver...')
        self.mpc = USV_MPC(dt_mpc=self.dt_mpc, N_mpc=self.N_mpc)
        self.step_mpc = int(self.dt_mpc / DT_EXPERIMENT)
        self.sub_odom_gt = self.create_subscription(Odometry, '/wamv/sensors/position/ground_truth_odometry', self.odom_cb, 10)
        self.sub_odom = self.create_subscription(Odometry, 'current_odom', self.odom_cb, 10)
        self.sub_odom_alt = self.create_subscription(Odometry, '/odom', self.odom_cb, 10)
        self.sub_play = self.create_subscription(Empty, '/start_trajectory', self.trigger_play_cb, 10)
        self.sub_play_bool = self.create_subscription(Bool, '/play', self.trigger_play_bool_cb, 10)
        self.srv_trigger = self.create_service(Trigger, '/trigger_start', self.trigger_service_cb)
        self.pub_markers = self.create_publisher(MarkerArray, '/waypoint_markers', 10)
        self.pub_planned_path = self.create_publisher(Path, '/planned_path', 10)
        self.pub_mpc_horizon = self.create_publisher(Path, '/mpc_horizon', 10)
        self.pub_lf = self.create_publisher(Float64, '/wamv/thrusters/left_front/cmd', 10)
        self.pub_lr = self.create_publisher(Float64, '/wamv/thrusters/left_rear/cmd', 10)
        self.pub_rf = self.create_publisher(Float64, '/wamv/thrusters/right_front/cmd', 10)
        self.pub_rr = self.create_publisher(Float64, '/wamv/thrusters/right_rear/cmd', 10)
        self.pub_FR = self.create_publisher(Float64, 'thrusters/FR/cmd', 10)
        self.pub_FL = self.create_publisher(Float64, 'thrusters/FL/cmd', 10)
        self.pub_BR = self.create_publisher(Float64, 'thrusters/BR/cmd', 10)
        self.pub_BL = self.create_publisher(Float64, 'thrusters/BL/cmd', 10)
        self.pub_vrx_FR = self.create_publisher(Float64, '/wamv/thrusters/front_right/thrust', 10)
        self.pub_vrx_FL = self.create_publisher(Float64, '/wamv/thrusters/front_left/thrust', 10)
        self.pub_vrx_BR = self.create_publisher(Float64, '/wamv/thrusters/back_right/thrust', 10)
        self.pub_vrx_BL = self.create_publisher(Float64, '/wamv/thrusters/back_left/thrust', 10)
        self.timer = self.create_timer(DT_EXPERIMENT, self.control_loop)
        self.get_logger().info(f'Nodo Optimal4 Cascade MPC iniciado a {1.0 / DT_EXPERIMENT:.1f} Hz. Listo para Odometría y Play.')

    def odom_cb(self, msg: Odometry):
        self.curr_x = msg.pose.pose.position.x
        self.curr_y = msg.pose.pose.position.y
        self.curr_z = msg.pose.pose.position.z
        self.curr_psi = yaw_from_quaternion(msg.pose.pose.orientation)
        self.curr_u = msg.twist.twist.linear.x
        self.curr_v = msg.twist.twist.linear.y
        self.curr_r = msg.twist.twist.angular.z
        if msg.header.frame_id:
            self.frame_id = msg.header.frame_id
        if not self.origin_fixed:
            self.origin_x = self.curr_x
            self.origin_y = self.curr_y
            self.origin_z = self.curr_z
            self.origin_psi = self.curr_psi
            self.origin_fixed = True
            self.get_logger().info(f"=== FIJANDO ORIGEN TF 'usv_init_origin' en X={self.origin_x:.3f}m, Y={self.origin_y:.3f}m, Psi={math.degrees(self.origin_psi):.2f}° ===")
            self.compute_global_waypoints()
            self.publish_rviz_waypoint_markers()
        if not self.odom_received:
            self.odom_received = True
            self.get_logger().info(f'Primera Odometría recibida en ({self.curr_x:.2f}, {self.curr_y:.2f}), Rumbo: {math.degrees(self.curr_psi):.1f} deg.')
            if self.auto_start and (not self.is_active):
                self.start_trajectory_execution()

    def broadcast_origin_tf(self):
        if not self.origin_fixed or not rclpy.ok():
            return
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = self.frame_id
        t.child_frame_id = self.init_tf_frame
        t.transform.translation.x = float(self.origin_x)
        t.transform.translation.y = float(self.origin_y)
        t.transform.translation.z = float(self.origin_z)
        t.transform.rotation = quaternion_from_yaw(self.origin_psi)
        self.tf_broadcaster.sendTransform(t)

    def compute_global_waypoints(self):
        c0, s0 = (math.cos(self.origin_psi), math.sin(self.origin_psi))
        R0 = np.array([[c0, -s0], [s0, c0]])
        self.waypoints_global = np.zeros_like(self.waypoints_local)
        for i in range(len(self.waypoints_local)):
            wpt_rot = R0 @ self.waypoints_local[i]
            self.waypoints_global[i, 0] = self.origin_x + wpt_rot[0]
            self.waypoints_global[i, 1] = self.origin_y + wpt_rot[1]

    def trigger_play_cb(self, msg: Empty):
        self.get_logger().info("Disparo '/start_trajectory' (Play) recibido.")
        self.start_trajectory_execution()

    def trigger_play_bool_cb(self, msg: Bool):
        if msg.data:
            self.get_logger().info("Disparo '/play = True' recibido.")
            self.start_trajectory_execution()

    def trigger_service_cb(self, request, response):
        self.get_logger().info("Servicio '/trigger_start' (Play) invocado.")
        success, msg_str = self.start_trajectory_execution()
        response.success = success
        response.message = msg_str
        return response

    def start_trajectory_execution(self):
        if not self.odom_received:
            warn_msg = 'No se puede iniciar: Esperando primera Odometría del barco.'
            self.get_logger().warn(warn_msg)
            return (False, warn_msg)
        self.compute_global_waypoints()
        self.get_logger().info('Resolviendo optimización Minimum-Jerk con comprobación de saturación...')
        traj, result_plan, summary, final_times = solve_with_saturation_check(self.waypoints_global, self.times_initial, dt=DT_EXPERIMENT, verbose=False)
        t_plan = result_plan['t']
        self.t_sim = np.arange(0, t_plan[-1], DT_EXPERIMENT)
        x_d = np.interp(self.t_sim, t_plan, result_plan['eta'][:, 0])
        y_d = np.interp(self.t_sim, t_plan, result_plan['eta'][:, 1])
        psi_d_unwrapped = np.unwrap(result_plan['eta'][:, 2])
        psi_d = np.interp(self.t_sim, t_plan, psi_d_unwrapped)
        u_d = np.interp(self.t_sim, t_plan, result_plan['nu'][:, 0])
        v_d = np.interp(self.t_sim, t_plan, result_plan['nu'][:, 1])
        r_d = np.interp(self.t_sim, t_plan, result_plan['nu'][:, 2])
        self.Z_d_global = np.vstack((x_d, y_d, psi_d, u_d, v_d, r_d))
        self.publish_rviz_waypoint_markers()
        self.publish_rviz_planned_path()
        self.start_time_ros = self.get_clock().now()
        self.is_active = True
        succ_msg = f'Trayectoria en Cascada generada con éxito ({len(self.t_sim)} puntos, Duración: {self.t_sim[-1]:.1f}s).'
        self.get_logger().info(succ_msg)
        return (True, succ_msg)

    def publish_rviz_waypoint_markers(self):
        if self.waypoints_global is None or not rclpy.ok():
            return
        marker_array = MarkerArray()
        now_ros = self.get_clock().now().to_msg()
        for i, (wx, wy) in enumerate(self.waypoints_global):
            sp_marker = Marker()
            sp_marker.header.frame_id = self.frame_id
            sp_marker.header.stamp = now_ros
            sp_marker.ns = 'waypoints_spheres'
            sp_marker.id = i
            sp_marker.type = Marker.SPHERE
            sp_marker.action = Marker.ADD
            sp_marker.pose.position.x = float(wx)
            sp_marker.pose.position.y = float(wy)
            sp_marker.pose.position.z = 0.15
            sp_marker.pose.orientation.w = 1.0
            sp_marker.scale.x = 0.3
            sp_marker.scale.y = 0.3
            sp_marker.scale.z = 0.3
            if i == 0:
                sp_marker.color.r = 0.0
                sp_marker.color.g = 1.0
                sp_marker.color.b = 0.2
                sp_marker.color.a = 0.95
            elif i == len(self.waypoints_global) - 1:
                sp_marker.color.r = 1.0
                sp_marker.color.g = 0.1
                sp_marker.color.b = 0.1
                sp_marker.color.a = 0.95
            else:
                sp_marker.color.r = 1.0
                sp_marker.color.g = 0.8
                sp_marker.color.b = 0.0
                sp_marker.color.a = 0.95
            marker_array.markers.append(sp_marker)
            text_marker = Marker()
            text_marker.header.frame_id = self.frame_id
            text_marker.header.stamp = now_ros
            text_marker.ns = 'waypoints_text'
            text_marker.id = 100 + i
            text_marker.type = Marker.TEXT_VIEW_FACING
            text_marker.action = Marker.ADD
            text_marker.pose.position.x = float(wx)
            text_marker.pose.position.y = float(wy)
            text_marker.pose.position.z = 0.6
            text_marker.scale.z = 0.4
            text_marker.color.r = 1.0
            text_marker.color.g = 1.0
            text_marker.color.b = 1.0
            text_marker.color.a = 1.0
            text_marker.text = f'WP {i}'
            marker_array.markers.append(text_marker)
        self.pub_markers.publish(marker_array)

    def publish_rviz_planned_path(self):
        if self.Z_d_global is None or not rclpy.ok():
            return
        path_msg = Path()
        path_msg.header.frame_id = self.frame_id
        path_msg.header.stamp = self.get_clock().now().to_msg()
        n_pts = self.Z_d_global.shape[1]
        for i in range(n_pts):
            ps = PoseStamped()
            ps.header.frame_id = self.frame_id
            ps.pose.position.x = float(self.Z_d_global[0, i])
            ps.pose.position.y = float(self.Z_d_global[1, i])
            ps.pose.position.z = 0.1
            ps.pose.orientation = quaternion_from_yaw(self.Z_d_global[2, i])
            path_msg.poses.append(ps)
        self.pub_planned_path.publish(path_msg)

    def control_loop(self):
        self.broadcast_origin_tf()
        self.publish_rviz_waypoint_markers()
        if not self.is_active or self.Z_d_global is None:
            return
        elapsed = (self.get_clock().now() - self.start_time_ros).nanoseconds * 1e-09
        curr_idx = int(round(elapsed / DT_EXPERIMENT))
        n_steps = self.Z_d_global.shape[1]
        if curr_idx >= n_steps:
            curr_idx = n_steps - 1
            self.get_logger().info('Trayectoria finalizada. Manteniendo punto de reposo final.', throttle_duration_sec=5.0)
        X_ref_vals = np.zeros((6, self.N_mpc + 1))
        for k in range(self.N_mpc + 1):
            idx = curr_idx + k * self.step_mpc
            if idx >= n_steps:
                idx = n_steps - 1
            X_ref_vals[:, k] = self.Z_d_global[:, idx]
            if k == 0:
                diff = (X_ref_vals[2, 0] - self.curr_psi + np.pi) % (2 * np.pi) - np.pi
                X_ref_vals[2, 0] = self.curr_psi + diff
            else:
                diff = (X_ref_vals[2, k] - X_ref_vals[2, k - 1] + np.pi) % (2 * np.pi) - np.pi
                X_ref_vals[2, k] = X_ref_vals[2, k - 1] + diff
        x0_curr = np.array([self.curr_x, self.curr_y, self.curr_psi, self.curr_u, self.curr_v, self.curr_r])
        u_opt, x_opt = self.mpc.solve(x0_curr, X_ref_vals)
        Tu_dem, Tr_dem = (u_opt[0], u_opt[1])
        if x_opt is not None and rclpy.ok():
            mpc_path = Path()
            mpc_path.header.frame_id = self.frame_id
            mpc_path.header.stamp = self.get_clock().now().to_msg()
            for k in range(x_opt.shape[1]):
                ps = PoseStamped()
                ps.header.frame_id = self.frame_id
                ps.pose.position.x = float(x_opt[0, k])
                ps.pose.position.y = float(x_opt[1, k])
                ps.pose.position.z = 0.2
                ps.pose.orientation = quaternion_from_yaw(x_opt[2, k])
                mpc_path.poses.append(ps)
            self.pub_mpc_horizon.publish(mpc_path)
        y_abs = 1.027135
        T_right = 0.5 * (Tu_dem + Tr_dem / y_abs)
        T_left = 0.5 * (Tu_dem - Tr_dem / y_abs)
        T_FR_dem, T_BR_dem = (T_right / 2.0, T_right / 2.0)
        T_FL_dem, T_BL_dem = (T_left / 2.0, T_left / 2.0)
        cmd_FR = inverse_thrust(T_FR_dem)
        cmd_FL = inverse_thrust(T_FL_dem)
        cmd_BR = inverse_thrust(T_BR_dem)
        cmd_BL = inverse_thrust(T_BL_dem)
        if not rclpy.ok():
            return
        self.pub_lf.publish(Float64(data=cmd_FL))
        self.pub_lr.publish(Float64(data=cmd_BL))
        self.pub_rf.publish(Float64(data=cmd_FR))
        self.pub_rr.publish(Float64(data=cmd_BR))
        self.pub_FR.publish(Float64(data=cmd_FR))
        self.pub_FL.publish(Float64(data=cmd_FL))
        self.pub_BR.publish(Float64(data=cmd_BR))
        self.pub_BL.publish(Float64(data=cmd_BL))
        self.pub_vrx_FR.publish(Float64(data=cmd_FR * T_MAX if cmd_FR > 0 else cmd_FR * abs(T_MIN)))
        self.pub_vrx_FL.publish(Float64(data=cmd_FL * T_MAX if cmd_FL > 0 else cmd_FL * abs(T_MIN)))
        self.pub_vrx_BR.publish(Float64(data=cmd_BR * T_MAX if cmd_BR > 0 else cmd_BR * abs(T_MIN)))
        self.pub_vrx_BL.publish(Float64(data=cmd_BL * T_MAX if cmd_BL > 0 else cmd_BL * abs(T_MIN)))

def main(args=None):
    rclpy.init(args=args)
    node = Optimal4CascadeControllerROS2()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
if __name__ == '__main__':
    main()
