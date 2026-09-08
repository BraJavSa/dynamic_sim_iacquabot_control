import os
import sys
import time
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import casadi as ca
from usv_params import m11, m22, m33, X_u, X_uu, Y_v, Y_vv, N_r, N_rr, T_MAX, T_MIN, DT_EXPERIMENT, cmd_from_thrust_poly, thrust_from_cmd_poly
from main import solve_with_saturation_check, plot_results

def get_boat_polygon(x, y, psi, scale=1.5):
    pts = np.array([[scale, 0], [-scale / 1.5, scale / 2], [-scale / 1.5, -scale / 2], [scale, 0]])
    c, s = (np.cos(psi), np.sin(psi))
    R = np.array([[c, -s], [s, c]])
    pts_rot = (R @ pts.T).T
    pts_rot[:, 0] += x
    pts_rot[:, 1] += y
    return pts_rot
y_FR, y_FL = (-1.027135, 1.027135)
y_BR, y_BL = (-1.027135, 1.027135)

def inverse_thrust(T):
    T = float(np.clip(T, T_MIN, T_MAX))
    return float(cmd_from_thrust_poly(T))

def forward_thrust(cmd):
    cmd = float(np.clip(cmd, -1.0, 1.0))
    return float(thrust_from_cmd_poly(cmd))

class USV_MPC:

    def __init__(self, dt_mpc=0.2, N_mpc=35):
        self.dt = dt_mpc
        self.N = N_mpc
        self.nx = 6
        self.nu = 2
        self.opti = ca.Opti()
        self.X = self.opti.variable(self.nx, self.N + 1)
        self.U = self.opti.variable(self.nu, self.N)
        self.x0 = self.opti.parameter(self.nx)
        self.X_ref = self.opti.parameter(self.nx, self.N + 1)
        self.U_ref = self.opti.parameter(self.nu, self.N)
        self.f = self.build_dynamics()
        self.opti.subject_to(self.X[:, 0] == self.x0)
        for k in range(self.N):
            k1 = self.f(self.X[:, k], self.U[:, k])
            k2 = self.f(self.X[:, k] + self.dt / 2 * k1, self.U[:, k])
            k3 = self.f(self.X[:, k] + self.dt / 2 * k2, self.U[:, k])
            k4 = self.f(self.X[:, k] + self.dt * k3, self.U[:, k])
            x_next = self.X[:, k] + self.dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
            self.opti.subject_to(self.X[:, k + 1] == x_next)
        self.opti.subject_to(self.opti.bounded(-113.0, self.U[0, :], 145.0))
        self.opti.subject_to(self.opti.bounded(-130.0, self.U[1, :], 130.0))
        Q = np.diag([1000.0, 1000.0, 0.0, 10.0, 0.0, 0.0])
        R = np.diag([0.1, 0.5])
        cost = 0
        for k in range(self.N):
            err = self.X[:, k] - self.X_ref[:, k]
            err_u = self.U[:, k] - self.U_ref[:, k]
            cost += ca.mtimes([err.T, Q, err]) + ca.mtimes([err_u.T, R, err_u])
        err_N = self.X[:, self.N] - self.X_ref[:, self.N]
        cost += ca.mtimes([err_N.T, Q * 10, err_N])
        self.opti.minimize(cost)
        p_opts = {'expand': True, 'print_time': False}
        s_opts = {'max_iter': 60, 'print_level': 0, 'sb': 'yes', 'tol': 0.0001}
        self.opti.solver('ipopt', p_opts, s_opts)

    def build_dynamics(self):
        x = ca.MX.sym('x')
        y = ca.MX.sym('y')
        psi = ca.MX.sym('psi')
        u = ca.MX.sym('u')
        v = ca.MX.sym('v')
        r = ca.MX.sym('r')
        z = ca.vertcat(x, y, psi, u, v, r)
        Tu = ca.MX.sym('Tu')
        Tr = ca.MX.sym('Tr')
        w = ca.vertcat(Tu, Tr)
        dx = u * ca.cos(psi) - v * ca.sin(psi)
        dy = u * ca.sin(psi) + v * ca.cos(psi)
        dpsi = r
        du = (Tu + m22 * v * r - X_u * u - X_uu * ca.fabs(u) * u) / m11
        dv = (-m11 * u * r - Y_v * v - Y_vv * ca.fabs(v) * v) / m22
        dr = (Tr - (m22 - m11) * u * v - N_r * r - N_rr * ca.fabs(r) * r) / m33
        dz = ca.vertcat(dx, dy, dpsi, du, dv, dr)
        return ca.Function('f', [z, w], [dz])

    def solve(self, x0, X_ref_vals, U_ref_vals=None):
        self.opti.set_value(self.x0, x0)
        self.opti.set_value(self.X_ref, X_ref_vals)
        if U_ref_vals is None:
            U_ref_vals = np.zeros((2, self.N))
        self.opti.set_value(self.U_ref, U_ref_vals)
        try:
            sol = self.opti.solve()
            u_opt = sol.value(self.U)
            x_opt = sol.value(self.X)
            self.opti.set_initial(self.X, x_opt)
            self.opti.set_initial(self.U, u_opt)
            return (u_opt[:, 0], x_opt)
        except Exception:
            u_opt = self.opti.debug.value(self.U)
            x_opt = self.opti.debug.value(self.X)
            return (u_opt[:, 0], x_opt)


def plant_derivative(u, v, r, Tu, Tr):
    du = (Tu + m22 * v * r - X_u * u - X_uu * abs(u) * u) / m11
    dv = (-m11 * u * r - Y_v * v - Y_vv * abs(v) * v) / m22
    dr = (Tr - (m22 - m11) * u * v - N_r * r - N_rr * abs(r) * r) / m33
    return (du, dv, dr)

def rk4_step(u, v, r, x, y, psi, Tu, Tr, dt):
    du1, dv1, dr1 = plant_derivative(u, v, r, Tu, Tr)
    dx1 = u * np.cos(psi) - v * np.sin(psi)
    dy1 = u * np.sin(psi) + v * np.cos(psi)
    dpsi1 = r
    du2, dv2, dr2 = plant_derivative(u + 0.5 * dt * du1, v + 0.5 * dt * dv1, r + 0.5 * dt * dr1, Tu, Tr)
    dx2 = (u + 0.5 * dt * du1) * np.cos(psi + 0.5 * dt * dpsi1) - (v + 0.5 * dt * dv1) * np.sin(psi + 0.5 * dt * dpsi1)
    dy2 = (u + 0.5 * dt * du1) * np.sin(psi + 0.5 * dt * dpsi1) + (v + 0.5 * dt * dv1) * np.cos(psi + 0.5 * dt * dpsi1)
    dpsi2 = r + 0.5 * dt * dr1
    du3, dv3, dr3 = plant_derivative(u + 0.5 * dt * du2, v + 0.5 * dt * dv2, r + 0.5 * dt * dr2, Tu, Tr)
    dx3 = (u + 0.5 * dt * du2) * np.cos(psi + 0.5 * dt * dpsi2) - (v + 0.5 * dt * dv2) * np.sin(psi + 0.5 * dt * dpsi2)
    dy3 = (u + 0.5 * dt * du2) * np.sin(psi + 0.5 * dt * dpsi2) + (v + 0.5 * dt * dv2) * np.cos(psi + 0.5 * dt * dpsi2)
    dpsi3 = r + 0.5 * dt * dr2
    du4, dv4, dr4 = plant_derivative(u + dt * du3, v + dt * dv3, r + dt * dr3, Tu, Tr)
    dx4 = (u + dt * du3) * np.cos(psi + dt * dpsi3) - (v + dt * dv3) * np.sin(psi + dt * dpsi3)
    dy4 = (u + dt * du3) * np.sin(psi + dt * dpsi3) + (v + dt * dv3) * np.cos(psi + dt * dpsi3)
    dpsi4 = r + dt * dr3
    u_new = u + dt / 6.0 * (du1 + 2 * du2 + 2 * du3 + du4)
    v_new = v + dt / 6.0 * (dv1 + 2 * dv2 + 2 * dv3 + dv4)
    r_new = r + dt / 6.0 * (dr1 + 2 * dr2 + 2 * dr3 + dr4)
    x_new = x + dt / 6.0 * (dx1 + 2 * dx2 + 2 * dx3 + dx4)
    y_new = y + dt / 6.0 * (dy1 + 2 * dy2 + 2 * dy3 + dy4)
    psi_new = psi + dt / 6.0 * (dpsi1 + 2 * dpsi2 + 2 * dpsi3 + dpsi4)
    return (u_new, v_new, r_new, x_new, y_new, psi_new)

def generate_animation(t_sim, x_d_array, y_d_array, psi_d_array, hist_x, hist_y, hist_psi, hist_x_opt, waypoints_xyz, anim_outpath=None, fps=20, t_wp7=None):
    if anim_outpath is None:
        anim_outpath = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'usv_optimal_mpc_animation.gif')
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.plot(x_d_array, y_d_array, 'k--', alpha=0.6, label='Ruta Planeada (Minimum-Jerk)')
    ax.scatter(waypoints_xyz[:, 0], waypoints_xyz[:, 1], c='red', s=60, zorder=5, label='Waypoints')
    for i, (wx, wy) in enumerate(waypoints_xyz[:, :2]):
        ax.annotate(str(i), (wx, wy), textcoords='offset points', xytext=(5, 5), fontsize=9, color='darkred')
    line_real, = ax.plot([], [], 'b-', lw=2, alpha=0.8, label='Trayectoria Real USV')
    poly_des, = ax.plot([], [], 'r-', lw=2, label='Deseado (Barco Rojo)')
    poly_real, = ax.plot([], [], 'g-', lw=2, label='Real (Barco Verde)')
    line_mpc, = ax.plot([], [], 'y-', lw=2, label='Horizonte MPC (7s)')
    ax.set_xlabel('X [m]')
    ax.set_ylabel('Y [m]')
    ax.set_title('Animación de Navegación USV: Planeado vs Real + MPC')
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True)
    ax.set_aspect('equal')
    if t_wp7 is not None:
        n_frames = int(np.searchsorted(t_sim, t_wp7))
    else:
        n_frames = len(t_sim)
    frame_step = max(1, int(n_frames / 150))
    frame_indices = np.arange(0, n_frames, frame_step)

    def init():
        line_real.set_data([], [])
        poly_des.set_data([], [])
        poly_real.set_data([], [])
        line_mpc.set_data([], [])
        return (line_real, poly_des, poly_real, line_mpc)

    def update(frame_idx):
        i = frame_indices[frame_idx]
        x_d, y_d, psi_d = (x_d_array[i], y_d_array[i], psi_d_array[i])
        x_r, y_r, psi_r = (hist_x[i], hist_y[i], hist_psi[i])
        pts_des = get_boat_polygon(x_d, y_d, psi_d)
        pts_real = get_boat_polygon(x_r, y_r, psi_r)
        poly_des.set_data(pts_des[:, 0], pts_des[:, 1])
        poly_real.set_data(pts_real[:, 0], pts_real[:, 1])
        line_real.set_data(hist_x[:i + 1], hist_y[:i + 1])
        if hist_x_opt[i] is not None:
            x_opt_i = hist_x_opt[i]
            line_mpc.set_data(x_opt_i[0, :], x_opt_i[1, :])
        ax.set_xlim(x_r - 12, x_r + 12)
        ax.set_ylim(y_r - 12, y_r + 12)
        return (line_real, poly_des, poly_real, line_mpc)
    anim = FuncAnimation(fig, update, frames=len(frame_indices), init_func=init, blit=False)
    out_dir = os.path.dirname(os.path.abspath(anim_outpath))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    print(f'Guardando archivo de animación en: {anim_outpath}...')
    anim.save(anim_outpath, writer=PillowWriter(fps=fps))
    plt.close(fig)
    print(f'¡Animación guardada con éxito en {anim_outpath}!')
    return anim_outpath

def run_simulation(waypoints_xyz=None, times_initial=None, outpath=None, anim_outpath=None, generate_anim=True, return_data=False):
    matplotlib.use('Agg')
    if waypoints_xyz is None:
        waypoints_xyz = np.array([[0.0, 0.0, 0.0], [8.0, 0.0, 0.0], [14.0, 5.0, np.pi / 2], [14.0, 13.0, np.pi / 2], [20.0, 17.0, 0.0], [28.0, 17.0, 0.0], [32.0, 10.0, -np.pi / 2], [26.0, 4.0, -3 * np.pi / 4], [20.0, 1.5, -3 * np.pi / 4]])
    if times_initial is None:
        times_initial = np.array([0.0, 7.0, 14.0, 20.0, 27.0, 33.0, 40.0, 48.0, 55.0])

    print('=== STEP 1: Generating Optimal Minimum-Jerk Trajectory ===')
    traj, result_plan, summary, final_times = solve_with_saturation_check(waypoints_xyz, times_initial, dt=DT_EXPERIMENT, verbose=True)
    dt_sim = DT_EXPERIMENT
    t_plan = result_plan['t']
    t_sim = np.arange(0, t_plan[-1], dt_sim)
    x_d_array = np.interp(t_sim, t_plan, result_plan['eta'][:, 0])
    y_d_array = np.interp(t_sim, t_plan, result_plan['eta'][:, 1])
    psi_d_unwrapped = np.unwrap(result_plan['eta'][:, 2])
    psi_d_array = np.interp(t_sim, t_plan, psi_d_unwrapped)
    u_d_array = np.interp(t_sim, t_plan, result_plan['nu'][:, 0])
    v_d_array = np.interp(t_sim, t_plan, result_plan['nu'][:, 1])
    r_d_array = np.interp(t_sim, t_plan, result_plan['nu'][:, 2])
    tau_u_plan = np.interp(t_sim, t_plan, result_plan['tau'][:, 0])
    tau_r_plan = np.interp(t_sim, t_plan, result_plan['tau'][:, 2])
    Z_d_array = np.vstack((x_d_array, y_d_array, psi_d_array, u_d_array, v_d_array, r_d_array))

    print('\n=== STEP 2: Initializing MPC and Simulating Plant Dynamics ===')
    dt_mpc = 0.2
    N_mpc = 35
    mpc = USV_MPC(dt_mpc=dt_mpc, N_mpc=N_mpc)
    step_mpc = int(dt_mpc / dt_sim)
    x, y, psi = (x_d_array[0], y_d_array[0], psi_d_array[0])
    u, v, r = (0.1, 0.0, 0.0)
    hist_x, hist_y, hist_psi = ([x], [y], [psi])
    hist_u, hist_v, hist_r = ([u], [v], [r])
    hist_Tu_dem, hist_Tr_dem = ([], [])
    hist_Tu_act, hist_Tr_act = ([], [])
    hist_T_FR, hist_T_FL, hist_T_BR, hist_T_BL = ([], [], [], [])
    hist_x_opt = []
    n_steps = len(t_sim)
    for i in range(n_steps):
        X_ref_vals = np.zeros((6, N_mpc + 1))
        U_ref_vals = np.zeros((2, N_mpc))
        for k in range(N_mpc + 1):
            idx = i + k * step_mpc
            if idx >= n_steps:
                idx = n_steps - 1
            X_ref_vals[:, k] = Z_d_array[:, idx]
            if k < N_mpc:
                U_ref_vals[0, k] = tau_u_plan[idx]
                U_ref_vals[1, k] = tau_r_plan[idx]
            if k == 0:
                diff = (X_ref_vals[2, 0] - psi + np.pi) % (2 * np.pi) - np.pi
                X_ref_vals[2, 0] = psi + diff
            else:
                diff = (X_ref_vals[2, k] - X_ref_vals[2, k - 1] + np.pi) % (2 * np.pi) - np.pi
                X_ref_vals[2, k] = X_ref_vals[2, k - 1] + diff
        x0_curr = np.array([x, y, psi, u, v, r])
        u_opt, x_opt = mpc.solve(x0_curr, X_ref_vals, U_ref_vals)
        hist_x_opt.append(x_opt)
        Tu_dem, Tr_dem = (u_opt[0], u_opt[1])
        hist_Tu_dem.append(Tu_dem)
        hist_Tr_dem.append(Tr_dem)
        y_abs = abs(y_FL)
        T_right = 0.5 * (Tu_dem + Tr_dem / y_abs)
        T_left = 0.5 * (Tu_dem - Tr_dem / y_abs)
        T_FR_dem, T_BR_dem = (T_right / 2.0, T_right / 2.0)
        T_FL_dem, T_BL_dem = (T_left / 2.0, T_left / 2.0)
        cmd_FR = inverse_thrust(T_FR_dem)
        cmd_FL = inverse_thrust(T_FL_dem)
        cmd_BR = inverse_thrust(T_BR_dem)
        cmd_BL = inverse_thrust(T_BL_dem)
        T_FR_act = forward_thrust(cmd_FR)
        T_FL_act = forward_thrust(cmd_FL)
        T_BR_act = forward_thrust(cmd_BR)
        T_BL_act = forward_thrust(cmd_BL)
        hist_T_FR.append(T_FR_act)
        hist_T_FL.append(T_FL_act)
        hist_T_BR.append(T_BR_act)
        hist_T_BL.append(T_BL_act)
        Tu_actual = T_FR_act + T_FL_act + T_BR_act + T_BL_act
        Tr_actual = y_abs * (T_FR_act + T_BR_act) - y_abs * (T_FL_act + T_BL_act)
        hist_Tu_act.append(Tu_actual)
        hist_Tr_act.append(Tr_actual)
        u, v, r, x, y, psi = rk4_step(u, v, r, x, y, psi, Tu_actual, Tr_actual, dt_sim)
        if i < n_steps - 1:
            hist_x.append(x)
            hist_y.append(y)
            hist_psi.append(psi)
            hist_u.append(u)
            hist_v.append(v)
            hist_r.append(r)
    hist_x = np.array(hist_x)
    hist_y = np.array(hist_y)
    hist_psi = np.array(hist_psi)
    hist_u = np.array(hist_u)
    hist_v = np.array(hist_v)
    hist_r = np.array(hist_r)
    err_x = hist_x - x_d_array
    err_y = hist_y - y_d_array
    err_pos = np.hypot(err_x, err_y)
    err_psi = np.degrees((hist_psi - psi_d_array + np.pi) % (2 * np.pi) - np.pi)
    res_dict = {'t_sim': t_sim, 'x_d': x_d_array, 'y_d': y_d_array, 'psi_d': psi_d_array, 'u_d': u_d_array, 'v_d': v_d_array, 'r_d': r_d_array, 'hist_x': hist_x, 'hist_y': hist_y, 'hist_psi': hist_psi, 'hist_u': hist_u, 'hist_v': hist_v, 'hist_r': hist_r, 'err_pos': err_pos, 'err_psi': err_psi, 'hist_Tu_dem': np.array(hist_Tu_dem), 'hist_Tr_dem': np.array(hist_Tr_dem), 'hist_Tu_act': np.array(hist_Tu_act), 'hist_Tr_act': np.array(hist_Tr_act), 'tau_u_plan': tau_u_plan, 'tau_r_plan': tau_r_plan, 'final_times': final_times, 'waypoints_xyz': waypoints_xyz}

    t_wp7 = final_times[7]
    mask_wp7 = t_sim <= t_wp7
    if generate_anim:
        print('\n=== STEP 3: Generating Animated GIF/MP4 Simulation ===')
        generate_animation(t_sim, x_d_array, y_d_array, psi_d_array, hist_x, hist_y, hist_psi, hist_x_opt, waypoints_xyz, anim_outpath=anim_outpath, fps=20, t_wp7=t_wp7)
    print('\n=== STEP 4: Plotting Static Diagnostic Figures ===')
    fig, axs = plt.subplots(3, 2, figsize=(14, 12))
    t_plot = t_sim[mask_wp7]
    x_d_plot = x_d_array[mask_wp7]
    y_d_plot = y_d_array[mask_wp7]
    hist_x_plot = hist_x[mask_wp7]
    hist_y_plot = hist_y[mask_wp7]
    psi_d_plot = psi_d_array[mask_wp7]
    hist_psi_plot = hist_psi[mask_wp7]
    u_d_plot = u_d_array[mask_wp7]
    hist_u_plot = hist_u[mask_wp7]
    v_d_plot = v_d_array[mask_wp7]
    hist_v_plot = hist_v[mask_wp7]
    r_d_plot = r_d_array[mask_wp7]
    hist_r_plot = hist_r[mask_wp7]
    err_pos_plot = err_pos[mask_wp7]
    err_psi_plot = err_psi[mask_wp7]
    hist_Tu_dem_plot = np.array(hist_Tu_dem)[mask_wp7]
    hist_Tu_act_plot = np.array(hist_Tu_act)[mask_wp7]
    hist_Tr_dem_plot = np.array(hist_Tr_dem)[mask_wp7]
    hist_Tr_act_plot = np.array(hist_Tr_act)[mask_wp7]
    hist_T_FR_plot = np.array(hist_T_FR)[mask_wp7]
    hist_T_FL_plot = np.array(hist_T_FL)[mask_wp7]
    hist_T_BR_plot = np.array(hist_T_BR)[mask_wp7]
    hist_T_BL_plot = np.array(hist_T_BL)[mask_wp7]
    waypoints_show = waypoints_xyz[:8]
    ax = axs[0, 0]
    ax.plot(x_d_plot, y_d_plot, 'r--', lw=2, label='Planeada (Minimum-Jerk)')
    ax.plot(hist_x_plot, hist_y_plot, 'b-', lw=2, label='Real (MPC + Planta)')
    ax.scatter(waypoints_show[:, 0], waypoints_show[:, 1], c='k', s=50, zorder=5, label='Waypoints')
    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')
    ax.set_title('Seguimiento de Trayectoria en el Plano XY')
    ax.axis('equal')
    ax.legend()
    ax.grid(True)
    ax = axs[0, 1]
    ax.plot(t_plot, np.degrees(psi_d_plot), 'r--', label='psi Planeado [deg]')
    ax.plot(t_plot, np.degrees(hist_psi_plot), 'b-', label='psi Real [deg]')
    ax.set_xlabel('t [s]')
    ax.set_ylabel('psi [deg]')
    ax.set_title('Seguimiento de Rumbo (Yaw)')
    ax.legend()
    ax.grid(True)
    ax = axs[1, 0]
    ax.plot(t_plot, u_d_plot, 'r--', label='u plan')
    ax.plot(t_plot, hist_u_plot, 'r-', label='u real')
    ax.plot(t_plot, v_d_plot, 'g--', label='v plan')
    ax.plot(t_plot, hist_v_plot, 'g-', label='v real')
    ax.plot(t_plot, r_d_plot, 'b--', label='r plan')
    ax.plot(t_plot, hist_r_plot, 'b-', label='r real')
    ax.set_xlabel('t [s]')
    ax.set_title('Velocidades en Marco Cuerpo (Planeada vs Real)')
    ax.legend(ncol=3, fontsize=8)
    ax.grid(True)
    ax = axs[1, 1]
    ax.plot(t_plot, err_pos_plot, 'm-', label='Error de Posición [m]')
    ax.plot(t_plot, err_psi_plot, 'c-', label='Error de Yaw [deg]')
    ax.set_xlabel('t [s]')
    ax.set_title('Errores de Seguimiento de la Planta')
    ax.legend()
    ax.grid(True)
    ax = axs[2, 0]
    ax.plot(t_plot, hist_Tu_dem_plot, 'r--', label='$\\tau_u$ demandada')
    ax.plot(t_plot, hist_Tu_act_plot, 'r-', label='$\\tau_u$ real')
    ax.plot(t_plot, hist_Tr_dem_plot, 'b--', label='$\\tau_r$ demandada')
    ax.plot(t_plot, hist_Tr_act_plot, 'b-', label='$\\tau_r$ real')
    ax.set_xlabel('t [s]')
    ax.set_title('Acciones de Control (Fuerza y Torque)')
    ax.legend(ncol=2, fontsize=8)
    ax.grid(True)
    ax = axs[2, 1]
    ax.plot(t_plot, hist_T_FR_plot, label='T_FR')
    ax.plot(t_plot, hist_T_FL_plot, label='T_FL')
    ax.plot(t_plot, hist_T_BR_plot, label='T_BR')
    ax.plot(t_plot, hist_T_BL_plot, label='T_BL')
    ax.axhline(T_MAX, color='k', ls='--', lw=1, label='T_max')
    ax.axhline(T_MIN, color='k', ls='--', lw=1, label='T_min')
    ax.set_xlabel('t [s]')
    ax.set_ylabel('Empuje [N]')
    ax.set_title('Empuje Individual por Propulsor (Planta)')
    ax.legend(ncol=2, fontsize=8)
    ax.grid(True)

    plt.tight_layout()
    if outpath is None:
        outpath = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'usv_optimal_mpc_simulation.png')
    out_dir = os.path.dirname(os.path.abspath(outpath))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    plt.savefig(outpath, dpi=150)
    plt.close(fig)
    print(f'\nSimulación completada con éxito. Gráfica guardada en: {outpath}')
    if return_data:
        return (res_dict, outpath, anim_outpath)
    return (outpath, anim_outpath)
if __name__ == '__main__':
    if '--save-data' in sys.argv:
        idx = sys.argv.index('--save-data')
        outfile = sys.argv[idx + 1]
        res_dict, _, _ = run_simulation(generate_anim=False, return_data=True)
        np.savez(outfile, **res_dict)
    else:
        run_simulation()
