#!/usr/bin/env python3

import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
sys.path.append(os.path.dirname(__file__))
from usv_mpc import USV_MPC
import sys
sys.path.append(os.path.dirname(__file__))
from usv_mpc import USV_MPC
m11 = 90.03238
m22 = 91.066933
m33 = 334.867746
Xu = 23.237495
Xuu = 38.22715
Yv = 26.016985
Yvv = 56.929867
Nr = 79.463013
Nrr = 30.613898
y_FR = -1.027135
y_FL = 1.027135
y_BR = -1.027135
y_BL = 1.027135
T200 = {'pos': {'A': 1e-06, 'K': 40.0209, 'B': 2.6249, 'v': 0.1615, 'C': 0.9432, 'M': 1e-05}, 'neg': {'A': -31.499, 'K': -1e-05, 'B': 3.6986, 'v': 0.3264, 'C': 0.9713, 'M': -1.0}}
max_fwd = 36.3827
max_rev = -28.4393
Kp = np.diag([10.0, 10.0, 10.0])
Kd = np.diag([5.0, 5.0, 5.0])

def inverse_thrust(T):
    T = np.clip(T, max_rev, max_fwd)
    if abs(T) < 0.01:
        return 0.0
    params = T200['pos'] if T > 0 else T200['neg']
    A, K, B, v, C, M = (params['A'], params['K'], params['B'], params['v'], params['C'], params['M'])
    ratio = (K - A) / (T - A)
    if ratio <= 0:
        return 0.0
    term = ratio ** v - C
    if term <= 0:
        return 0.0
    return float(np.clip(M - 1.0 / B * np.log(term), -1.0, 1.0))

def forward_thrust(cmd):
    if abs(cmd) < 0.01:
        return 0.0
    params = T200['pos'] if cmd > 0 else T200['neg']
    A, K, B, v, C, M = (params['A'], params['K'], params['B'], params['v'], params['C'], params['M'])
    return A + (K - A) / (C + np.exp(-B * (cmd - M))) ** (1.0 / v)

def plant_derivative(u, v, r, Tu, Tr):
    du = (Tu + m22 * v * r - Xu * u - Xuu * abs(u) * u) / m11
    dv = (-m11 * u * r - Yv * v - Yvv * abs(v) * v) / m22
    dr = (Tr - (m22 - m11) * u * v - Nr * r - Nrr * abs(r) * r) / m33
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

def main():
    print('Loading trajectory...')
    csv_path = '/home/brayan/vrx_ws/src/dynamic_sim_iacquabot_control/src/trajectory_planning/path_results/vessel_minco_time_sqp_reconstruction.csv'
    df = pd.read_csv(csv_path)
    dt = 1.0 / 30.0
    rt_factor = 5.0
    sleep_time = dt / rt_factor
    t_csv = df['t'].values
    t_sim = np.arange(0, t_csv[-1], dt)
    x_d_array = np.interp(t_sim, t_csv, df['x'].values)
    y_d_array = np.interp(t_sim, t_csv, df['y'].values)
    psi_d_array = np.unwrap(df['psi'].values)
    psi_d_array = np.interp(t_sim, t_csv, psi_d_array)
    qw_d_array = np.cos(psi_d_array / 2.0)
    qz_d_array = np.sin(psi_d_array / 2.0)
    u_d_array = np.interp(t_sim, t_csv, df['u'].values)
    v_d_array = np.interp(t_sim, t_csv, df['v'].values)
    r_d_array = np.interp(t_sim, t_csv, df['r'].values)
    udot_d = np.gradient(u_d_array, dt)
    vdot_d = np.gradient(v_d_array, dt)
    rdot_d = np.gradient(r_d_array, dt)
    x, y, psi = (x_d_array[0], y_d_array[0], psi_d_array[0])
    u, v, r = (u_d_array[0], v_d_array[0], r_d_array[0])
    qw = np.cos(psi / 2.0)
    qz = np.sin(psi / 2.0)
    hist_x, hist_y, hist_psi = ([], [], [])
    hist_u, hist_v, hist_r = ([], [], [])
    print(f'Simulating {len(t_sim)} steps at 30Hz, running {rt_factor}x faster than real-time...')
    start_wall = time.time()
    import matplotlib
    matplotlib.use('TkAgg')
    import matplotlib.pyplot as plt
    plt.ion()
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.plot(df['x'], df['y'], 'k--', alpha=0.5, label='Ruta Planeada')
    line_real, = ax.plot([], [], 'b-', alpha=0.6, label='Trayectoria Real')
    poly_des, = ax.plot([], [], 'r-', linewidth=2, label='Deseado (Triángulo Rojo)')
    poly_real, = ax.plot([], [], 'g-', linewidth=2, label='Real (Triángulo Verde)')
    line_mpc, = ax.plot([], [], 'y-', linewidth=1.5, label='Horizonte MPC (5s)')
    ax.set_xlabel('X [m]')
    ax.set_ylabel('Y [m]')
    ax.set_title('Simulación Animada (30Hz, x5 RTF)')
    ax.legend(loc='upper right')
    ax.grid(True)
    ax.set_aspect('equal')
    plt.show(block=False)

    def get_boat_polygon(x, y, psi, scale=1.5):
        pts = np.array([[scale, 0], [-scale / 1.5, scale / 2], [-scale / 1.5, -scale / 2], [scale, 0]])
        c, s = (np.cos(psi), np.sin(psi))
        R = np.array([[c, -s], [s, c]])
        pts_rot = (R @ pts.T).T
        pts_rot[:, 0] += x
        pts_rot[:, 1] += y
        return pts_rot
    dt_mpc = 0.2
    N_mpc = 25
    mpc = USV_MPC(dt_mpc, N_mpc)
    step_mpc = int(dt_mpc / dt)
    Z_d_array = np.vstack((x_d_array, y_d_array, psi_d_array, u_d_array, v_d_array, r_d_array))
    for i in range(len(t_sim)):
        x_d, y_d, psi_d = (x_d_array[i], y_d_array[i], psi_d_array[i])
        X_ref_vals = np.zeros((6, N_mpc + 1))
        for k in range(N_mpc + 1):
            idx = i + k * step_mpc
            if idx >= len(t_sim):
                idx = len(t_sim) - 1
            X_ref_vals[:, k] = Z_d_array[:, idx]
            diff = (X_ref_vals[2, k] - psi + np.pi) % (2 * np.pi) - np.pi
            X_ref_vals[2, k] = psi + diff
        x0 = np.array([x, y, psi, u, v, r])
        u_opt, x_opt = mpc.solve(x0, X_ref_vals)
        tau_u = u_opt[0]
        tau_r = u_opt[1]
        y_left = abs(y_FL)
        T_right = 0.5 * (tau_u + tau_r / y_left)
        T_left = 0.5 * (tau_u - tau_r / y_left)
        T_FR, T_BR = (T_right / 2.0, T_right / 2.0)
        T_FL, T_BL = (T_left / 2.0, T_left / 2.0)
        cmd_FR = inverse_thrust(T_FR)
        cmd_FL = inverse_thrust(T_FL)
        cmd_BR = inverse_thrust(T_BR)
        cmd_BL = inverse_thrust(T_BL)
        T_FR_actual = forward_thrust(cmd_FR)
        T_FL_actual = forward_thrust(cmd_FL)
        T_BR_actual = forward_thrust(cmd_BR)
        T_BL_actual = forward_thrust(cmd_BL)
        Tu_actual = T_FR_actual + T_FL_actual + T_BR_actual + T_BL_actual
        Tr_actual = y_left * (T_FR_actual + T_BR_actual) - y_left * (T_FL_actual + T_BL_actual)
        u, v, r, x, y, psi = rk4_step(u, v, r, x, y, psi, Tu_actual, Tr_actual, dt)
        qw = np.cos(psi / 2.0)
        qz = np.sin(psi / 2.0)
        hist_x.append(x)
        hist_y.append(y)
        hist_psi.append(psi)
        hist_u.append(u)
        hist_v.append(v)
        hist_r.append(r)
        t_target = start_wall + i * dt / rt_factor
        t_now = time.time()
        if t_now < t_target:
            time.sleep(t_target - t_now)
        if i % 10 == 0:
            pts_des = get_boat_polygon(x_d, y_d, psi_d)
            pts_real = get_boat_polygon(x, y, psi)
            poly_des.set_data(pts_des[:, 0], pts_des[:, 1])
            poly_real.set_data(pts_real[:, 0], pts_real[:, 1])
            line_real.set_data(hist_x, hist_y)
            line_mpc.set_data(x_opt[0, :], x_opt[1, :])
            ax.set_xlim(x - 15, x + 15)
            ax.set_ylim(y - 15, y + 15)
            fig.canvas.flush_events()
    end_wall = time.time()
    sim_time = len(t_sim) * dt
    print(f'\nSimulation complete!')
    print(f'Simulation Time (Virtual): {sim_time:.2f} s')
    print(f'Wall Clock Time (Actual) : {end_wall - start_wall:.2f} s')
    print(f'Real-Time Factor         : {sim_time / (end_wall - start_wall):.2f}x')
    plt.ioff()
    plt.show(block=True)
if __name__ == '__main__':
    main()
