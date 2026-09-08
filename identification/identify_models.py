#!/usr/bin/env python3

import os
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy.signal import savgol_filter
from scipy.optimize import lsq_linear, minimize

# Set aesthetic publication style
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['axes.edgecolor'] = '#333333'
plt.rcParams['axes.linewidth'] = 1.0

MOTOR_TEMPLATE = {
    'order': ['FR', 'FL', 'BR', 'BL'],
    'T200': {
        'pos': {'A': 1e-06, 'K': 40.0209, 'B': 2.6249, 'v': 0.1615, 'C': 0.9432, 'M': 1e-05},
        'neg': {'A': -31.499, 'K': -1e-05, 'B': 3.6986, 'v': 0.3264, 'C': 0.9713, 'M': -1.0}
    },
    'max_fwd': 36.3827,
    'max_rev': -28.4393,
    'positions_yx': [[-0.29, 0.60], [0.29, 0.60], [-0.29, -0.15], [0.29, -0.15]],
    'angles_deg': [0.0, 0.0, 0.0, 0.0]
}

PARAM_NAMES_FULL = ['m11', 'm22', 'm33', 'Xu', 'Xuu', 'Yv', 'Yvv', 'Nr', 'Nrr']

def branch_thrust(cmd, params):
    return params['A'] + (params['K'] - params['A']) / (params['C'] + np.exp(-params['B'] * (cmd - params['M']))) ** (1.0 / params['v'])

def cmd_to_thrust_forces(u_left, u_right, motor=None):
    if motor is None:
        motor = MOTOR_TEMPLATE
    params_pos = motor['T200']['pos']
    params_neg = motor['T200']['neg']
    max_fwd = motor['max_fwd']
    max_rev = motor['max_rev']
    
    cmd_FR = u_right
    cmd_FL = u_left
    cmd_BR = u_right
    cmd_BL = u_left

    def get_thrust(cmd):
        t = np.zeros_like(cmd)
        pos = cmd > 0.01
        neg = cmd < -0.01
        if np.any(pos):
            t[pos] = branch_thrust(cmd[pos], params_pos)
        if np.any(neg):
            t[neg] = branch_thrust(cmd[neg], params_neg)
        return np.clip(t, max_rev, max_fwd)

    T_FR = get_thrust(cmd_FR)
    T_FL = get_thrust(cmd_FL)
    T_BR = get_thrust(cmd_BR)
    T_BL = get_thrust(cmd_BL)
    y_FR = motor['positions_yx'][0][0]
    y_FL = motor['positions_yx'][1][0]
    y_BR = motor['positions_yx'][2][0]
    y_BL = motor['positions_yx'][3][0]

    Tu = T_FR + T_FL + T_BR + T_BL
    Tr = -y_FR * T_FR - y_FL * T_FL - y_BR * T_BR - y_BL * T_BL
    return Tu, Tr

def clean_outliers(sig, max_val=5.0):
    outliers = np.abs(sig) > max_val
    if np.any(outliers):
        cleaned = sig.copy()
        for i in np.where(outliers)[0]:
            valid_indices = np.where(~outliers)[0]
            if len(valid_indices) > 0:
                nearest_valid = valid_indices[np.argmin(np.abs(valid_indices - i))]
                cleaned[i] = sig[nearest_valid]
        return cleaned
    return sig

def build_phi_tau_full(acc_u, acc_v, acc_r, u, v, r, Tu, Tr):
    N = len(u)
    Phi_list, Tau_list = [], []
    for k in range(N):
        row_u = np.zeros(9)
        row_u[0] = acc_u[k]
        row_u[1] = -v[k] * r[k]
        row_u[3] = u[k]
        row_u[4] = abs(u[k]) * u[k]
        Phi_list.append(row_u)
        Tau_list.append(Tu[k])

        row_v = np.zeros(9)
        row_v[0] = u[k] * r[k]
        row_v[1] = acc_v[k]
        row_v[5] = v[k]
        row_v[6] = abs(v[k]) * v[k]
        Phi_list.append(row_v)
        Tau_list.append(0.0)

        row_r = np.zeros(9)
        row_r[0] = -u[k] * v[k]
        row_r[1] = u[k] * v[k]
        row_r[2] = acc_r[k]
        row_r[7] = r[k]
        row_r[8] = abs(r[k]) * r[k]
        Phi_list.append(row_r)
        Tau_list.append(Tr[k])
    return np.array(Phi_list), np.array(Tau_list)

def rk4_integrate_full(t, u0, v0, r0, Tu, Tr, d):
    N = len(t)
    u_sim, v_sim, r_sim = np.zeros(N), np.zeros(N), np.zeros(N)
    u_sim[0], v_sim[0], r_sim[0] = u0, v0, r0
    m11, m22, m33 = d['m11'], d['m22'], d['m33']
    Xu, Xuu = d['Xu'], d['Xuu']
    Yv, Yvv = d['Yv'], d['Yvv']
    Nr, Nrr = d['Nr'], d['Nrr']

    for k in range(N - 1):
        dt = t[k + 1] - t[k]
        uk, vk, rk = u_sim[k], v_sim[k], r_sim[k]
        tu1, tr1 = Tu[k], Tr[k]
        du1 = (tu1 + m22 * vk * rk - Xu * uk - Xuu * abs(uk) * uk) / m11
        dv1 = (-m11 * uk * rk - Yv * vk - Yvv * abs(vk) * vk) / m22
        dr1 = (tr1 - (m22 - m11) * uk * vk - Nr * rk - Nrr * abs(rk) * rk) / m33

        u2, v2, r2 = uk + 0.5 * dt * du1, vk + 0.5 * dt * dv1, rk + 0.5 * dt * dr1
        tu2, tr2 = 0.5 * (Tu[k] + Tu[k + 1]), 0.5 * (Tr[k] + Tr[k + 1])
        du2 = (tu2 + m22 * v2 * r2 - Xu * u2 - Xuu * abs(u2) * u2) / m11
        dv2 = (-m11 * u2 * r2 - Yv * v2 - Yvv * abs(v2) * v2) / m22
        dr2 = (tr2 - (m22 - m11) * u2 * v2 - Nr * r2 - Nrr * abs(r2) * r2) / m33

        u3, v3, r3 = uk + 0.5 * dt * du2, vk + 0.5 * dt * dv2, rk + 0.5 * dt * dr2
        du3 = (tu2 + m22 * v3 * r3 - Xu * u3 - Xuu * abs(u3) * u3) / m11
        dv3 = (-m11 * u3 * r3 - Yv * v3 - Yvv * abs(v3) * v3) / m22
        dr3 = (tr2 - (m22 - m11) * u3 * v3 - Nr * r3 - Nrr * abs(r3) * r3) / m33

        u4, v4, r4 = uk + dt * du3, vk + dt * dv3, rk + dt * dr3
        tu4, tr4 = Tu[k + 1], Tr[k + 1]
        du4 = (tu4 + m22 * v4 * r4 - Xu * u4 - Xuu * abs(u4) * u4) / m11
        dv4 = (-m11 * u4 * r4 - Yv * v4 - Yvv * abs(v4) * v4) / m22
        dr4 = (tr4 - (m22 - m11) * u4 * v4 - Nr * r4 - Nrr * abs(r4) * r4) / m33

        u_sim[k + 1] = uk + dt / 6.0 * (du1 + 2.0 * du2 + 2.0 * du3 + du4)
        v_sim[k + 1] = vk + dt / 6.0 * (dv1 + 2.0 * dv2 + 2.0 * dv3 + dv4)
        r_sim[k + 1] = rk + dt / 6.0 * (dr1 + 2.0 * dr2 + 2.0 * dr3 + dr4)
    return u_sim, v_sim, r_sim

def compute_metrics(y_real, y_sim):
    rms_val = float(np.sqrt(np.mean((y_real - y_sim) ** 2)))
    mae_val = float(np.mean(np.abs(y_real - y_sim)))
    return rms_val, mae_val

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, '..', 'data')
    
    mat_file = os.path.join(data_dir, 'wamvsim_20260908_102152.mat')
    if not os.path.isfile(mat_file):
        raise FileNotFoundError(f'Required identification dataset not found: {mat_file}')
    print(f'Using identification dataset: {mat_file}')
    
    mat_data = loadmat(mat_file)
    u_raw = clean_outliers(mat_data['vx'].flatten())
    v_raw = clean_outliers(mat_data['vy'].flatten())
    r_raw = clean_outliers(mat_data['wz'].flatten())
    
    u = savgol_filter(u_raw, 25, 2)
    v = savgol_filter(v_raw, 25, 2)
    r = savgol_filter(r_raw, 25, 2)
    
    dt = 1.0 / 30.0
    N_samples = len(u)
    t_arr = mat_data['t'].flatten() if 't' in mat_data else np.arange(N_samples) * dt
    
    acc_u = np.gradient(u, t_arr)
    acc_v = np.gradient(v, t_arr)
    acc_r = np.gradient(r, t_arr)

    Tu_arr, Tr_arr = cmd_to_thrust_forces(mat_data['u_left'].flatten(), mat_data['u_right'].flatten())

    train_idx = t_arr <= 240.0 if t_arr[-1] >= 260.0 else np.ones(N_samples, dtype=bool)

    # 1. Least Squares Initial Estimation
    Phi1, Tau1 = build_phi_tau_full(acc_u[train_idx], acc_v[train_idx], acc_r[train_idx], 
                                     u[train_idx], v[train_idx], r[train_idx], 
                                     Tu_arr[train_idx], Tr_arr[train_idx])
    lb1 = [10.0, 10.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    ub1 = [500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 5000.0]
    theta_ls = lsq_linear(Phi1, Tau1, bounds=(lb1, ub1)).x
    print("Least Squares estimation completed.")

    # 2. Gradient-based Optimization (Gradient Descent / L-BFGS-B Output Error Method)
    t_train = t_arr[train_idx]
    u_real_train = u_raw[train_idx]
    v_real_train = v_raw[train_idx]
    r_real_train = r_raw[train_idx]
    Tu_train = Tu_arr[train_idx]
    Tr_train = Tr_arr[train_idx]

    def objective_function(theta):
        d_temp = {name: float(theta[i]) for i, name in enumerate(PARAM_NAMES_FULL)}
        u_sim, v_sim, r_sim = rk4_integrate_full(
            t_train, u_real_train[0], v_real_train[0], r_real_train[0], 
            Tu_train, Tr_train, d_temp
        )
        err_u = np.mean((u_real_train - u_sim)**2)
        err_v = np.mean((v_real_train - v_sim)**2)
        err_r = np.mean((r_real_train - r_sim)**2)
        return err_u + err_v + err_r

    print("Starting gradient-based optimization (Gradient Descent / L-BFGS-B)...")
    res_opt = minimize(objective_function, theta_ls, method='L-BFGS-B', 
                       bounds=list(zip(lb1, ub1)), options={'maxiter': 200, 'disp': True})
    theta_opt = res_opt.x
    print("Gradient-based optimization completed.")

    dict_full = {name: float(theta_opt[i]) for i, name in enumerate(PARAM_NAMES_FULL)}
    dict_full['description'] = "Standard Fossen 3DOF 9-parameter model optimized via Least Squares + Gradient Descent"

    # Export JSON
    all_models_dict = {
        "full-dynamics": dict_full
    }
    json_path = os.path.join(script_dir, 'identified_models.json')
    with open(json_path, 'w') as f:
        json.dump(all_models_dict, f, indent=4)
    print(f"Saved optimized identified_models.json to {json_path}")

    # Full validation / simulation over entire dataset
    u_c9, v_c9, r_c9 = rk4_integrate_full(
        t_arr, u_raw[0], v_raw[0], r_raw[0], Tu_arr, Tr_arr, dict_full
    )

    rms_u, _ = compute_metrics(u_raw, u_c9)
    rms_v, _ = compute_metrics(v_raw, v_c9)
    rms_r, _ = compute_metrics(r_raw, r_c9)

    # Plot Measured vs Simulated (Optimized 9-parameter model)
    fig, axs = plt.subplots(3, 1, figsize=(10, 8), sharex=True, dpi=300)
    
    c_real = '#1F2937'   # Dark Charcoal / Navy
    c_sim = '#2563EB'    # Vivid Blue

    # 1. Surge Velocity (u)
    axs[0].plot(t_arr, u_raw, color=c_real, linestyle='-', linewidth=1.5, label='Measured (Real)', alpha=0.85)
    axs[0].plot(t_arr, u_c9, color=c_sim, linestyle='--', linewidth=1.8, label=f'Optimized Sim 9-param (RMS = {rms_u:.3f} m/s)')
    axs[0].set_ylabel('Surge $u$ [m/s]', fontsize=12, fontweight='bold')
    axs[0].set_title('USV 9-Parameter Model (LS + Gradient Descent): Measured vs. Simulated', fontsize=14, fontweight='bold', pad=12)
    axs[0].grid(True, linestyle=':', alpha=0.6)
    axs[0].legend(loc='upper right', frameon=True, facecolor='#FFFFFF', edgecolor='#CCCCCC', fontsize=10)
    axs[0].set_xlim(t_arr[0], t_arr[-1])

    # 2. Sway Velocity (v)
    axs[1].plot(t_arr, v_raw, color=c_real, linestyle='-', linewidth=1.5, label='Measured (Real)', alpha=0.85)
    axs[1].plot(t_arr, v_c9, color=c_sim, linestyle='--', linewidth=1.8, label=f'Optimized Sim 9-param (RMS = {rms_v:.3f} m/s)')
    axs[1].set_ylabel('Sway $v$ [m/s]', fontsize=12, fontweight='bold')
    axs[1].grid(True, linestyle=':', alpha=0.6)
    axs[1].legend(loc='upper right', frameon=True, facecolor='#FFFFFF', edgecolor='#CCCCCC', fontsize=10)
    axs[1].set_xlim(t_arr[0], t_arr[-1])

    # 3. Yaw Rate (r)
    axs[2].plot(t_arr, r_raw, color=c_real, linestyle='-', linewidth=1.5, label='Measured (Real)', alpha=0.85)
    axs[2].plot(t_arr, r_c9, color=c_sim, linestyle='--', linewidth=1.8, label=f'Optimized Sim 9-param (RMS = {rms_r:.3f} rad/s)')
    axs[2].set_ylabel('Yaw Rate $r$ [rad/s]', fontsize=12, fontweight='bold')
    axs[2].set_xlabel('Time [s]', fontsize=12, fontweight='bold')
    axs[2].grid(True, linestyle=':', alpha=0.6)
    axs[2].legend(loc='upper right', frameon=True, facecolor='#FFFFFF', edgecolor='#CCCCCC', fontsize=10)
    axs[2].set_xlim(t_arr[0], t_arr[-1])

    plt.tight_layout()
    plot_path = os.path.join(script_dir, 'velocity_comparison_9_parameters_optimized.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved optimized 9-parameter validation plot to {plot_path}")

if __name__ == '__main__':
    main()