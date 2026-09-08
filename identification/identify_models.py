#!/usr/bin/env python3

import os
import json
import csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from scipy.optimize import lsq_linear, minimize
from scipy.signal import savgol_filter

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
PARAM_NAMES_LINEAR = ['m11', 'm22', 'm33', 'Xu', 'Yv', 'Nr']
PARAM_NAMES_SYMMETRIC = ['N_dot_r', 'Xu', 'Yv', 'Nr']
RIGID_MASS = 23.344
RIGID_INERTIA_Z = 2.923312

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

def build_phi_tau_linear(acc_u, acc_v, acc_r, u, v, r, Tu, Tr):
    phi_list, tau_list = [], []
    for k in range(len(u)):
        row_u = np.zeros(6)
        row_u[0] = acc_u[k]
        row_u[1] = -v[k] * r[k]
        row_u[3] = u[k]
        phi_list.append(row_u)
        tau_list.append(Tu[k])

        row_v = np.zeros(6)
        row_v[0] = u[k] * r[k]
        row_v[1] = acc_v[k]
        row_v[4] = v[k]
        phi_list.append(row_v)
        tau_list.append(0.0)

        row_r = np.zeros(6)
        row_r[0] = -u[k] * v[k]
        row_r[1] = u[k] * v[k]
        row_r[2] = acc_r[k]
        row_r[5] = r[k]
        phi_list.append(row_r)
        tau_list.append(Tr[k])
    return np.array(phi_list), np.array(tau_list)

def build_phi_tau_symmetric(acc_u, acc_v, acc_r, u, v, r, Tu, Tr):
    phi_list, tau_list = [], []
    for k in range(len(u)):
        row_u = np.zeros(4)
        row_u[1] = u[k]
        phi_list.append(row_u)
        tau_list.append(Tu[k] - RIGID_MASS * (acc_u[k] - v[k] * r[k]))

        row_v = np.zeros(4)
        row_v[2] = v[k]
        phi_list.append(row_v)
        tau_list.append(-RIGID_MASS * (acc_v[k] + u[k] * r[k]))

        row_r = np.zeros(4)
        row_r[0] = -acc_r[k]
        row_r[3] = r[k]
        phi_list.append(row_r)
        tau_list.append(Tr[k] - RIGID_INERTIA_Z * acc_r[k])
    return np.array(phi_list), np.array(tau_list)

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

def full_parameters(parameter_names, values):
    parameters = dict(zip(parameter_names, values))
    if parameter_names == PARAM_NAMES_FULL:
        return {
            'm11': parameters['m11'], 'm22': parameters['m22'],
            'm33': parameters['m33'], 'Xu': parameters['Xu'],
            'Xuu': parameters['Xuu'], 'Yv': parameters['Yv'],
            'Yvv': parameters['Yvv'], 'Nr': parameters['Nr'],
            'Nrr': parameters['Nrr'],
        }
    if parameter_names == PARAM_NAMES_LINEAR:
        return {
            'm11': parameters['m11'], 'm22': parameters['m22'],
            'm33': parameters['m33'], 'Xu': parameters['Xu'],
            'Xuu': 0.0, 'Yv': parameters['Yv'], 'Yvv': 0.0,
            'Nr': parameters['Nr'], 'Nrr': 0.0,
        }
    return {
        'm11': RIGID_MASS, 'm22': RIGID_MASS,
        'm33': RIGID_INERTIA_Z - parameters['N_dot_r'], 'Xu': parameters['Xu'],
        'Xuu': 0.0, 'Yv': parameters['Yv'], 'Yvv': 0.0,
        'Nr': parameters['Nr'], 'Nrr': 0.0,
    }

def physical_parameters(parameter_names, values):
    parameters = dict(zip(parameter_names, values))
    if parameter_names == PARAM_NAMES_FULL:
        return {
            'm': RIGID_MASS, 'Iz': RIGID_INERTIA_Z,
            'X_dot_u': parameters['m11'] - RIGID_MASS,
            'Y_dot_v': parameters['m22'] - RIGID_MASS,
            'N_dot_r': parameters['m33'] - RIGID_INERTIA_Z,
            'Xu': parameters['Xu'], 'Xuu': parameters['Xuu'],
            'Yv': parameters['Yv'], 'Yvv': parameters['Yvv'],
            'Nr': parameters['Nr'], 'Nrr': parameters['Nrr'],
        }
    if parameter_names == PARAM_NAMES_LINEAR:
        return {
            'm': RIGID_MASS, 'Iz': RIGID_INERTIA_Z,
            'X_dot_u': parameters['m11'] - RIGID_MASS,
            'Y_dot_v': parameters['m22'] - RIGID_MASS,
            'N_dot_r': parameters['m33'] - RIGID_INERTIA_Z,
            'Xu': parameters['Xu'], 'Yv': parameters['Yv'],
            'Nr': parameters['Nr'],
        }
    return {
        'm': RIGID_MASS, 'Iz': RIGID_INERTIA_Z,
        'X_dot_u': 0.0, 'Y_dot_v': 0.0,
        'N_dot_r': parameters['N_dot_r'],
        'Xu': parameters['Xu'], 'Yv': parameters['Yv'],
        'Nr': parameters['Nr'],
    }

def compute_metrics(y_real, y_sim):
    rms_val = float(np.sqrt(np.mean((y_real - y_sim) ** 2)))
    mae_val = float(np.mean(np.abs(y_real - y_sim)))
    return rms_val, mae_val

def clean_outliers(signal, max_value=5.0):
    outliers = np.abs(signal) > max_value
    if not np.any(outliers):
        return signal

    cleaned = signal.copy()
    valid_indices = np.where(~outliers)[0]
    for index in np.where(outliers)[0]:
        if valid_indices.size:
            nearest_valid = valid_indices[np.argmin(np.abs(valid_indices - index))]
            cleaned[index] = signal[nearest_valid]
    return cleaned

def load_latest_dataset(data_dir):
    csv_files = list(Path(data_dir).rglob('wamvsim_*.csv'))
    if not csv_files:
        raise FileNotFoundError(f'No identification CSV files found in: {data_dir}')

    csv_file = max(csv_files, key=lambda path: path.stat().st_mtime)
    with csv_file.open(mode='r', newline='') as file:
        reader = csv.DictReader(file)
        rows = list(reader)

    required_fields = {'t', 'vx', 'vy', 'wz', 'u_left', 'u_right'}
    missing_fields = required_fields.difference(reader.fieldnames or [])
    if missing_fields:
        missing = ', '.join(sorted(missing_fields))
        raise ValueError(f'Dataset {csv_file} is missing columns: {missing}')
    if not rows:
        raise ValueError(f'Dataset is empty: {csv_file}')

    values = {
        field: np.asarray([float(row[field]) for row in rows], dtype=float)
        for field in required_fields
    }
    return csv_file, values

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, '..', 'data')
    csv_file, dataset = load_latest_dataset(data_dir)
    print(f'Using latest identification dataset: {csv_file}')

    u_raw = clean_outliers(dataset['vx'])
    v_raw = clean_outliers(dataset['vy'])
    r_raw = clean_outliers(dataset['wz'])

    u = savgol_filter(u_raw, 25, 2)
    v = savgol_filter(v_raw, 25, 2)
    r = savgol_filter(r_raw, 25, 2)
    
    N_samples = len(u)
    t_arr = dataset['t']
    
    acc_u = np.gradient(u, t_arr)
    acc_v = np.gradient(v, t_arr)
    acc_r = np.gradient(r, t_arr)

    Tu_arr, Tr_arr = cmd_to_thrust_forces(dataset['u_left'], dataset['u_right'])

    train_idx = t_arr <= 240.0 if t_arr[-1] >= 260.0 else np.ones(N_samples, dtype=bool)

    # 1. Least Squares Initial Estimation
    Phi1, Tau1 = build_phi_tau_full(acc_u[train_idx], acc_v[train_idx], acc_r[train_idx], 
                                     u[train_idx], v[train_idx], r[train_idx], 
                                     Tu_arr[train_idx], Tr_arr[train_idx])
    lb1 = [10.0, 10.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    ub1 = [500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 5000.0]
    theta_ls = lsq_linear(Phi1, Tau1, bounds=(lb1, ub1)).x
    print("Least Squares estimation completed.")

    Phi6, Tau6 = build_phi_tau_linear(
        acc_u[train_idx], acc_v[train_idx], acc_r[train_idx],
        u[train_idx], v[train_idx], r[train_idx],
        Tu_arr[train_idx], Tr_arr[train_idx]
    )
    lb6 = [10.0, 10.0, 2.0, 0.0, 0.0, 0.0]
    ub6 = [500.0, 500.0, 500.0, 500.0, 500.0, 500.0]
    theta6 = lsq_linear(Phi6, Tau6, bounds=(lb6, ub6)).x
    print("6-parameter least squares estimation completed.")

    Phi5, Tau5 = build_phi_tau_symmetric(
        acc_u[train_idx], acc_v[train_idx], acc_r[train_idx],
        u[train_idx], v[train_idx], r[train_idx],
        Tu_arr[train_idx], Tr_arr[train_idx]
    )
    lb5 = [-500.0, 0.0, 0.0, 0.0]
    ub5 = [500.0, 500.0, 500.0, 500.0]
    theta5 = lsq_linear(Phi5, Tau5, bounds=(lb5, ub5)).x
    print("5-parameter least squares estimation completed.")

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

    dict_full = physical_parameters(PARAM_NAMES_FULL, theta_opt)
    dict_full['description'] = "Standard Fossen 3DOF 9-parameter model optimized via Least Squares + Gradient Descent"

    dict_linear = physical_parameters(PARAM_NAMES_LINEAR, theta6)
    dict_linear['description'] = 'Fossen 3DOF 6-parameter linear-damping model estimated by least squares'

    dict_symmetric = physical_parameters(PARAM_NAMES_SYMMETRIC, theta5)
    dict_symmetric['description'] = 'Fossen 3DOF model with X_dot_u = Y_dot_v = 0 estimated by least squares'

    full_linear = full_parameters(PARAM_NAMES_LINEAR, theta6)
    full_symmetric = full_parameters(PARAM_NAMES_SYMMETRIC, theta5)
    full_dynamics = full_parameters(PARAM_NAMES_FULL, theta_opt)

    # Export JSON
    all_models_dict = {
        "full-dynamics": dict_full,
        "linear-6-parameters": dict_linear,
        "symmetric-5-parameters": dict_symmetric,
    }
    json_path = os.path.join(script_dir, 'identified_models.json')
    with open(json_path, 'w') as f:
        json.dump(all_models_dict, f, indent=4)
    print(f"Saved optimized identified_models.json to {json_path}")

    # Full validation / simulation over entire dataset
    u_c9, v_c9, r_c9 = rk4_integrate_full(
        t_arr, u[0], v[0], r[0], Tu_arr, Tr_arr, full_dynamics
    )
    u_c6, v_c6, r_c6 = rk4_integrate_full(
        t_arr, u[0], v[0], r[0], Tu_arr, Tr_arr, full_linear
    )
    u_c5, v_c5, r_c5 = rk4_integrate_full(
        t_arr, u[0], v[0], r[0], Tu_arr, Tr_arr, full_symmetric
    )

    rms_u, _ = compute_metrics(u_raw, u_c9)
    rms_v, _ = compute_metrics(v_raw, v_c9)
    rms_r, _ = compute_metrics(r_raw, r_c9)
    rms_u6, _ = compute_metrics(u_raw, u_c6)
    rms_v6, _ = compute_metrics(v_raw, v_c6)
    rms_r6, _ = compute_metrics(r_raw, r_c6)
    rms_u5, _ = compute_metrics(u_raw, u_c5)
    rms_v5, _ = compute_metrics(v_raw, v_c5)
    rms_r5, _ = compute_metrics(r_raw, r_c5)

    plt.rcParams.update({
        'font.size': 10.0,
        'axes.labelsize': 10.0,
        'xtick.labelsize': 8.0,
        'ytick.labelsize': 8.0,
        'font.family': 'serif',
        'mathtext.fontset': 'cm',
        'figure.facecolor': 'white'
    })

    fig, axs = plt.subplots(2, 2, figsize=(8.0, 5.2), dpi=300)
    c_real, c_5param, c_6param, c_9param = '#000000', '#B22222', '#003366', '#2E8B57'
    line_width = 1.2
    t_max = t_arr[-1]

    axs[0, 0].plot(t_arr, u_raw, color=c_real, linestyle='-', label='Real', linewidth=line_width)
    axs[0, 0].plot(t_arr, u_c5, color=c_5param, linestyle=':', label='5-Param', linewidth=line_width)
    axs[0, 0].plot(t_arr, u_c6, color=c_6param, linestyle='-.', label='6-Param', linewidth=line_width)
    axs[0, 0].plot(t_arr, u_c9, color=c_9param, linestyle='--', label='9-Param', linewidth=line_width)
    axs[0, 0].set_ylabel(r'Surge Velocity $u \ [\mathrm{m/s}]$')
    axs[0, 0].set_xlabel(r'Time $t \ [\mathrm{s}]$')

    axs[0, 1].plot(t_arr, v_raw, color=c_real, linestyle='-', label='Real', linewidth=line_width)
    axs[0, 1].plot(t_arr, v_c5, color=c_5param, linestyle=':', label='5-Param', linewidth=line_width)
    axs[0, 1].plot(t_arr, v_c6, color=c_6param, linestyle='-.', label='6-Param', linewidth=line_width)
    axs[0, 1].plot(t_arr, v_c9, color=c_9param, linestyle='--', label='9-Param', linewidth=line_width)
    axs[0, 1].set_ylabel(r'Sway Velocity $v \ [\mathrm{m/s}]$')
    axs[0, 1].set_xlabel(r'Time $t \ [\mathrm{s}]$')

    axs[1, 0].plot(t_arr, r_raw, color=c_real, linestyle='-', label='Real', linewidth=line_width)
    axs[1, 0].plot(t_arr, r_c5, color=c_5param, linestyle=':', label='5-Param', linewidth=line_width)
    axs[1, 0].plot(t_arr, r_c6, color=c_6param, linestyle='-.', label='6-Param', linewidth=line_width)
    axs[1, 0].plot(t_arr, r_c9, color=c_9param, linestyle='--', label='9-Param', linewidth=line_width)
    axs[1, 0].set_ylabel(r'Yaw Rate $r \ [\mathrm{rad/s}]$')
    axs[1, 0].set_xlabel(r'Time $t \ [\mathrm{s}]$')

    for axis in (axs[0, 0], axs[0, 1], axs[1, 0]):
        axis.set_xlim(0, t_max)
        axis.grid(True, linestyle='--', alpha=0.7)
        axis.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))

    categories = [
        'Surge Velocity\n' + r'$u \ [\mathrm{m/s}]$',
        'Sway Velocity\n' + r'$v \ [\mathrm{m/s}]$',
        'Yaw Rate\n' + r'$r \ [\mathrm{rad/s}]$'
    ]
    x_pos = np.arange(len(categories))
    bar_width = 0.24
    axs[1, 1].bar(x_pos - bar_width, [rms_u5, rms_v5, rms_r5], bar_width, color=c_5param, edgecolor='black')
    axs[1, 1].bar(x_pos, [rms_u6, rms_v6, rms_r6], bar_width, color=c_6param, edgecolor='black')
    axs[1, 1].bar(x_pos + bar_width, [rms_u, rms_v, rms_r], bar_width, color=c_9param, edgecolor='black')
    axs[1, 1].set_ylabel(r'$\mathrm{RMSE \ Error}$')
    axs[1, 1].set_xticks(x_pos)
    axs[1, 1].set_xticklabels(categories)
    axs[1, 1].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    axs[1, 1].grid(True, axis='y', linestyle='--', alpha=0.7)

    handles, labels = axs[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=4,
               bbox_to_anchor=(0.5, -0.01), frameon=True, edgecolor='black')
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plot_path = os.path.join(script_dir, 'velocity_comparison_9_parameters_optimized.pdf')
    plt.savefig(plot_path, format='pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved optimized 9-parameter validation plot to {plot_path}")

if __name__ == '__main__':
    main()