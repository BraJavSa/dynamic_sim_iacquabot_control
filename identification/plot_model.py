import os
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.io import loadmat
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
    
    # Load identified models from JSON
    json_path = os.path.join(script_dir, 'identified_models.json')
    with open(json_path, 'r') as f:
        models_dict = json.load(f)
    
    dict_full = models_dict['full-dynamics']
    print("Loaded 9-parameter model parameters from JSON.")

    mat_file = os.path.join(data_dir, 'wamvsim_20260908_035002.mat')
    mat_data = loadmat(mat_file)
    
    u_raw = clean_outliers(mat_data['vx'].flatten())
    v_raw = clean_outliers(mat_data['vy'].flatten())
    r_raw = clean_outliers(mat_data['wz'].flatten())
    
    dt = 1.0 / 30.0
    N_samples = len(u_raw)
    t_arr = mat_data['t'].flatten() if 't' in mat_data else np.arange(N_samples) * dt

    Tu_arr, Tr_arr = cmd_to_thrust_forces(mat_data['u_left'].flatten(), mat_data['u_right'].flatten())

    # Simulate using 9-parameter model
    u_c9, v_c9, r_c9 = rk4_integrate_full(
        t_arr, u_raw[0], v_raw[0], r_raw[0], Tu_arr, Tr_arr, dict_full
    )

    # Compute error metrics
    rms_u, _ = compute_metrics(u_raw, u_c9)
    rms_v, _ = compute_metrics(v_raw, v_c9)
    rms_r, _ = compute_metrics(r_raw, r_c9)

    # Plot Measured vs Simulated (9-parameter model only)
    fig, axs = plt.subplots(3, 1, figsize=(10, 8), sharex=True, dpi=300)
    
    c_real = '#1F2937'   # Dark Charcoal / Navy
    c_sim = '#2563EB'    # Vivid Blue for 9-param model

    # 1. Surge Velocity (u)
    axs[0].plot(t_arr, u_raw, color=c_real, linestyle='-', linewidth=1.5, label='Measured (Real)', alpha=0.85)
    axs[0].plot(t_arr, u_c9, color=c_sim, linestyle='--', linewidth=1.8, label=f'Simulated 9-param (RMS = {rms_u:.3f} m/s)')
    axs[0].set_ylabel('Surge $u$ [m/s]', fontsize=12, fontweight='bold')
    axs[0].set_title('USV 9-Parameter Model: Measured vs. Simulated Validation', fontsize=14, fontweight='bold', pad=12)
    axs[0].grid(True, linestyle=':', alpha=0.6)
    axs[0].legend(loc='upper right', frameon=True, facecolor='#FFFFFF', edgecolor='#CCCCCC', fontsize=10)
    axs[0].set_xlim(t_arr[0], t_arr[-1])

    # 2. Sway Velocity (v)
    axs[1].plot(t_arr, v_raw, color=c_real, linestyle='-', linewidth=1.5, label='Measured (Real)', alpha=0.85)
    axs[1].plot(t_arr, v_c9, color=c_sim, linestyle='--', linewidth=1.8, label=f'Simulated 9-param (RMS = {rms_v:.3f} m/s)')
    axs[1].set_ylabel('Sway $v$ [m/s]', fontsize=12, fontweight='bold')
    axs[1].grid(True, linestyle=':', alpha=0.6)
    axs[1].legend(loc='upper right', frameon=True, facecolor='#FFFFFF', edgecolor='#CCCCCC', fontsize=10)
    axs[1].set_xlim(t_arr[0], t_arr[-1])

    # 3. Yaw Rate (r)
    axs[2].plot(t_arr, r_raw, color=c_real, linestyle='-', linewidth=1.5, label='Measured (Real)', alpha=0.85)
    axs[2].plot(t_arr, r_c9, color=c_sim, linestyle='--', linewidth=1.8, label=f'Simulated 9-param (RMS = {rms_r:.3f} rad/s)')
    axs[2].set_ylabel('Yaw Rate $r$ [rad/s]', fontsize=12, fontweight='bold')
    axs[2].set_xlabel('Time [s]', fontsize=12, fontweight='bold')
    axs[2].grid(True, linestyle=':', alpha=0.6)
    axs[2].legend(loc='upper right', frameon=True, facecolor='#FFFFFF', edgecolor='#CCCCCC', fontsize=10)
    axs[2].set_xlim(t_arr[0], t_arr[-1])

    plt.tight_layout()
    plot_path = os.path.join(script_dir, 'velocity_comparison_9_parameters_only.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved 9-parameter validation plot to {plot_path}")

if __name__ == '__main__':
    main()