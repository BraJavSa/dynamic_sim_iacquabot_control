import numpy as np
from usv_params import B_MATRIX, T_MAX, T_MIN, cmd_from_thrust_array, thrust_from_cmd_poly
from usv_params_paper import m as M_PAPER, Iz as IZ_PAPER, beta1 as BETA1, beta2 as BETA2, beta3 as BETA3
B_ACT = B_MATRIX[[0, 2], :]
B_ACT_PINV = np.linalg.pinv(B_ACT)

def reconstruct_state_and_wrench(F, F_d, F_dd, t):
    F1, F2 = (F[:, 0], F[:, 1])
    F1_d, F2_d = (F_d[:, 0], F_d[:, 1])
    F1_dd, F2_dd = (F_dd[:, 0], F_dd[:, 1])
    speed = np.hypot(F1_d, F2_d)
    psi_vel_raw = np.arctan2(F2_d, F1_d)
    valid_vel = speed > 0.001
    idx = np.arange(len(t))
    if np.any(valid_vel):
        psi_vel_unwrapped = np.unwrap(psi_vel_raw[valid_vel])
        psi = np.interp(idx, idx[valid_vel], psi_vel_unwrapped)
    else:
        psi = np.zeros_like(t)
    u = F2_d * np.sin(psi) + F1_d * np.cos(psi)
    v = F2_d * np.cos(psi) - F1_d * np.sin(psi)
    r = np.gradient(psi, t)
    r_dot = np.gradient(r, t)
    tau1 = (F2_dd - BETA1 * F2_d) * np.sin(psi) + (F1_dd - BETA1 * F1_d) * np.cos(psi)
    tau3 = r_dot + BETA3 * r
    tau_u = M_PAPER * tau1
    tau_r = IZ_PAPER * tau3
    eta = np.column_stack([F1, F2, psi])
    nu = np.column_stack([u, v, r])
    tau_actuado = np.column_stack([tau_u, tau_r])
    return (eta, nu, tau_actuado)

def wrench_to_thrusters(tau_actuado):
    return tau_actuado @ B_ACT_PINV.T

def check_saturation(T, t_max=T_MAX, t_min=T_MIN):
    over_max = T > t_max
    under_min = T < t_min
    violated = np.any(over_max) or np.any(under_min)
    summary = {'violated': bool(violated), 'max_thrust_demanded': float(T.max()), 'min_thrust_demanded': float(T.min()), 'n_samples_over_max': int(np.sum(over_max)), 'n_samples_under_min': int(np.sum(under_min)), 't_max_limit': t_max, 't_min_limit': t_min}
    return (summary, over_max | under_min)

def thrust_to_cmd(T):
    return cmd_from_thrust_array(T, c_min=-1.0, c_max=1.0)

def cmd_to_actual_thrust(c):
    return thrust_from_cmd_poly(c)
