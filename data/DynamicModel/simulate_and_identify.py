#!/usr/bin/env python3
import os
import json
import pandas as pd
import numpy as np
from scipy.optimize import lsq_linear

M11_REAL = 50.05
M22_REAL = 84.36
M33_REAL = 17.21
XU_REAL = 151.57
YV_REAL = 132.50
NR_REAL = 34.56

DP = 0.26

A_POS, K_POS, B_POS, M_POS, V_POS, C_POS = -12.07098855, 73.72259622, 14.20242467, 0.99474311, 6.83239913, 1.0
A_NEG, K_NEG, B_NEG, M_NEG, V_NEG, C_NEG = -70.9610860, 7.47710923, 2.69365001, -3.79303820, 4.09908178e-04, 1.0
MAX_FORCE_FWD = 65.92
MAX_FORCE_REV = -49.38

def thruster_thrust(cmd):
    cmd = np.atleast_1d(np.asarray(cmd, dtype=float))
    T = np.zeros_like(cmd)
    pos = cmd > 0.01
    neg = cmd < -0.01
    if np.any(pos):
        cp = cmd[pos]
        T[pos] = A_POS + (K_POS - A_POS) / ((C_POS + np.exp(-B_POS * (cp - M_POS))) ** (1.0 / V_POS))
    if np.any(neg):
        cn = cmd[neg]
        T[neg] = A_NEG + (K_NEG - A_NEG) / ((C_NEG + np.exp(-B_NEG * (cn - M_NEG))) ** (1.0 / V_NEG))
    return np.clip(T, MAX_FORCE_REV, MAX_FORCE_FWD)

def compute_forces(cmd_left, cmd_right):
    T1 = thruster_thrust(cmd_left)
    T2 = thruster_thrust(cmd_right)
    Tu = T1 + T2
    Tr = (T1 - T2) * DP
    return Tu, Tr

def derivatives_6param(state, Tu, Tr, m11, m22, m33, Xu, Yv, Nr):
    x, y, psi, u, v, r = state
    dx = u * np.cos(psi) - v * np.sin(psi)
    dy = u * np.sin(psi) + v * np.cos(psi)
    dpsi = r
    du = (Tu + m22 * v * r - Xu * u) / m11
    dv = (-m11 * u * r - Yv * v) / m22
    dr = (Tr + (m11 - m22) * u * v - Nr * r) / m33
    return np.array([dx, dy, dpsi, du, dv, dr])

def rk4_simulate(t_arr, cmd_left, cmd_right, params):
    N = len(t_arr)
    state = np.zeros((N, 6))
    Tu_arr, Tr_arr = compute_forces(cmd_left, cmd_right)
    
    m11 = params.get('m11', M11_REAL)
    m22 = params.get('m22', M22_REAL)
    m33 = params.get('m33', M33_REAL)
    Xu = params.get('Xu', XU_REAL)
    Yv = params.get('Yv', YV_REAL)
    Nr = params.get('Nr', NR_REAL)
    
    for k in range(N - 1):
        dt = t_arr[k + 1] - t_arr[k]
        sk = state[k]
        tu1, tr1 = Tu_arr[k], Tr_arr[k]
        tu_mid, tr_mid = 0.5 * (Tu_arr[k] + Tu_arr[k + 1]), 0.5 * (Tr_arr[k] + Tr_arr[k + 1])
        tu2, tr2 = Tu_arr[k + 1], Tr_arr[k + 1]
        
        k1 = derivatives_6param(sk, tu1, tr1, m11, m22, m33, Xu, Yv, Nr)
        k2 = derivatives_6param(sk + 0.5 * dt * k1, tu_mid, tr_mid, m11, m22, m33, Xu, Yv, Nr)
        k3 = derivatives_6param(sk + 0.5 * dt * k2, tu_mid, tr_mid, m11, m22, m33, Xu, Yv, Nr)
        k4 = derivatives_6param(sk + dt * k3, tu2, tr2, m11, m22, m33, Xu, Yv, Nr)
        
        state[k + 1] = sk + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        
    return state, Tu_arr, Tr_arr

def perform_least_squares_6param(t_arr, state, Tu_arr, Tr_arr):
    u = state[:, 3]
    v = state[:, 4]
    r = state[:, 5]
    
    du = np.gradient(u, t_arr)
    dv = np.gradient(v, t_arr)
    dr = np.gradient(r, t_arr)
    
    N = len(t_arr)
    Phi_list, Tau_list = [], []
    for k in range(N):
        Phi_list.append([du[k], -v[k] * r[k], 0.0, u[k], 0.0, 0.0])
        Tau_list.append(Tu_arr[k])
        
        Phi_list.append([u[k] * r[k], dv[k], 0.0, 0.0, v[k], 0.0])
        Tau_list.append(0.0)
        
        Phi_list.append([-u[k] * v[k], u[k] * v[k], dr[k], 0.0, 0.0, r[k]])
        Tau_list.append(Tr_arr[k])
        
    Phi = np.array(Phi_list)
    Tau = np.array(Tau_list)
    
    lb = [1.0, 1.0, 0.1, 0.0, 0.0, 0.0]
    ub = [500.0, 500.0, 500.0, 1000.0, 1000.0, 1000.0]
    res = lsq_linear(Phi, Tau, bounds=(lb, ub))
    theta = res.x
    
    return {
        'm11': float(theta[0]), 'm22': float(theta[1]), 'm33': float(theta[2]),
        'Xu': float(theta[3]), 'Yv': float(theta[4]), 'Nr': float(theta[5])
    }

def perform_least_squares_5param(t_arr, state, Tu_arr, Tr_arr):
    u = state[:, 3]
    v = state[:, 4]
    r = state[:, 5]
    
    du = np.gradient(u, t_arr)
    dv = np.gradient(v, t_arr)
    dr = np.gradient(r, t_arr)
    
    N = len(t_arr)
    Phi_list, Tau_list = [], []
    for k in range(N):
        Phi_list.append([du[k] - v[k] * r[k], 0.0, u[k], 0.0, 0.0])
        Tau_list.append(Tu_arr[k])
        
        Phi_list.append([dv[k] + u[k] * r[k], 0.0, 0.0, v[k], 0.0])
        Tau_list.append(0.0)
        
        Phi_list.append([0.0, dr[k], 0.0, 0.0, r[k]])
        Tau_list.append(Tr_arr[k])
        
    Phi = np.array(Phi_list)
    Tau = np.array(Tau_list)
    
    lb = [1.0, 0.1, 0.0, 0.0, 0.0]
    ub = [500.0, 500.0, 1000.0, 1000.0, 1000.0]
    res = lsq_linear(Phi, Tau, bounds=(lb, ub))
    theta = res.x
    
    return {
        'm': float(theta[0]), 'm11': float(theta[0]), 'm22': float(theta[0]),
        'm33': float(theta[1]), 'Xu': float(theta[2]), 'Yv': float(theta[3]), 'Nr': float(theta[4])
    }

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_30_path = os.path.join(script_dir, 'experiment_cmd_30Hz_simple.csv')
    
    df_30 = pd.read_csv(csv_30_path)
    t_30 = np.arange(len(df_30)) * (1.0 / 30.0)
    cmd_l_30 = df_30['cmd_left'].values
    cmd_r_30 = df_30['cmd_right'].values
    
    real_params_6 = {'m11': M11_REAL, 'm22': M22_REAL, 'm33': M33_REAL, 'Xu': XU_REAL, 'Yv': YV_REAL, 'Nr': NR_REAL}
    state_30_real, Tu_30, Tr_30 = rk4_simulate(t_30, cmd_l_30, cmd_r_30, params=real_params_6)
    
    params_6id = perform_least_squares_6param(t_30, state_30_real, Tu_30, Tr_30)
    params_5id = perform_least_squares_5param(t_30, state_30_real, Tu_30, Tr_30)
    
    json_6param = {
        "model_type": "6-parameter Fossen model identified at 30Hz",
        "m11": params_6id['m11'], "m22": params_6id['m22'], "m33": params_6id['m33'],
        "Xu": params_6id['Xu'], "Yv": params_6id['Yv'], "Nr": params_6id['Nr'],
        "dP": DP,
        "units": {"m11": "kg", "m22": "kg", "m33": "kg*m^2", "Xu": "N*s/m", "Yv": "N*s/m", "Nr": "N*m*s", "dP": "m"}
    }
    
    json_5param = {
        "model_type": "5-parameter simplified model (m11 = m22 = m) identified at 30Hz",
        "m": params_5id['m'], "m11": params_5id['m11'], "m22": params_5id['m22'],
        "m33": params_5id['m33'], "Xu": params_5id['Xu'], "Yv": params_5id['Yv'],
        "Nr": params_5id['Nr'], "dP": DP,
        "units": {"m": "kg", "m11": "kg", "m22": "kg", "m33": "kg*m^2", "Xu": "N*s/m", "Yv": "N*s/m", "Nr": "N*m*s", "dP": "m"}
    }
    
    json_thruster = {
        "model_type": "Generalized Logistic (Richards) Thruster Curve Model",
        "pos": {"A": A_POS, "K": K_POS, "B": B_POS, "M": M_POS, "v": V_POS, "C": C_POS},
        "neg": {"A": A_NEG, "K": K_NEG, "B": B_NEG, "M": M_NEG, "v": V_NEG, "C": C_NEG},
        "limits": {"max_force_fwd": MAX_FORCE_FWD, "max_force_rev": MAX_FORCE_REV},
        "units": {"A": "N", "K": "N", "B": "1/cmd", "M": "cmd", "v": "dimensionless", "C": "dimensionless", "max_force_fwd": "N", "max_force_rev": "N"}
    }
    
    json_6_path = os.path.join(script_dir, 'model_6param_30Hz.json')
    json_5_path = os.path.join(script_dir, 'model_5param_30Hz.json')
    json_thruster_path = os.path.join(script_dir, 'thruster_richards_params.json')
    
    with open(json_6_path, 'w') as f:
        json.dump(json_6param, f, indent=4)
    with open(json_5_path, 'w') as f:
        json.dump(json_5param, f, indent=4)
    with open(json_thruster_path, 'w') as f:
        json.dump(json_thruster, f, indent=4)

if __name__ == '__main__':
    main()
