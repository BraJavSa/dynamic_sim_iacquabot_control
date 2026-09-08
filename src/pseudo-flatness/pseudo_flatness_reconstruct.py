import numpy as np
from usv_params import B_MATRIX, T_MAX, T_MIN, cmd_from_thrust_array, thrust_from_cmd_poly, m11, m22, m33, X_u, X_uu, Y_v, Y_vv, N_r, N_rr
B_ACT = B_MATRIX[[0, 2], :]
B_ACT_PINV = np.linalg.pinv(B_ACT)

def _uvr_from_xy(psi, x_d, y_d):
    c, s = (np.cos(psi), np.sin(psi))
    u = x_d * c + y_d * s
    v = -x_d * s + y_d * c
    return (u, v)
R_DOT_HARD_LIMIT = 5.0
LAM_TIKHONOV = 0.015

def _psi_dot_ode(psi, x_d, y_d, x_dd, y_dd):
    u, v = _uvr_from_xy(psi, x_d, y_d)
    d22 = Y_v + Y_vv * abs(v)
    coef_r = (m22 - m11) / m22 * u
    rhs = -x_dd * np.sin(psi) + y_dd * np.cos(psi) + d22 / m22 * v
    r = coef_r * rhs / (coef_r ** 2 + LAM_TIKHONOV ** 2)
    return float(np.clip(r, -R_DOT_HARD_LIMIT, R_DOT_HARD_LIMIT))

def integrate_psi(t, x_d, y_d, x_dd, y_dd, psi0=None):
    N = len(t)
    psi = np.zeros(N)
    if psi0 is None:
        if np.hypot(x_d[0], y_d[0]) > 0.001:
            psi0 = float(np.arctan2(y_d[0], x_d[0]))
        else:
            psi0 = 0.0
    psi[0] = psi0

    def sample(arr, tt):
        return np.interp(tt, t, arr)
    for k in range(N - 1):
        dt = t[k + 1] - t[k]
        tk = t[k]
        xk, yk, xdk, ydk = (x_d[k], y_d[k], x_dd[k], y_dd[k])
        xk_h = sample(x_d, tk + dt / 2)
        yk_h = sample(y_d, tk + dt / 2)
        xdk_h = sample(x_dd, tk + dt / 2)
        ydk_h = sample(y_dd, tk + dt / 2)
        xk1 = sample(x_d, tk + dt)
        yk1 = sample(y_d, tk + dt)
        xdk1 = sample(x_dd, tk + dt)
        ydk1 = sample(y_dd, tk + dt)
        k1 = _psi_dot_ode(psi[k], xk, yk, xdk, ydk)
        k2 = _psi_dot_ode(psi[k] + dt / 2 * k1, xk_h, yk_h, xdk_h, ydk_h)
        k3 = _psi_dot_ode(psi[k] + dt / 2 * k2, xk_h, yk_h, xdk_h, ydk_h)
        k4 = _psi_dot_ode(psi[k] + dt * k3, xk1, yk1, xdk1, ydk1)
        psi[k + 1] = psi[k] + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
    r = np.array([_psi_dot_ode(psi[k], x_d[k], y_d[k], x_dd[k], y_dd[k]) for k in range(N)])
    return (psi, r)

def reconstruct_state_and_wrench(F, F_d, F_dd, t, psi0=None):
    x, y = (F[:, 0], F[:, 1])
    x_d, y_d = (F_d[:, 0], F_d[:, 1])
    x_dd, y_dd = (F_dd[:, 0], F_dd[:, 1])
    psi, r = integrate_psi(t, x_d, y_d, x_dd, y_dd, psi0=psi0)
    u, v = _uvr_from_xy(psi, x_d, y_d)
    r_dot = np.gradient(r, t)
    u_dot = np.gradient(u, t)
    d11 = X_u + X_uu * np.abs(u)
    d33 = N_r + N_rr * np.abs(r)
    tau_u = m11 * u_dot - m22 * v * r + d11 * u
    tau_r = m33 * r_dot - (m11 - m22) * u * v + d33 * r
    eta = np.column_stack([x, y, psi])
    nu = np.column_stack([u, v, r])
    tau_actuado = np.column_stack([tau_u, tau_r])
    return (eta, nu, tau_actuado)

def wrench_to_thrusters(tau_actuado):
    return tau_actuado @ B_ACT_PINV.T

def check_saturation(T, t_max=T_MAX, t_min=T_MIN, r=None, r_soft_limit=0.85 * R_DOT_HARD_LIMIT):
    over_max = T > t_max
    under_min = T < t_min
    violated = bool(np.any(over_max) or np.any(under_min))
    r_near_limit = False
    r_max_val = None
    if r is not None:
        r_max_val = float(np.max(np.abs(r)))
        r_near_limit = r_max_val > r_soft_limit
        violated = violated or r_near_limit
    summary = {'violated': violated, 'max_thrust_demanded': float(T.max()), 'min_thrust_demanded': float(T.min()), 'n_samples_over_max': int(np.sum(over_max)), 'n_samples_under_min': int(np.sum(under_min)), 't_max_limit': t_max, 't_min_limit': t_min, 'r_max': r_max_val, 'r_soft_limit': r_soft_limit, 'r_near_limit': r_near_limit}
    return (summary, over_max | under_min)

def thrust_to_cmd(T):
    return cmd_from_thrust_array(T, c_min=-1.0, c_max=1.0)

def cmd_to_actual_thrust(c):
    return thrust_from_cmd_poly(c)

def _real_9param_derivative(u, v, r, Tu, Tr):
    du = (Tu + m22 * v * r - X_u * u - X_uu * abs(u) * u) / m11
    dv = (-m11 * u * r - Y_v * v - Y_vv * abs(v) * v) / m22
    dr = (Tr - (m22 - m11) * u * v - N_r * r - N_rr * abs(r) * r) / m33
    return (du, dv, dr)

def rk4_step_9param(u, v, r, x, y, psi, Tu, Tr, dt):
    du1, dv1, dr1 = _real_9param_derivative(u, v, r, Tu, Tr)
    dx1 = u * np.cos(psi) - v * np.sin(psi)
    dy1 = u * np.sin(psi) + v * np.cos(psi)
    dpsi1 = r
    du2, dv2, dr2 = _real_9param_derivative(u + 0.5 * dt * du1, v + 0.5 * dt * dv1, r + 0.5 * dt * dr1, Tu, Tr)
    dx2 = (u + 0.5 * dt * du1) * np.cos(psi + 0.5 * dt * dpsi1) - (v + 0.5 * dt * dv1) * np.sin(psi + 0.5 * dt * dpsi1)
    dy2 = (u + 0.5 * dt * du1) * np.sin(psi + 0.5 * dt * dpsi1) + (v + 0.5 * dt * dv1) * np.cos(psi + 0.5 * dt * dpsi1)
    dpsi2 = r + 0.5 * dt * dr1
    du3, dv3, dr3 = _real_9param_derivative(u + 0.5 * dt * du2, v + 0.5 * dt * dv2, r + 0.5 * dt * dr2, Tu, Tr)
    dx3 = (u + 0.5 * dt * du2) * np.cos(psi + 0.5 * dt * dpsi2) - (v + 0.5 * dt * dv2) * np.sin(psi + 0.5 * dt * dpsi2)
    dy3 = (u + 0.5 * dt * du2) * np.sin(psi + 0.5 * dt * dpsi2) + (v + 0.5 * dt * dv2) * np.cos(psi + 0.5 * dt * dpsi2)
    dpsi3 = r + 0.5 * dt * dr2
    du4, dv4, dr4 = _real_9param_derivative(u + dt * du3, v + dt * dv3, r + dt * dr3, Tu, Tr)
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

def simulate_open_loop_plant(result, dt=None):
    t = result['t']
    tau = result['tau']
    eta = result['eta']
    nu = result['nu']
    n = len(t)
    x = np.zeros(n)
    y = np.zeros(n)
    psi = np.zeros(n)
    u = np.zeros(n)
    v = np.zeros(n)
    r = np.zeros(n)
    x[0], y[0], psi[0] = (eta[0, 0], eta[0, 1], eta[0, 2])
    u[0], v[0], r[0] = (nu[0, 0], nu[0, 1], nu[0, 2])
    for k in range(n - 1):
        dt_k = t[k + 1] - t[k]
        Tu, Tr = (tau[k, 0], tau[k, 1])
        u[k + 1], v[k + 1], r[k + 1], x[k + 1], y[k + 1], psi[k + 1] = rk4_step_9param(u[k], v[k], r[k], x[k], y[k], psi[k], Tu, Tr, dt_k)
    eta_real = np.column_stack([x, y, psi])
    nu_real = np.column_stack([u, v, r])
    return (eta_real, nu_real)
