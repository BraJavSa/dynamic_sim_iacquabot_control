import numpy as np
from usv_params import m_5, m33_5, Xu_5, Yv_5, Nr_5, dP_5, T_MAX, T_MIN, cmd_from_thrust_array, thrust_from_cmd_poly

def reconstruct_flatness_h2(pos, vel, acc, jerk, t):
    """
    Hypothesis 2 Flatness Reconstruction (m11 = m22 = m).
    Equations from flatness.tex Eq. (46)-(50):
    A = ddy + (Yv/m) * dy
    B = ddx + (Yv/m) * dx
    psi = atan2(A, B)
    """
    x, y = pos[:, 0], pos[:, 1]
    dx, dy = vel[:, 0], vel[:, 1]
    ddx, ddy = acc[:, 0], acc[:, 1]
    
    A = ddy + (Yv_5 / m_5) * dy
    B = ddx + (Yv_5 / m_5) * dx
    
    mag = np.hypot(A, B)
    psi_raw = np.zeros_like(x)
    
    for i in range(len(x)):
        if mag[i] > 0.05:
            psi_raw[i] = np.arctan2(A[i], B[i])
        else:
            speed = np.hypot(dx[i], dy[i])
            if speed > 0.05:
                psi_raw[i] = np.arctan2(dy[i], dx[i])
            elif i > 0:
                psi_raw[i] = psi_raw[i-1]
            else:
                psi_raw[i] = 0.0
                
    psi = np.unwrap(psi_raw)
    
    # Kinematic velocities
    u = dx * np.cos(psi) + dy * np.sin(psi)
    v = -dx * np.sin(psi) + dy * np.cos(psi)
    
    # Yaw rate r and r_dot via numerical gradient
    r = np.gradient(psi, t)
    dr = np.gradient(r, t)
    
    # Flatness control forces (Hypothesis 2)
    tau_u_raw = m_5 * (ddx * np.cos(psi) + ddy * np.sin(psi)) + Xu_5 * u
    tau_r_raw = m33_5 * dr + Nr_5 * r
    
    T1_raw = 0.5 * (tau_u_raw + tau_r_raw / dP_5)
    T2_raw = 0.5 * (tau_u_raw - tau_r_raw / dP_5)
    
    tau_u_max = 2.0 * T_MAX
    tau_u_min = 2.0 * T_MIN
    tau_r_max = (T_MAX - T_MIN) * dP_5
    tau_r_min = -tau_r_max
    
    tau_u = np.clip(tau_u_raw, tau_u_min, tau_u_max)
    tau_r = np.clip(tau_r_raw, tau_r_min, tau_r_max)
    
    # Thruster force allocation
    T1_dem = np.clip(T1_raw, T_MIN, T_MAX)
    T2_dem = np.clip(T2_raw, T_MIN, T_MAX)
    
    # Convert demanded thrusts to motor commands using 5th-deg polynomial
    cmd_1 = cmd_from_thrust_array(T1_dem)
    cmd_2 = cmd_from_thrust_array(T2_dem)
    
    # Actual forces delivered by thrusters under polynomial model
    T1_act = thrust_from_cmd_poly(cmd_1)
    T2_act = thrust_from_cmd_poly(cmd_2)
    
    tau_u_act = T1_act + T2_act
    tau_r_act = (T1_act - T2_act) * dP_5
    
    eta = np.column_stack([x, y, psi])
    nu = np.column_stack([u, v, r])
    tau_plan = np.column_stack([tau_u, tau_r])
    tau_act = np.column_stack([tau_u_act, tau_r_act])
    cmds = np.column_stack([cmd_1, cmd_2])
    
    return {
        'eta': eta,
        'nu': nu,
        'tau_plan': tau_plan,
        'tau_act': tau_act,
        'tau_u_raw': tau_u_raw,
        'tau_r_raw': tau_r_raw,
        'cmds': cmds,
        'T1_raw': T1_raw,
        'T2_raw': T2_raw,
        'T_plan': np.column_stack([T1_dem, T2_dem]),
        'T_act': np.column_stack([T1_act, T2_act])
    }
