import numpy as np
THRUSTER_POS = {'FR': np.array([1.6, -1.027135]), 'FL': np.array([1.6, 1.027135]), 'BR': np.array([-2.373776, -1.027135]), 'BL': np.array([-2.373776, 1.027135])}
B_MATRIX = np.array([[1.0, 1.0, 1.0, 1.0], [0.0, 0.0, 0.0, 0.0], [1.027135, -1.027135, 1.027135, -1.027135]])
T_MAX = 36.3827
T_MIN = -28.4393
SAMPLE_RATE_HZ = 30.0
DT_EXPERIMENT = 1.0 / SAMPLE_RATE_HZ
m11 = 90.03238
m22 = 91.066933
m33 = 334.867746
M_MATRIX = np.diag([m11, m22, m33])
M_INV = np.linalg.inv(M_MATRIX)
X_u, X_uu = (23.237495, 38.22715)
Y_v, Y_vv = (26.016985, 56.929867)
N_r, N_rr = (79.463013, 30.613898)

def coriolis_matrix(nu):
    u, v, r = nu
    C = np.array([[0.0, 0.0, -m22 * v], [0.0, 0.0, m11 * u], [m22 * v, -m11 * u, 0.0]])
    return C

def damping_matrix(nu):
    u, v, r = nu
    D = np.diag([X_u + X_uu * abs(u), Y_v + Y_vv * abs(v), N_r + N_rr * abs(r)])
    return D

def rotation_matrix(psi):
    c, s = (np.cos(psi), np.sin(psi))
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])

def S_matrix(r):
    return np.array([[0.0, -r, 0.0], [r, 0.0, 0.0], [0.0, 0.0, 0.0]])
_A_pos, _K_pos, _B_pos, _v_pos, _C_pos, _M_pos = (1e-06, 40.0209, 2.6249, 0.1615, 0.9432, 1e-05)
_A_neg, _K_neg, _B_neg, _v_neg, _C_neg, _M_neg = (-31.499, -1e-05, 3.6986, 0.3264, 0.9713, -1.0)

def thrust_curve_exact(c):
    c = np.atleast_1d(np.asarray(c, dtype=float))
    T = np.zeros_like(c)
    mask_pos = c > 0.01
    mask_neg = c < -0.01
    cp = c[mask_pos]
    T[mask_pos] = _A_pos + (_K_pos - _A_pos) / (_C_pos + np.exp(-_B_pos * (cp - _M_pos))) ** (1.0 / _v_pos)
    cn = c[mask_neg]
    T[mask_neg] = _A_neg + (_K_neg - _A_neg) / (_C_neg + np.exp(-_B_neg * (cn - _M_neg))) ** (1.0 / _v_neg)
    return T
THRUST_POLY_DEGREE = 5
_c_fit = np.linspace(-1.0, 1.0, 3000)
_T_fit = thrust_curve_exact(_c_fit)
THRUST_POLY_COEFFS = np.polyfit(_c_fit, _T_fit, THRUST_POLY_DEGREE)

def thrust_from_cmd_poly(c):
    return np.polyval(THRUST_POLY_COEFFS, c)

def cmd_from_thrust_poly(T_target, c_min=-1.0, c_max=1.0):
    p = THRUST_POLY_COEFFS.copy()
    p[-1] -= T_target
    roots = np.roots(p)
    real_roots = roots[np.abs(roots.imag) < 1e-06].real
    valid = real_roots[(real_roots >= c_min - 1e-06) & (real_roots <= c_max + 1e-06)]
    if len(valid) == 0:
        return c_max if T_target > 0 else c_min
    if len(valid) == 1:
        return float(valid[0])
    if T_target == 0:
        return float(valid[np.argmin(np.abs(valid))])
    target_side = np.sign(T_target)
    candidates = valid[np.sign(valid) == target_side]
    if len(candidates) > 0:
        return float(candidates[np.argmin(np.abs(candidates))])
    return float(valid[np.argmin(np.abs(valid))])

def cmd_from_thrust_array(T_array, c_min=-1.0, c_max=1.0):
    T_flat = np.asarray(T_array).ravel()
    c_flat = np.array([cmd_from_thrust_poly(Tv, c_min, c_max) for Tv in T_flat])
    return c_flat.reshape(np.asarray(T_array).shape)
