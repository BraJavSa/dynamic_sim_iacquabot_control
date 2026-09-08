"""
identify_plugin_model.py
=========================

Identifica los 9 parametros hidrodinamicos EXACTAMENTE como los usa
vrx::SimpleHydrodynamics (masa anadida sumada, Coriolis propio del
simulador), a partir de un CSV de comandos de los propulsores.

Flujo:
  1. Lee experiment_cmd_30Hz_simple.csv (columnas: cmd_left, cmd_right).
  2. Convierte comandos -> fuerzas (Tu, Tr) con el modelo de motores T200.
  3. Si el CSV NO trae columnas u,v,r medidas (dato real de Gazebo/VRX),
     genera una trayectoria sintetica integrando el modelo EXACTO del
     plugin con unos parametros "verdaderos" de referencia (TRUE_PARAMS,
     derivados de la tabla zeta que ya identificaste). Esto sirve para
     validar que el identificador recupera los parametros correctos
     antes de usarlo con datos reales.
     Si el CSV SI trae u,v,r (log real), esas columnas se usan tal cual.
  4. Guarda el dataset usado (comandos + fuerzas + u,v,r) en
     dataset_usado.csv.
  5. Identifica los 9 parametros del plugin por minimos cuadrados
     lineales (el modelo es lineal en los parametros dados u,v,r y sus
     derivadas).
  6. Simula el modelo identificado con las mismas fuerzas Tu,Tr y
     compara contra los datos "reales" (grafica real_vs_identificado.png).
  7. Imprime los parametros finales listos para pegar en el plugin.

Requiere: numpy, scipy, pandas, matplotlib (todas estandar).
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

# ---------------------------------------------------------------------------
# CONFIGURACION
# ---------------------------------------------------------------------------
CSV_PATH = "experiment_cmd_30Hz_simple.csv"
DT = 1.0 / 30.0  # 30 Hz

# Masa e inercia rigidas (conocidas, fijas)
M_RIGID = 23.344          # kg
IZ_RIGID = 2.923312       # kg*m^2

# Parametros "verdaderos" de referencia, derivados de tu tabla identificada
# (zeta1 = m - X_udot, zeta2 = m - Y_vdot, zeta3 = Iz - N_rdot).
# SOLO se usan como ground-truth para generar datos sinteticos cuando el
# CSV no trae u,v,r medidos. Si tienes datos reales, esto se ignora.
TRUE_PARAMS = dict(
    X_udot=M_RIGID - 24.77,
    Y_vdot=M_RIGID - 30.28,
    N_rdot=IZ_RIGID - 4.93,
    Xu=13.91, Xuu=14.92,
    Yv=49.04, Yvv=1.01,
    Nr=6.03, Nrr=2.91,
)

# ---------------------------------------------------------------------------
# MODELO DE PROPULSORES (igual al tuyo, vectorizado)
# ---------------------------------------------------------------------------
MOTOR_TEMPLATE = {
    'order': ['FR', 'FL', 'BR', 'BL'],
    'T200': {
        'pos': {'A': 1e-06, 'K': 40.0209, 'B': 2.6249, 'v': 0.1615, 'C': 0.9432, 'M': 1e-05},
        'neg': {'A': -31.499, 'K': -1e-05, 'B': 3.6986, 'v': 0.3264, 'C': 0.9713, 'M': -1.0},
    },
    'max_fwd': 36.3827,
    'max_rev': -28.4393,
    'positions_yx': [[-0.29, 0.60], [0.29, 0.60], [-0.29, -0.15], [0.29, -0.15]],
    'angles_deg': [0.0, 0.0, 0.0, 0.0],
}


def branch_thrust(cmd, p):
    return p['A'] + (p['K'] - p['A']) / (p['C'] + np.exp(-p['B'] * (cmd - p['M']))) ** (1.0 / p['v'])


def _get_thrust(cmd, motor):
    p_pos, p_neg = motor['T200']['pos'], motor['T200']['neg']
    t = np.zeros_like(cmd, dtype=float)
    pos, neg = cmd > 0.01, cmd < -0.01
    if np.any(pos):
        t[pos] = branch_thrust(cmd[pos], p_pos)
    if np.any(neg):
        t[neg] = branch_thrust(cmd[neg], p_neg)
    return np.clip(t, motor['max_rev'], motor['max_fwd'])


def cmd_to_thrust_forces(cmd_left, cmd_right, motor=None):
    """Identico a tu funcion original: devuelve Tu [N] y Tr [N*m]."""
    motor = motor or MOTOR_TEMPLATE
    cmd_left = np.asarray(cmd_left, dtype=float)
    cmd_right = np.asarray(cmd_right, dtype=float)

    T_FR = _get_thrust(cmd_right, motor)
    T_FL = _get_thrust(cmd_left, motor)
    T_BR = _get_thrust(cmd_right, motor)
    T_BL = _get_thrust(cmd_left, motor)

    y_FR, y_FL, y_BR, y_BL = [p[0] for p in motor['positions_yx']]

    Tu = T_FR + T_FL + T_BR + T_BL
    Tr = -(y_FR * T_FR + y_FL * T_FL + y_BR * T_BR + y_BL * T_BL)
    return Tu, Tr


# ---------------------------------------------------------------------------
# MODELO EXACTO DEL PLUGIN (ecuaciones de la seccion 3 de tu documento)
# ---------------------------------------------------------------------------
def _derivs(state, Tu, Tr, p, m, Iz):
    u, v, r = state
    a1 = m + p['X_udot']
    a2 = m + p['Y_vdot']
    a3 = Iz + p['N_rdot']
    udot = (Tu + a2 * v * r - p['Xu'] * u - p['Xuu'] * abs(u) * u) / a1
    vdot = (-(m - p['X_udot']) * u * r - p['Yv'] * v - p['Yvv'] * abs(v) * v) / a2
    rdot = (Tr + (p['X_udot'] + p['Y_vdot']) * u * v - p['Nr'] * r - p['Nrr'] * abs(r) * r) / a3
    return np.array([udot, vdot, rdot])


def simulate_plugin(Tu, Tr, dt, p, m=M_RIGID, Iz=IZ_RIGID, state0=(0.0, 0.0, 0.0)):
    """Integra el modelo EXACTO del plugin con RK4 (Tu, Tr constantes por paso)."""
    N = len(Tu)
    U = np.zeros(N)
    V = np.zeros(N)
    R = np.zeros(N)
    s = np.array(state0, dtype=float)
    U[0], V[0], R[0] = s
    for k in range(N - 1):
        f1 = _derivs(s, Tu[k], Tr[k], p, m, Iz)
        f2 = _derivs(s + 0.5 * dt * f1, Tu[k], Tr[k], p, m, Iz)
        f3 = _derivs(s + 0.5 * dt * f2, Tu[k], Tr[k], p, m, Iz)
        f4 = _derivs(s + dt * f3, Tu[k], Tr[k], p, m, Iz)
        s = s + (dt / 6.0) * (f1 + 2 * f2 + 2 * f3 + f4)
        U[k + 1], V[k + 1], R[k + 1] = s
    return U, V, R


# ---------------------------------------------------------------------------
# IDENTIFICACION (minimos cuadrados lineales, modelo exacto del plugin)
# ---------------------------------------------------------------------------
def estimate_derivatives(x, dt):
    """Derivada con filtro Savitzky-Golay (suaviza y deriva a la vez)."""
    n = len(x)
    win = min(31, n if n % 2 == 1 else n - 1)
    win = max(5, win)
    poly = 3 if win > 3 else 2
    return savgol_filter(x, window_length=win, polyorder=poly, deriv=1, delta=dt)


def identify_plugin(u, v, r, Tu, Tr, dt, m=M_RIGID, Iz=IZ_RIGID):
    """
    Resuelve, por minimos cuadrados, el sistema LINEAL (exacto) que se
    obtiene al escribir las 3 ecuaciones del plugin en funcion de
    theta = [a1, a2, a3, Xu, Xuu, Yv, Yvv, Nr, Nrr], con
    a1 = m+X_udot, a2 = m+Y_vdot, a3 = Iz+N_rdot.
    """
    udot = estimate_derivatives(u, dt)
    vdot = estimate_derivatives(v, dt)
    rdot = estimate_derivatives(r, dt)

    N = len(u)
    A = np.zeros((3 * N, 9))
    b = np.zeros(3 * N)

    # surge: a1*udot - a2*(v r) + Xu u + Xuu|u|u = Tu
    A[0:N, 0] = udot
    A[0:N, 1] = -(v * r)
    A[0:N, 3] = u
    A[0:N, 4] = np.abs(u) * u
    b[0:N] = Tu

    # sway: a2*vdot - a1*(u r) + Yv v + Yvv|v|v = -2m*(u r)
    A[N:2 * N, 1] = vdot
    A[N:2 * N, 0] = -(u * r)
    A[N:2 * N, 5] = v
    A[N:2 * N, 6] = np.abs(v) * v
    b[N:2 * N] = -2.0 * m * (u * r)

    # yaw: a3*rdot - (a1+a2)*(u v) + Nr r + Nrr|r|r = Tr - 2m*(u v)
    A[2 * N:3 * N, 2] = rdot
    A[2 * N:3 * N, 0] = -(u * v)
    A[2 * N:3 * N, 1] = -(u * v)
    A[2 * N:3 * N, 7] = r
    A[2 * N:3 * N, 8] = np.abs(r) * r
    b[2 * N:3 * N] = Tr - 2.0 * m * (u * v)

    theta, *_ = np.linalg.lstsq(A, b, rcond=None)
    a1, a2, a3, Xu, Xuu, Yv, Yvv, Nr, Nrr = theta

    return dict(
        X_udot=a1 - m, Y_vdot=a2 - m, N_rdot=a3 - Iz,
        Xu=Xu, Xuu=Xuu, Yv=Yv, Yvv=Yvv, Nr=Nr, Nrr=Nrr,
    )


def print_params(title, p, m=M_RIGID, Iz=IZ_RIGID):
    print(f"\n{title}")
    print("-" * len(title))
    for k in ['X_udot', 'Y_vdot', 'N_rdot', 'Xu', 'Xuu', 'Yv', 'Yvv', 'Nr', 'Nrr']:
        print(f"  {k:8s} = {p[k]: .6f}")
    print(f"  (zeta1 = m - X_udot = {m - p['X_udot']:.4f} kg)")
    print(f"  (zeta2 = m - Y_vdot = {m - p['Y_vdot']:.4f} kg)")
    print(f"  (zeta3 = Iz - N_rdot = {Iz - p['N_rdot']:.4f} kg*m^2)")


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def main():
    df = pd.read_csv(CSV_PATH)
    if not {'cmd_left', 'cmd_right'}.issubset(df.columns):
        raise ValueError(f"{CSV_PATH} debe tener columnas cmd_left y cmd_right")

    Tu, Tr = cmd_to_thrust_forces(df['cmd_left'].values, df['cmd_right'].values)
    t = np.arange(len(Tu)) * DT

    tiene_datos_reales = {'u', 'v', 'r'}.issubset(df.columns)

    if tiene_datos_reales:
        print("Usando u, v, r medidos (encontrados en el CSV) como datos reales.")
        u_real, v_real, r_real = df['u'].values, df['v'].values, df['r'].values
    else:
        print("El CSV no trae u,v,r medidos.")
        print("Generando trayectoria de referencia integrando el modelo EXACTO")
        print("del plugin con TRUE_PARAMS (derivados de tu tabla identificada).")
        print("Reemplaza esto por tu log real de Gazebo/VRX cuando lo tengas.")
        u_real, v_real, r_real = simulate_plugin(Tu, Tr, DT, TRUE_PARAMS)

    out = pd.DataFrame({
        't': t, 'cmd_left': df['cmd_left'].values, 'cmd_right': df['cmd_right'].values,
        'Tu': Tu, 'Tr': Tr, 'u': u_real, 'v': v_real, 'r': r_real,
    })
    out.to_csv('dataset_usado.csv', index=False)
    print("Datos guardados en dataset_usado.csv")

    identified = identify_plugin(u_real, v_real, r_real, Tu, Tr, DT)
    print_params("PARAMETROS IDENTIFICADOS (modelo exacto del plugin)", identified)

    if not tiene_datos_reales:
        print_params("PARAMETROS VERDADEROS (referencia usada para sintetizar)", TRUE_PARAMS)

    u_id, v_id, r_id = simulate_plugin(Tu, Tr, DT, identified,
                                        state0=(u_real[0], v_real[0], r_real[0]))

    fig, axs = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    labels = ['u (surge) [m/s]', 'v (sway) [m/s]', 'r (yaw) [rad/s]']
    for ax, real, ident, lab in zip(axs, (u_real, v_real, r_real), (u_id, v_id, r_id), labels):
        ax.plot(t, real, label='real', lw=1.5)
        ax.plot(t, ident, '--', label='identificado', lw=1.5)
        ax.set_ylabel(lab)
        ax.legend()
        ax.grid(True, alpha=0.3)
    axs[-1].set_xlabel('t [s]')
    fig.suptitle('Real vs modelo del plugin identificado')
    fig.tight_layout()
    fig.savefig('real_vs_identificado.png', dpi=150)
    print("\nGrafica guardada en real_vs_identificado.png")


if __name__ == '__main__':
    main()