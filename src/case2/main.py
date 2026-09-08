import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from min_jerk_qp import MinJerkTrajectoryFlat
from flatness_reconstruct import reconstruct_state_and_wrench, wrench_to_thrusters, check_saturation, thrust_to_cmd, cmd_to_actual_thrust
from usv_params import T_MAX, T_MIN, DT_EXPERIMENT

def build_trajectory(waypoints_xy, times):
    waypoints_xy = np.asarray(waypoints_xy, dtype=float)[:, :2]
    return MinJerkTrajectoryFlat(waypoints_xy, times)

def evaluate_dynamics(traj, dt=DT_EXPERIMENT):
    ts, F, F_d, F_dd, F_ddd = traj.sample(dt=dt)
    eta, nu, tau_actuado = reconstruct_state_and_wrench(F, F_d, F_dd, ts)
    T = wrench_to_thrusters(tau_actuado)
    cmd = thrust_to_cmd(T)
    T_actual = cmd_to_actual_thrust(cmd)
    return dict(t=ts, eta=eta, F=F, F_d=F_d, F_dd=F_dd, F_ddd=F_ddd, nu=nu, tau=tau_actuado, T=T, cmd=cmd, T_actual=T_actual)

def time_scale_waypoints(times, factor):
    t0 = times[0]
    return t0 + (np.asarray(times) - t0) * factor

def solve_with_saturation_check(waypoints_xy, times, dt=DT_EXPERIMENT, max_iter=15, growth=1.15, verbose=True):
    current_times = np.array(times, dtype=float)
    for it in range(max_iter):
        traj = build_trajectory(waypoints_xy, current_times)
        result = evaluate_dynamics(traj, dt=dt)
        summary, mask = check_saturation(result['T'])
        if verbose:
            print(f'[iter {it}] T_total={current_times[-1] - current_times[0]:.2f}s max_T={summary['max_thrust_demanded']:.2f}N min_T={summary['min_thrust_demanded']:.2f}N sat_violado={summary['violated']}')
        if not summary['violated']:
            return (traj, result, summary, current_times)
        current_times = time_scale_waypoints(current_times, growth)
    raise RuntimeError('No se logró cumplir la saturación de empuje tras max_iter escalados de tiempo. Esto sugiere que la geometría de la trayectoria (curvas muy cerradas / cambios de rumbo muy bruscos) exige fuerzas físicamente inalcanzables con cualquier tiempo total razonable -- rediseña los waypoints (curvas más suaves) en vez de seguir escalando t.')

def plot_results(result, times, traj, waypoints_xy, T_max=T_MAX, T_min=T_MIN, outpath=None):

    if outpath is None:
        outpath = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'usv_trajectory.png')
    out_dir = os.path.dirname(os.path.abspath(outpath))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    waypoints_xy = np.asarray(waypoints_xy, dtype=float)[:, :2]
    t_wp7 = times[7]
    mask = result['t'] <= t_wp7
    t = result['t'][mask]
    eta = result['eta'][mask]
    tau = result['tau'][mask]
    T = result['T'][mask]
    cmd = result['cmd'][mask]
    T_actual = result['T_actual'][mask]
    waypoints_show = waypoints_xy[:8]
    fig, axs = plt.subplots(4, 2, figsize=(13, 16))
    ax = axs[0, 0]
    ax.plot(eta[:, 0], eta[:, 1], 'b-', lw=2, label='Trayectoria x,y(t)', zorder=2)
    ax.scatter(waypoints_show[:, 0], waypoints_show[:, 1], c='r', s=60, zorder=5, label='Waypoints')
    for i, (wx, wy) in enumerate(waypoints_show):
        ax.annotate(str(i), (wx, wy), textcoords='offset points', xytext=(6, 6), fontsize=9, color='darkred')
    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')
    ax.set_title('Trayectoria óptima (minimum-jerk) en el plano\n(psi se deriva, no se planifica)')
    ax.axis('equal')
    ax.legend(loc='best', fontsize=8)
    ax.grid(True)
    ax = axs[0, 1]
    ax.plot(t, np.degrees(eta[:, 2]), 'g-')
    ax.set_xlabel('t [s]')
    ax.set_ylabel('psi [deg]')
    ax.set_title('Rumbo (yaw) reconstruido por planitud vs tiempo')
    ax.grid(True)
    ax = axs[1, 0]
    nu = result['nu'][mask]
    ax.plot(t, nu[:, 0], label='u (surge) [m/s]')
    ax.plot(t, nu[:, 1], label='v (sway) [m/s]')
    ax.plot(t, nu[:, 2], label='r (yaw rate) [rad/s]')
    ax.set_xlabel('t [s]')
    ax.set_title('Velocidades en marco del cuerpo (reconstruidas)')
    ax.legend()
    ax.grid(True)
    ax = axs[1, 1]
    ax.plot(t, tau[:, 0], label='$\\tau_u$ (surge) [N]')
    ax.plot(t, tau[:, 1], label='$\\tau_r$ (yaw) [N·m]')
    ax.axhline(0, color='gray', lw=0.8)
    ax.set_xlabel('t [s]')
    ax.set_title('Fuerzas generalizadas actuadas (Ecs. 4-5 del paper)')
    ax.legend()
    ax.grid(True)
    ax = axs[2, 0]
    labels = ['FR', 'FL', 'BR', 'BL']
    for i, lab in enumerate(labels):
        ax.plot(t, T[:, i], label=f'T_{lab}')
    ax.axhline(T_max, color='k', ls='--', lw=1, label='límites')
    ax.axhline(T_min, color='k', ls='--', lw=1)
    ax.set_xlabel('t [s]')
    ax.set_ylabel('Empuje [N]')
    ax.set_title('Empuje individual por propulsor')
    ax.legend(ncol=2, fontsize=8)
    ax.grid(True)
    ax = axs[2, 1]
    F_ddd = result['F_ddd'][mask]
    ax.plot(t, F_ddd[:, 0], label='jerk x')
    ax.plot(t, F_ddd[:, 1], label='jerk y')
    ax.set_xlabel('t [s]')
    ax.set_title('Jerk de la salida plana F=[x,y] (funcional minimizado)')
    ax.legend()
    ax.grid(True)
    ax = axs[3, 0]
    for i, lab in enumerate(labels):
        ax.plot(t, cmd[:, i], label=f'c_{lab}')
    ax.axhline(1.0, color='k', ls='--', lw=1, label='límites [-1,1]')
    ax.axhline(-1.0, color='k', ls='--', lw=1)
    ax.set_xlabel('t [s]')
    ax.set_ylabel('Comando normalizado')
    ax.set_title('Comando real por propulsor (curva grado 5 invertida)')
    ax.legend(ncol=2, fontsize=8)
    ax.grid(True)
    ax = axs[3, 1]
    ax.plot(t, T[:, 0], 'b-', lw=1.8, label='T deseado (FR)')
    ax.plot(t, T_actual[:, 0], 'r--', lw=1.8, label='T entregado (FR, tras curva)')
    ax.set_xlabel('t [s]')
    ax.set_ylabel('Empuje [N]')
    ax.set_title('Empuje deseado vs. entregado (ejemplo propulsor FR)')
    ax.legend(fontsize=8)
    ax.grid(True)
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    print(f'Figura guardada en: {outpath}')
    return outpath

if __name__ == '__main__':
    waypoints_xy = np.array([[0.0, 0.0], [8.0, 0.0], [14.0, 5.0], [14.0, 13.0], [20.0, 17.0], [28.0, 17.0], [32.0, 10.0], [26.0, 4.0], [20.0, 1.5]])
    times_initial = np.array([0.0, 7.0, 14.0, 20.0, 27.0, 33.0, 40.0, 48.0, 55.0])
    traj, result, summary, final_times = solve_with_saturation_check(waypoints_xy, times_initial, dt=DT_EXPERIMENT, verbose=True)
    print('\n--- Resumen final ---')
    print(f'Tiempo total de la maniobra: {final_times[-1] - final_times[0]:.2f} s (original: {times_initial[-1] - times_initial[0]:.2f} s)')
    print(f'Empuje máximo demandado: {summary['max_thrust_demanded']:.2f} N (límite: {T_MAX} N)')
    print(f'Empuje mínimo demandado: {summary['min_thrust_demanded']:.2f} N (límite: {T_MIN} N)')
    cmd = result['cmd']
    T_actual = result['T_actual']
    T_desired = result['T']
    cmd_saturated = np.sum(np.abs(cmd) >= 0.999)
    thrust_error = np.abs(T_actual - T_desired)
    print(f'\nComando saturado (|c|>=0.999) en {cmd_saturated} muestras de {cmd.size} totales')
    print(f'Error máx. entre empuje deseado y entregado (por curva real): {thrust_error.max():.3f} N')
    print(f'Error RMS entre empuje deseado y entregado: {np.sqrt(np.mean(thrust_error ** 2)):.3f} N')
    plot_results(result, final_times, traj, waypoints_xy)

