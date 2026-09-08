import numpy as np

class MinJerkTrajectory1D:
    POLY_ORDER = 5

    def __init__(self, waypoints, times):
        self.waypoints = np.asarray(waypoints, dtype=float)
        self.times = np.asarray(times, dtype=float)
        self.n_seg = len(waypoints) - 1
        assert self.n_seg >= 1, 'Se necesitan al menos 2 waypoints'
        assert len(times) == len(waypoints)
        self.coeffs = self._solve()

    def _poly_basis(self, tau, order=0):
        n = self.POLY_ORDER
        basis = np.zeros(n + 1)
        for i in range(n + 1):
            if i < order:
                continue
            coeff = 1.0
            for k in range(order):
                coeff *= i - k
            basis[i] = coeff * tau ** (i - order) if i - order >= 0 else 0.0
        return basis

    def _solve(self):
        n_seg = self.n_seg
        n_coef = self.POLY_ORDER + 1
        n_vars = n_seg * n_coef
        H = np.zeros((n_vars, n_vars))
        for k in range(n_seg):
            T = self.times[k + 1] - self.times[k]
            Hk = self._segment_cost_matrix(T)
            H[k * n_coef:(k + 1) * n_coef, k * n_coef:(k + 1) * n_coef] = Hk
        A_rows = []
        b_vals = []
        for k in range(n_seg):
            T = self.times[k + 1] - self.times[k]
            row = np.zeros(n_vars)
            row[k * n_coef:(k + 1) * n_coef] = self._poly_basis(0.0, order=0)
            A_rows.append(row)
            b_vals.append(self.waypoints[k])
            row = np.zeros(n_vars)
            row[k * n_coef:(k + 1) * n_coef] = self._poly_basis(T, order=0)
            A_rows.append(row)
            b_vals.append(self.waypoints[k + 1])
        for order in (1, 2):
            row = np.zeros(n_vars)
            row[0:n_coef] = self._poly_basis(0.0, order=order)
            A_rows.append(row)
            b_vals.append(0.0)
            T_last = self.times[-1] - self.times[-2]
            row = np.zeros(n_vars)
            row[(n_seg - 1) * n_coef:n_seg * n_coef] = self._poly_basis(T_last, order=order)
            A_rows.append(row)
            b_vals.append(0.0)
        for k in range(n_seg - 1):
            T = self.times[k + 1] - self.times[k]
            for order in (1, 2):
                row = np.zeros(n_vars)
                row[k * n_coef:(k + 1) * n_coef] = self._poly_basis(T, order=order)
                row[(k + 1) * n_coef:(k + 2) * n_coef] = -self._poly_basis(0.0, order=order)
                A_rows.append(row)
                b_vals.append(0.0)
        A = np.array(A_rows)
        b = np.array(b_vals)
        m = A.shape[0]
        KKT = np.zeros((n_vars + m, n_vars + m))
        KKT[:n_vars, :n_vars] = H + 1e-09 * np.eye(n_vars)
        KKT[:n_vars, n_vars:] = A.T
        KKT[n_vars:, :n_vars] = A
        rhs = np.concatenate([np.zeros(n_vars), b])
        sol = np.linalg.lstsq(KKT, rhs, rcond=None)[0]
        c = sol[:n_vars]
        return c.reshape(n_seg, n_coef)

    def _segment_cost_matrix(self, T):
        n = self.POLY_ORDER + 1
        H = np.zeros((n, n))
        for i in range(3, n):
            for j in range(3, n):
                ci = i * (i - 1) * (i - 2)
                cj = j * (j - 1) * (j - 2)
                power = i - 3 + (j - 3) + 1
                H[i, j] = ci * cj * T ** power / power
        return H

    def eval(self, t, order=0):
        k = np.searchsorted(self.times, t, side='right') - 1
        k = min(max(k, 0), self.n_seg - 1)
        tau = t - self.times[k]
        basis = self._poly_basis(tau, order=order)
        return float(self.coeffs[k] @ basis)

def unwrap_angle_sequence(angles):
    unwrapped = np.zeros_like(angles, dtype=float)
    unwrapped[0] = angles[0]
    for i in range(1, len(angles)):
        diff = (angles[i] - unwrapped[i - 1] + np.pi) % (2.0 * np.pi) - np.pi
        unwrapped[i] = unwrapped[i - 1] + diff
    return unwrapped

class MinJerkTrajectory3D:

    def __init__(self, waypoints_xyz, times):
        waypoints_xyz = np.asarray(waypoints_xyz, dtype=float)
        self.times = np.asarray(times, dtype=float)
        self.traj_x = MinJerkTrajectory1D(waypoints_xyz[:, 0], times)
        self.traj_y = MinJerkTrajectory1D(waypoints_xyz[:, 1], times)
        psi_unwrapped = unwrap_angle_sequence(waypoints_xyz[:, 2])
        self.traj_psi = MinJerkTrajectory1D(psi_unwrapped, times)

    def eval(self, t, order=0):
        return np.array([self.traj_x.eval(t, order), self.traj_y.eval(t, order), self.traj_psi.eval(t, order)])

    def sample(self, dt=0.05):
        t0, tf = (self.times[0], self.times[-1])
        ts = np.arange(t0, tf + 1e-09, dt)
        eta = np.array([self.eval(t, 0) for t in ts])
        eta_d = np.array([self.eval(t, 1) for t in ts])
        eta_dd = np.array([self.eval(t, 2) for t in ts])
        eta_ddd = np.array([self.eval(t, 3) for t in ts])
        return (ts, eta, eta_d, eta_dd, eta_ddd)
