import casadi as ca
import numpy as np

class USV_MPC:

    def __init__(self, dt_mpc=0.2, N_mpc=25):
        self.dt = dt_mpc
        self.N = N_mpc
        self.nx = 6
        self.nu = 2
        self.m11 = 90.03238
        self.m22 = 91.066933
        self.m33 = 334.867746
        self.Xu = 23.237495
        self.Xuu = 38.22715
        self.Yv = 26.016985
        self.Yvv = 56.929867
        self.Nr = 79.463013
        self.Nrr = 30.613898
        self.opti = ca.Opti()
        self.X = self.opti.variable(self.nx, self.N + 1)
        self.U = self.opti.variable(self.nu, self.N)
        self.x0 = self.opti.parameter(self.nx)
        self.X_ref = self.opti.parameter(self.nx, self.N + 1)
        self.f = self.build_dynamics()
        self.opti.subject_to(self.X[:, 0] == self.x0)
        for k in range(self.N):
            k1 = self.f(self.X[:, k], self.U[:, k])
            k2 = self.f(self.X[:, k] + self.dt / 2 * k1, self.U[:, k])
            k3 = self.f(self.X[:, k] + self.dt / 2 * k2, self.U[:, k])
            k4 = self.f(self.X[:, k] + self.dt * k3, self.U[:, k])
            x_next = self.X[:, k] + self.dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
            self.opti.subject_to(self.X[:, k + 1] == x_next)
        self.opti.subject_to(self.opti.bounded(-113.0, self.U[0, :], 145.0))
        self.opti.subject_to(self.opti.bounded(-130.0, self.U[1, :], 130.0))
        Q = np.diag([20.0, 20.0, 30.0, 2.0, 2.0, 5.0])
        R = np.diag([0.001, 0.005])
        cost = 0
        for k in range(self.N):
            err = self.X[:, k] - self.X_ref[:, k]
            cost += ca.mtimes([err.T, Q, err]) + ca.mtimes([self.U[:, k].T, R, self.U[:, k]])
        err_N = self.X[:, self.N] - self.X_ref[:, self.N]
        cost += ca.mtimes([err_N.T, Q * 10, err_N])
        self.opti.minimize(cost)
        p_opts = {'expand': True, 'print_time': False}
        s_opts = {'max_iter': 50, 'print_level': 0, 'sb': 'yes', 'tol': 0.0001}
        self.opti.solver('ipopt', p_opts, s_opts)

    def build_dynamics(self):
        x = ca.MX.sym('x')
        y = ca.MX.sym('y')
        psi = ca.MX.sym('psi')
        u = ca.MX.sym('u')
        v = ca.MX.sym('v')
        r = ca.MX.sym('r')
        z = ca.vertcat(x, y, psi, u, v, r)
        Tu = ca.MX.sym('Tu')
        Tr = ca.MX.sym('Tr')
        w = ca.vertcat(Tu, Tr)
        dx = u * ca.cos(psi) - v * ca.sin(psi)
        dy = u * ca.sin(psi) + v * ca.cos(psi)
        dpsi = r
        du = (Tu + self.m22 * v * r - self.Xu * u - self.Xuu * ca.fabs(u) * u) / self.m11
        dv = (-self.m11 * u * r - self.Yv * v - self.Yvv * ca.fabs(v) * v) / self.m22
        dr = (Tr - (self.m22 - self.m11) * u * v - self.Nr * r - self.Nrr * ca.fabs(r) * r) / self.m33
        dz = ca.vertcat(dx, dy, dpsi, du, dv, dr)
        return ca.Function('f', [z, w], [dz])

    def solve(self, x0, X_ref_vals):
        self.opti.set_value(self.x0, x0)
        self.opti.set_value(self.X_ref, X_ref_vals)
        try:
            sol = self.opti.solve()
            u_opt = sol.value(self.U)
            x_opt = sol.value(self.X)
            self.opti.set_initial(self.X, x_opt)
            self.opti.set_initial(self.U, u_opt)
            return (u_opt[:, 0], x_opt)
        except Exception as e:
            u_opt = self.opti.debug.value(self.U)
            x_opt = self.opti.debug.value(self.X)
            return (u_opt[:, 0], x_opt)
