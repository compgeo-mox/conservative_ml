import numpy as np
from scipy.stats import qmc
import scipy.sparse as sps

import porepy as pp
import pygeon as pg


class Sampler:
    def __init__(self, mdg, keyword):
        self.l_bounds = [0, 0]
        self.u_bounds = [4, 4]

        self.mdg = mdg
        self.keyword = keyword
        self.cell_mass = pg.cell_mass(self.mdg)
        self.face_mass = pg.face_mass(self.mdg)

    def get_f(self, mu):
        def source(x):
            return np.sin(2 * np.pi * mu[0] * x[0]) * np.sin(2 * np.pi * mu[1] * x[1])

        f = self.cell_mass @ pg.PwConstants(self.keyword).interpolate(
            self.mdg.subdomains()[0], source
        )
        return f

    def get_g(self):
        return np.zeros(self.mdg.num_subdomain_faces())


class SamplerR(Sampler):
    def __init__(self, mdg, keyword, hs):
        super().__init__(mdg, keyword)

        self.hs = hs

    def get_q_f(self, mu):
        f = self.get_f(mu)
        return self.hs.step1(f)

    def get_r(self, mu):
        g = self.get_g()
        q_f = self.get_q_f(mu)
        return self.hs.step2(q_f, g)

    def generate_set(self, num, seed=None):
        lhc = qmc.LatinHypercube(len(self.l_bounds), seed=seed)
        mu_samples = lhc.random(num)
        mu_samples = qmc.scale(mu_samples, self.l_bounds, self.u_bounds)

        for mu in mu_samples:
            yield mu, self.get_r(mu)

    def compute_loss_r(self, r_true, r):
        diff = self.hs.curl_op @ (r_true - r)
        return np.sqrt(diff @ self.face_mass @ diff)

    def compute_loss_p(self, mu, r_true, r):
        q_f = self.get_q_f(mu)

        _, p_true = self.hs.step3(q_f, r_true)
        _, p = self.hs.step3(q_f, r)
        diff = p_true - p

        return np.sqrt(diff @ self.cell_mass @ diff)

    def compute_loss(self, mu, r_true, r, weights=[0.5, 0.5]):
        loss_r = self.compute_loss_r(r_true, r)
        loss_p = self.compute_loss_p(mu, r_true, r)

        return weights[0] * loss_r + weights[1] * loss_p

    def visualize(self, mu, r, file_name):
        # visualization of the results
        q_f = self.get_q_f(mu)
        q, p = self.hs.step3(q_f, r)

        # post process of r
        if self.mdg.dim_max() == 2:
            proj_r = pg.eval_at_cell_centers(self.mdg, pg.Lagrange1(self.keyword))
            cell_r = proj_r @ r
        if self.mdg.dim_max() == 3:
            proj_r = pg.eval_at_cell_centers(self.mdg, pg.Nedelec0(self.keyword))
            cell_r = (proj_r @ r).reshape((3, -1), order="F")

        # post process velocity
        proj_q = pg.eval_at_cell_centers(self.mdg, pg.RT0(self.keyword))
        cell_q = (proj_q @ q).reshape((3, -1), order="F")

        # post process pressure
        proj_p = pg.eval_at_cell_centers(self.mdg, pg.PwConstants(self.keyword))
        cell_p = proj_p @ p

        # save the solutions to be exported in the data dictionary of the mdg
        for _, data in self.mdg.subdomains(return_data=True):
            pp.set_solution_values("cell_r", cell_r, data, 0)
            pp.set_solution_values("cell_q", cell_q, data, 0)
            pp.set_solution_values("cell_p", cell_p, data, 0)

        # export the solutions
        save = pp.Exporter(self.mdg, file_name)
        save.write_vtu(["cell_r", "cell_q", "cell_p"])


class SamplerQ(Sampler):
    def __init__(self, mdg, keyword):
        super().__init__(mdg, keyword)

        self.swp = pg.Sweeper(mdg)

        div = self.cell_mass @ pg.div(mdg)
        self.spp = sps.bmat([[self.face_mass, -div.T], [div, None]]).tocsc()
        self.curl_op = pg.curl(mdg)

    def get_q0(self, mu):
        f = self.get_f(mu)
        g = self.get_g()

        q_f = self.swp.sweep(f)

        rhs = np.hstack((g, f))
        qp = sps.linalg.spsolve(self.spp, rhs)
        return qp[: self.face_mass.shape[0]] - q_f

    def generate_set(self, num, seed=None):
        lhc = qmc.LatinHypercube(len(self.l_bounds), seed=seed)
        mu_samples = lhc.random(num)
        mu_samples = qmc.scale(mu_samples, self.l_bounds, self.u_bounds)

        for mu in mu_samples:
            yield mu, self.get_q0(mu)

    def compute_loss_r(self, q0_true, r):
        diff = q0_true - self.curl_op @ r
        return np.sqrt(diff @ self.face_mass @ diff)

    def compute_loss_p(self, mu, q0_true, r):
        f = self.get_f(mu)

        q_f = self.swp.sweep(f)
        p_true = self.swp.sweep_transpose(self.face_mass @ (q_f + q0_true))
        p = self.swp.sweep_transpose(self.face_mass @ (q_f + self.curl_op @ r))

        diff = p_true - p

        return np.sqrt(diff @ self.cell_mass @ diff)

    def compute_loss(self, mu, q0_true, r, weights=[0.5, 0.5]):
        loss_r = self.compute_loss_r(q0_true, r)
        loss_p = self.compute_loss_p(mu, q0_true, r)

        return weights[0] * loss_r + weights[1] * loss_p

    def visualize(self, mu, q0, file_name):
        # visualization of the results
        f = self.get_f(mu)
        q_f = self.swp.sweep(f)

        q = q_f + q0
        p = self.swp.sweep_transpose(self.face_mass @ q)

        # post process velocity
        proj_q = pg.eval_at_cell_centers(self.mdg, pg.RT0(self.keyword))
        cell_q = (proj_q @ q).reshape((3, -1), order="F")

        # post process pressure
        proj_p = pg.eval_at_cell_centers(self.mdg, pg.PwConstants(self.keyword))
        cell_p = proj_p @ p

        # save the solutions to be exported in the data dictionary of the mdg
        for _, data in self.mdg.subdomains(return_data=True):
            pp.set_solution_values("cell_q", cell_q, data, 0)
            pp.set_solution_values("cell_p", cell_p, data, 0)

        # export the solutions
        save = pp.Exporter(self.mdg, file_name)
        save.write_vtu(["cell_q", "cell_p"])


class SamplerSB(Sampler):
    def __init__(self, mdg, keyword):
        super().__init__(mdg, keyword)

        self.swp = pg.Sweeper(mdg)

        self.B = self.cell_mass @ pg.div(mdg)
        self.spp = sps.bmat([[self.face_mass, -self.B.T], [self.B, None]]).tocsc()

    def S_I(self, f):
        return self.swp.sweep(f)

    def S_0(self, r):
        return r - self.S_I(self.B @ r)

    def get_q0(self, mu):
        f = self.get_f(mu)
        g = self.get_g()

        rhs = np.hstack((g, f))
        qp = sps.linalg.spsolve(self.spp, rhs)

        q = qp[: self.face_mass.shape[0]]

        return self.S_0(q)

    def generate_set(self, num, seed=None):
        lhc = qmc.LatinHypercube(len(self.l_bounds), seed=seed)
        mu_samples = lhc.random(num)
        mu_samples = qmc.scale(mu_samples, self.l_bounds, self.u_bounds)

        for mu in mu_samples:
            yield mu, self.get_q0(mu)

    def compute_loss_r(self, q0_true, r):
        diff = q0_true - self.S_0(r)
        return np.sqrt(diff @ self.face_mass @ diff)

    def compute_loss_p(self, mu, q0_true, r):
        f = self.get_f(mu)

        q_f = self.swp.sweep(f)
        p_true = self.swp.sweep_transpose(self.face_mass @ (q_f + q0_true))
        p = self.swp.sweep_transpose(self.face_mass @ (q_f + self.S_0(r)))

        diff = p_true - p

        return np.sqrt(diff @ self.cell_mass @ diff)

    def compute_loss(self, mu, q0_true, r, weights=[0.5, 0.5]):
        loss_r = self.compute_loss_r(q0_true, r)
        loss_p = self.compute_loss_p(mu, q0_true, r)

        return weights[0] * loss_r + weights[1] * loss_p

    def visualize(self, mu, q0, file_name):
        # visualization of the results
        f = self.get_f(mu)
        q_f = self.swp.sweep(f)

        q = q_f + q0
        p = self.swp.sweep_transpose(self.face_mass @ q)

        # post process velocity
        proj_q = pg.eval_at_cell_centers(self.mdg, pg.RT0(self.keyword))
        cell_q = (proj_q @ q).reshape((3, -1), order="F")

        # post process pressure
        proj_p = pg.eval_at_cell_centers(self.mdg, pg.PwConstants(self.keyword))
        cell_p = proj_p @ p

        # save the solutions to be exported in the data dictionary of the mdg
        for _, data in self.mdg.subdomains(return_data=True):
            pp.set_solution_values("cell_q", cell_q, data, 0)
            pp.set_solution_values("cell_p", cell_p, data, 0)

        # export the solutions
        save = pp.Exporter(self.mdg, file_name)
        save.write_vtu(["cell_q", "cell_p"])
