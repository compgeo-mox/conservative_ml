import numpy as np
from scipy.stats import qmc
import scipy.sparse as sps

import porepy as pp
import pygeon as pg


class Sampler:
    def __init__(self, mdg, keyword):
        self.mdg = mdg
        self.keyword = keyword

        self.cell_mass = pg.cell_mass(self.mdg, keyword="unit")
        self.face_mass = pg.face_mass(self.mdg, keyword="unit")

        self.sptr = pg.SpanningTree(mdg)

        self.B = self.cell_mass @ pg.div(mdg)
        self.spp = sps.bmat([[self.face_mass, -self.B.T], [self.B, None]]).tocsc()

    def S_I(self, f):
        return self.sptr.solve(f)

    def S_0(self, r):
        return r - self.S_I(self.B @ r)

    def get_q0(self, mu):
        f = self.get_f(mu=mu)
        g = self.get_g(mu=mu)

        rhs = np.hstack((g, f))
        qp = sps.linalg.spsolve(self.spp, rhs)

        q = qp[: self.mdg.num_subdomain_faces()]

        return q - self.S_I(self.B @ q)

    def generate_set(self, num, seed=None):
        lhc = qmc.LatinHypercube(len(self.l_bounds), seed=seed)
        mu_samples = lhc.random(num)
        mu_samples = qmc.scale(mu_samples, self.l_bounds, self.u_bounds)

        for mu in mu_samples:
            yield mu, self.get_q0(mu), self.S_I(self.get_f(mu=mu))

    def compute_loss_r(self, q0_true, r):
        diff = q0_true - self.S_0(r)
        return np.sqrt(diff @ self.face_mass @ diff)

    def compute_loss_p(self, mu, q0_true, r):
        f = self.get_f(mu=mu)

        q_f = self.sptr.sweep(f)
        p_true = self.sptr.sweep_transpose(self.face_mass @ (q_f + q0_true))
        p = self.sptr.sweep_transpose(self.face_mass @ (q_f + self.S_0(r)))

        diff = p_true - p

        return np.sqrt(diff @ self.cell_mass @ diff)

    def compute_loss(self, mu, q0_true, r, weights=[0.5, 0.5]):
        loss_r = self.compute_loss_r(q0_true, r)
        loss_p = self.compute_loss_p(mu, q0_true, r)

        return weights[0] * loss_r + weights[1] * loss_p

    def compute_qp(self, mu, q0):
        f = self.get_f(mu=mu)
        g = self.get_g(mu=mu)
        q_f = self.sptr.sweep(f)

        q = q_f + q0
        p = self.sptr.sweep_transpose(self.face_mass @ q - g)

        return q, p

    def visualize(self, mu, q0, file_name):
        # visualization of the results
        q, p = self.compute_qp(mu, q0)

        # post process velocity
        proj_q = pg.eval_at_cell_centers(self.mdg, pg.RT0(self.keyword))
        cell_q = (proj_q @ q).reshape((3, -1), order="F")

        # post process pressure
        proj_p = pg.eval_at_cell_centers(self.mdg, pg.PwConstants(self.keyword))
        cell_p = proj_p @ p

        dofs = np.cumsum([sd.num_cells for sd in self.mdg.subdomains()])
        dofs = np.hstack(([0], dofs))

        # save the solutions to be exported in the data dictionary of the mdg
        for idx, (_, data) in enumerate(self.mdg.subdomains(return_data=True)):
            pp.set_solution_values(
                "cell_q", cell_q[:, dofs[idx] : dofs[idx + 1]], data, 0
            )
            pp.set_solution_values("cell_p", cell_p[dofs[idx] : dofs[idx + 1]], data, 0)

        # export the solutions
        save = pp.Exporter(self.mdg, file_name)
        save.write_vtu(["cell_q", "cell_p"])
