import numpy as np
import scipy.sparse as sps
import porepy as pp

import pygeon as pg

import sys

sys.path.insert(0, "src/")
from sampler import Sampler
from setup import create_data
from generator import generate_samples


class SamplerSB(Sampler):
    def __init__(self, mdg, keyword, tol=1e-8, max_iter=100):
        super().__init__(mdg, keyword)

        self.l_bounds = [-1, -1, -2, 0]
        self.u_bounds = [1, 1, 2, 2]

        self.num_param = len(self.l_bounds)

        discr = pg.RT0(self.keyword)
        self.eval_at_cell_centers = pg.eval_at_cell_centers(self.mdg, discr=discr)

        self.max_iter = max_iter
        self.tol = tol

    def get_f(self, **kwargs):
        mu = kwargs["mu"]

        def source(x):
            return mu[0] * np.sin(2 * np.pi * x[0]) + mu[1] * np.sin(2 * np.pi * x[1])

        f = self.cell_mass @ pg.PwConstants(self.keyword).interpolate(
            self.mdg.subdomains()[0], source
        )
        return f

    def get_g(self, **kwargs):
        return np.zeros(self.mdg.num_subdomain_faces())

    def get_q0(self, mu):
        f = self.get_f(mu=mu)
        g = self.get_g(mu=mu)

        rhs = np.hstack((g, f))
        rhs_norm = np.linalg.norm(rhs)

        qp = np.zeros(rhs.shape)

        for iter in np.arange(self.max_iter):
            q = qp[: self.mdg.num_subdomain_faces()]
            face_mass = self.assemble_face_mass(q, mu)
            spp = sps.bmat([[face_mass, -self.B.T], [self.B, None]]).tocsc()

            if np.linalg.norm(spp @ qp - rhs) <= self.tol * rhs_norm:
                # print("Converged in {:} iterations".format(iter))
                break
            else:
                qp = sps.linalg.spsolve(spp, rhs)
        else:
            print(
                "Did not converge in {} iterations on sample \nmu = {}".format(iter, mu)
            )

        return q - self.S_I(self.B @ q)

    def assemble_face_mass(self, q, mu):
        k0 = 10 ** mu[2]
        k1 = 10 ** mu[3]
        for _, data in self.mdg.subdomains(return_data=True):
            q_interp = (self.eval_at_cell_centers @ q).reshape((3, -1), order="F")
            q_norm = np.linalg.norm(q_interp, axis=0)
            perm = pp.SecondOrderTensor(1 / (1 / k0 + q_norm / k1))
            data[pp.PARAMETERS][self.keyword]["second_order_tensor"] = perm

        return pg.face_mass(self.mdg, keyword=self.keyword)

    def compute_qp(self, mu, q0):
        f = self.get_f(mu=mu)
        g = self.get_g(mu=mu)
        q_f = self.swp.sweep(f)

        q = q_f + q0

        self.set_perm(mu)
        face_mass = self.assemble_face_mass(q, mu)
        p = self.swp.sweep_transpose(face_mass @ q - g)

        return q, p


if __name__ == "__main__":
    step_size = float(input("Mesh stepsize: "))  # 0.1
    num_samples = int(input("Number of samples: "))  # 10
    seed = 0  # seed for sampling

    mdg = pg.unit_grid(2, step_size)
    mdg.compute_geometry()

    keyword = "flow"
    create_data(mdg, keyword)

    # non-linear loop data
    tol = 1e-8
    max_iter = 100

    sampler = SamplerSB(mdg, keyword, tol, max_iter)
    generate_samples(sampler, num_samples, step_size, "snapshots.npz", seed)

    print("Done.")
