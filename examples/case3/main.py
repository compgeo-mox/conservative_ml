import numpy as np
import scipy.sparse as sps
import porepy as pp

import pygeon as pg

import sys

sys.path.insert(0, "src/")
from sampler import Sampler
from setup import create_data


class SamplerSB(Sampler):
    def __init__(self, mdg, keyword):
        super().__init__(mdg, keyword)

        self.l_bounds = [-1, -1, 1e-2, 1e-2]
        self.u_bounds = [1, 1, 1, 1]
        self.num_param = len(self.l_bounds)

        discr = pg.RT0(self.keyword)
        self.eval_at_cell_centers = pg.eval_at_cell_centers(self.mdg, discr=discr)

        self.max_iter = 100
        self.tol = 1e-8

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
        for _, data in self.mdg.subdomains(return_data=True):
            q_interp = (self.eval_at_cell_centers @ q).reshape((3, -1), order="F")
            q_norm = np.linalg.norm(q_interp, axis=0)
            perm = pp.SecondOrderTensor(1 / (1 / mu[2] + q_norm / mu[3]))
            data[pp.PARAMETERS][self.keyword]["second_order_tensor"] = perm

        return pg.face_mass(self.mdg, keyword=self.keyword)


def main(mdg, keyword, num_samples, seed=1):
    sampler = SamplerSB(mdg, keyword)

    q0_samples = np.empty(num_samples, dtype=np.ndarray)
    mu_samples = np.empty((num_samples, sampler.num_param))
    for idx, (mu, q0) in enumerate(sampler.generate_set(num_samples, seed=seed)):
        mu_samples[idx, :] = mu
        q0_samples[idx] = q0

    q0_samples = np.vstack(q0_samples)

    S_0 = sampler.S_0(sps.eye(q0_samples.shape[1]))

    return sampler, mu_samples, q0_samples, pg.curl(mdg), S_0


if __name__ == "__main__":
    step_size = float(input("Mesh stepsize: "))  # 0.1
    num_samples = int(input("Number of samples: "))  # 10
    mdg = pg.unit_grid(2, step_size)
    mdg.compute_geometry()

    create_data(mdg)

    keyword = "flow"

    sampler, mu, q0, curl, S_0 = main(mdg, keyword, num_samples)

    np.savez(
        "snapshots.npz",
        curl=curl.todense(),
        S_0=S_0.todense(),
        face_mass=sampler.face_mass.todense(),
        cell_mass=sampler.cell_mass.todense(),
        mu=mu,
        q0=q0,
        h=step_size,
    )
    print("Done.")
