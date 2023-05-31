import numpy as np
import scipy.sparse as sps

import pygeon as pg

import sys

sys.path.insert(0, "src/")
from sampler import Sampler
from setup import create_data


class SamplerSB(Sampler):
    def __init__(self, mdg, keyword):
        super().__init__(mdg, keyword)

        self.l_bounds = [0, 0]
        self.u_bounds = [4, 4]

    def get_f(self, **kwargs):
        mu = kwargs["mu"]

        def source(x):
            return np.sin(2 * np.pi * mu[0] * x[0]) * np.sin(2 * np.pi * mu[1] * x[1])

        f = self.cell_mass @ pg.PwConstants(self.keyword).interpolate(
            self.mdg.subdomains()[0], source
        )
        return f

    def get_g(self, **kwargs):
        return np.zeros(self.mdg.num_subdomain_faces())


def main(mdg, keyword, num_samples, seed=1):
    sampler = SamplerSB(mdg, keyword)

    q0_samples = np.empty(num_samples, dtype=np.ndarray)
    mu_samples = np.empty((num_samples, 2))
    for idx, (mu, q0) in enumerate(sampler.generate_set(num_samples, seed=seed)):
        mu_samples[idx, :] = mu
        q0_samples[idx] = q0

    q0_samples = np.vstack(q0_samples)

    S_0 = sampler.S_0(sps.eye(q0_samples.shape[1]))

    return sampler, mu_samples, q0_samples, pg.curl(mdg), S_0


if __name__ == "__main__":
    step_size = float(input("Mesh stepsize: "))
    num_samples = int(input("Number of samples: "))
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
