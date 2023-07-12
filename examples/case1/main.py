import numpy as np

import pygeon as pg

import sys

sys.path.insert(0, "src/")
from sampler import Sampler
from setup import create_data
from generator import generate_samples


class SamplerSB(Sampler):
    def __init__(self, mdg, keyword):
        super().__init__(mdg, keyword)

        self.l_bounds = [1, 1]
        self.u_bounds = [4, 4]

        self.num_param = len(self.l_bounds)

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


if __name__ == "__main__":
    num_samples = int(input("Number of samples: "))
    step_size = 1 / 32
    seed = 0  # seed for sampling

    mdg = pg.unit_grid(2, step_size)
    mdg.compute_geometry()

    keyword = "flow"
    create_data(mdg, keyword)

    sampler = SamplerSB(mdg, keyword)
    generate_samples(sampler, num_samples, step_size, "snapshots.npz", seed)

    print("Done.")
