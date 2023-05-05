import numpy as np

import pygeon as pg

import sys

sys.path.insert(0, "src/")
from hodge_solver import HodgeSolver
from sampler import SamplerR, SamplerQ
from setup import create_data


def main_sampler_r(mdg, keyword, num_samples):
    hs = HodgeSolver(mdg, pg.RT0(keyword))
    sampler = SamplerR(mdg, keyword, hs)

    r_samples = np.empty(num_samples, dtype=np.ndarray)
    mu_samples = np.empty((num_samples, 2))
    for idx, (mu, r) in enumerate(sampler.generate_set(num_samples, seed=1)):
        mu_samples[idx, :] = mu
        r_samples[idx] = r

    r_samples = np.vstack(r_samples)

    r_rand = np.random.rand(r_samples.shape[1])
    r_rand = r_samples[1, :]

    loss = sampler.compute_loss(mu_samples[0, :], r_samples[0, :], r_rand)
    print(loss)
    print(mu_samples)
    sampler.visualize(mu_samples[0, :], r_samples[0, :], "sol")


def main_sampler_q(mdg, keyword, num_samples):
    sampler = SamplerQ(mdg, keyword)

    q0_samples = np.empty(num_samples, dtype=np.ndarray)
    mu_samples = np.empty((num_samples, 2))
    for idx, (mu, q0) in enumerate(sampler.generate_set(num_samples, seed=1)):
        mu_samples[idx, :] = mu
        q0_samples[idx] = q0

    q0_samples = np.vstack(q0_samples)

    r_rand = np.random.rand(mdg.num_subdomain_ridges())

    loss = sampler.compute_loss(mu_samples[0, :], q0_samples[0, :], r_rand)
    print(loss)
    print(mu_samples)
    sampler.visualize(mu_samples[0, :], q0_samples[0, :], "sol")


if __name__ == "__main__":
    mdg = pg.grid_unitary(2, 0.125)
    mdg.compute_geometry()

    create_data(mdg)

    keyword = "flow"
    num_samples = 4

    main_sampler_q(mdg, keyword, num_samples)
