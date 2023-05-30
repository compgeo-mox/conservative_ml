import numpy as np
import pygeon as pg

from sampler import SamplerQ
from setup import create_data


if __name__ == "__main__":
    filename = input("Filename: ")
    obj = np.load(filename)
    mu, q0, stepsize = obj['mu'], obj['q0'], obj['h']

    mdg = pg.unit_grid(2, stepsize)
    mdg.compute_geometry()
    create_data(mdg)
    sampler = SamplerQ(mdg, keyword = "flow")
    sampler.visualize(mu, q0, "sol")
