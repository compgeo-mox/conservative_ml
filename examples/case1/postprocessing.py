import numpy as np
import pygeon as pg

from main import SamplerSB
from setup import create_data


if __name__ == "__main__":
    filename = input("Filename: ")
    obj = np.load(filename)
    mu, q0, step_size = obj["mu"], obj["q0"], obj["h"]

    mdg = pg.unit_grid(2, step_size)
    mdg.compute_geometry()
    create_data(mdg)
    sampler = SamplerSB(mdg, keyword="flow")
    sampler.visualize(mu, q0, "sol")
