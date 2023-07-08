import numpy as np
import pygeon as pg


from main import SamplerSB
from setup import create_data

if __name__ == "__main__":
    filename = "snapshots.npz"  # input("Filename: ")
    obj = np.load(filename)
    mu, q0, step_size = obj["mu"], obj["q0"], obj["h"]

    mdg = pg.unit_grid(2, step_size)
    mdg.compute_geometry()

    keyword = "flow"
    create_data(mdg, keyword)

    sampler = SamplerSB(mdg, keyword)

    # pos = 0 is used in the paper
    pos = 0
    print(mu[pos])
    sampler.visualize(mu[pos], q0[pos], "sol")
