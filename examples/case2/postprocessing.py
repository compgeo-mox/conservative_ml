import numpy as np
import pygeon as pg
import porepy as pp


from main import SamplerSB
from setup import create_data, fracture_grid


if __name__ == "__main__":
    filename = "snapshots.npz"  # input("Filename: ")
    obj = np.load(filename)
    mu, q0, step_size = obj["mu"], obj["q0"], obj["h"]

    mesh_kwargs = {"mesh_size_frac": step_size, "mesh_size_min": step_size}
    mdg = fracture_grid(mesh_kwargs)
    mdg.compute_geometry()

    keyword = "flow"
    create_data(mdg, keyword)

    sampler = SamplerSB(mdg, keyword)

    # pos = 0 is used in the paper
    pos = 0
    print(mu[pos])
    sampler.visualize(mu[pos], q0[pos], "sol")
