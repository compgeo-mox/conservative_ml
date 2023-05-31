import numpy as np
import pygeon as pg

from main import SamplerSB
from setup import create_data


if __name__ == "__main__":
    filename = input("Filename: ")
    obj = np.load(filename)
    mu, q0, step_size = obj["mu"], obj["q0"], obj["h"]

    mesh_kwargs = {"mesh_size_frac": step_size, "mesh_size_min": step_size / 10}
    mdg, _ = pp.md_grids_2d.seven_fractures_one_L_intersection(mesh_kwargs)
    pg.convert_from_pp(mdg)
    mdg.compute_geometry()

    create_data(mdg)
    sampler = SamplerSB(mdg, keyword="flow")
    sampler.visualize(mu, q0, "sol")
