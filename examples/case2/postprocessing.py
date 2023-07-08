import numpy as np
import pygeon as pg
import porepy as pp
import sys
import os


from main import SamplerSB
from setup import create_data, fracture_grid


if __name__ == "__main__":
    filename =  "ROMoutputs_case2.npz"
    obj = np.load(filename)
    mu, q0, step_size = obj["mu"], obj["q0"], obj["h"]
    mu, q0 = np.atleast_2d(mu), np.atleast_2d(q0)

    mesh_kwargs = {"mesh_size_frac": step_size, "mesh_size_min": step_size}
    mdg = fracture_grid(mesh_kwargs)
    mdg.compute_geometry()

    keyword = "flow"
    create_data(mdg, keyword)

    sampler = SamplerSB(mdg, keyword)

    # pos = 0 is used in the paper
    pos = int(sys.argv[1])
    aux = ("%.3f, "*len(mu[pos]))[:-2] % tuple(mu[pos])
    print("Parameter values: [%s]" % aux)
    sampler.visualize(mu[pos], q0[pos], "sol")
    print("Opening in Paraview...\n")
    os.system("sol_2.vtu")
    print("... done.")
