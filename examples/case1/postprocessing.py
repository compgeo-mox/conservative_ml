import numpy as np
import pygeon as pg
import sys
import os
import platform

from main import SamplerSB
from setup import create_data


if __name__ == "__main__":
    # pos = which solution to plot, among the given list (should be a nonnegative integer)
    # model = where the output comes from, should be one of "FOM", "PODNN", "BLACKBOX", "curl-DLROM", "st-DLROM"
    pos, model = int(sys.argv[1]), sys.argv[2]

    filename =  "ROMoutputs_case1.npz"
    obj = np.load(filename)
    mu, q0, step_size = obj["mu"], obj["q0_"+model], obj["h"]
    mu, q0 = np.atleast_2d(mu), np.atleast_2d(q0)
    mdg = pg.unit_grid(2, step_size)
    mdg.compute_geometry()

    keyword = "flow"
    create_data(mdg, keyword)

    sampler = SamplerSB(mdg, keyword)

    aux = ("%.3f, "*len(mu[pos]))[:-2] % tuple(mu[pos])
    print("Parameter values: [%s]" % aux)
    sampler.visualize(mu[pos], q0[pos], "sol")
    print("Opening in Paraview...")
    if(platform.system()=="Windows"):
        os.system("start /B sol_2.vtu")
    else:
        os.system("xdg-open sol_2.vtu &")
    print("... done.\n")
