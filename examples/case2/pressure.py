import numpy as np
import pygeon as pg
import sys
import os
import platform

from main import SamplerSB
from setup import create_data


if __name__ == "__main__":

    filename =  "ROMoutputs_case2.npz"
    obj = np.load(filename)
    step_size = obj["h"]
    mesh_kwargs = {"mesh_size_frac": step_size, "mesh_size_min": step_size}
    mdg = fracture_grid(mesh_kwargs)
    mdg.compute_geometry()
    keyword = "flow"
    create_data(mdg, keyword)
    sampler = SamplerSB(mdg, keyword)

    def pressure(mu, q0):
        return sampler.compute_qp(mu, q0)[1]

    def l2(p):
        return np.sqrt((p.reshape(1,-1) @ sampler.cell_mass @ p.reshape(-1,1)).reshape(1))

    models = ["podnn", "blackbox_l2", "blackbox_h1", "curl_dlrom", "s0_dlrom"]


    mu = np.atleast_2d(obj["mu"])
    n = len(mu)

    errors = {model:0 for model in models}
    for i in range(n):
        ptrue = pressure(mu[i], obj["q0_fom"][i])
        denominator = l2(ptrue)
        for model in models:
            p = pressure(mu[i], obj["q0_"+model][i])
            error = l2(ptrue-p)/denominator
            errors[model] += error/n

    num2p = lambda x: ("%.2f" % (100*x)) + "%"

    print("--- Average pressure errors ---")
    for model in models:
        print("%s\t%s" % (model.upper().replace("_","-") + ("\t" if model == "podnn" else ""), num2p(errors[model])))
    print("")
