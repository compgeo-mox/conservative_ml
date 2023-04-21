import numpy as np
import scipy.sparse as sps

import porepy as pp
import pygeon as pg

import sys

sys.path.insert(0, "src/")
from hodge_solver import HodgeSolver
from setup import *


def main():
    mdg = create_mdg(25)
    create_data(mdg)

    cell_mass = pg.cell_mass(mdg)
    f = cell_mass @ np.ones(mdg.num_subdomain_cells())
    g = np.zeros(mdg.num_subdomain_faces())

    hs = HodgeSolver(mdg, pg.RT0("flow"))
    q_f = hs.step1(f)
    r = hs.step2(q_f, g)
    q, p = hs.step3(q_f, r)

    print(q_f)
    # pp.plot_grid(mdg.subdomains()[0], cell_value=p)

    face_mass = pg.face_mass(mdg)
    div = pg.cell_mass(mdg) @ pg.div(mdg)
    spp = sps.bmat([[face_mass, -div.T], [div, None]]).tocsc()
    rhs = np.hstack((g, f))
    qp_ref = sps.linalg.spsolve(spp, rhs)
    q_ref = qp_ref[: g.size]
    p_ref = qp_ref[g.size :]

    err_p = p - p_ref
    err_q = q - q_ref

    print(err_p)


if __name__ == "__main__":
    main()
