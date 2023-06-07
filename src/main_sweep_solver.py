import numpy as np
import scipy.sparse as sps

import pygeon as pg

import time
import sys

sys.path.insert(0, "src/")
from sweep_solver import SweepSolver


def main():
    # Geometrical set-up
    sd = pg.unit_grid(2, 0.1)
    mdg = pg.as_mdg(sd)
    pg.convert_from_pp(mdg)
    mdg.compute_geometry()

    # Source and boundary terms
    np.random.seed(0)
    f = np.random.rand(mdg.num_subdomain_cells())
    g = np.random.rand(mdg.num_subdomain_faces())

    discr = pg.RT0("unit")

    q_direct, p_direct = direct_solve(mdg, discr, f, g)

    sws = SweepSolver(mdg, discr)
    q_sweep, p_sweep = sws.solve(f, g)

    assert np.allclose(q_direct, q_sweep)
    assert np.allclose(p_direct, p_sweep)


def direct_solve(mdg, discr, f, g):
    A = pg.face_mass(mdg, discr)
    B = pg.cell_mass(mdg) @ pg.div(mdg)

    spp = sps.bmat([[A, -B.T], [B, None]], format="csc")
    rhs = np.hstack((g, f))
    print("Full  problem is", spp.shape, "with", spp.nnz, "nonzeros")

    qp = sps.linalg.spsolve(spp, rhs)

    q = qp[: B.shape[1]]
    p = qp[B.shape[1] :]

    return q, p


if __name__ == "__main__":
    main()
