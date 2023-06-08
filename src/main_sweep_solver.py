import numpy as np
import scipy.sparse as sps

import pygeon as pg

import time
import sys

sys.path.insert(0, "src/")
from sweep_solver import SweepSolver, KrylovSweepSolver


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

    # Direct solve
    q_direct, p_direct, B = direct_solve(mdg, discr, f, g)

    # Sweeper solve
    sws = SweepSolver(mdg, discr)
    q_sweep, p_sweep = sws.solve(f, g)

    assert np.allclose(q_direct, q_sweep)
    assert np.allclose(p_direct, p_sweep)

    # Computing the homogeneous solution with conjugate gradients
    tol = 1e-10
    kry_sws = KrylovSweepSolver(mdg, discr, tol)
    q_kryl, p_kryl = kry_sws.solve(f, g)

    print(
        "CG with tol {:.0E}".format(tol),
        "has q-accuracy {:.2E}".format(np.linalg.norm(q_direct - q_kryl)),
    )
    print(
        "CG with tol {:.0E}".format(tol),
        "has p-accuracy {:.2E}".format(np.linalg.norm(p_direct - p_kryl)),
    )
    assert np.allclose(B @ q_direct, f)


def direct_solve(mdg, discr, f, g):
    A = pg.face_mass(mdg, discr)
    B = pg.cell_mass(mdg) @ pg.div(mdg)

    spp = sps.bmat([[A, -B.T], [B, None]], format="csc")
    rhs = np.hstack((g, f))
    print("Full  problem is", spp.shape, "with", spp.nnz, "nonzeros")

    qp = sps.linalg.spsolve(spp, rhs)

    q = qp[: B.shape[1]]
    p = qp[B.shape[1] :]

    return q, p, B


if __name__ == "__main__":
    main()
