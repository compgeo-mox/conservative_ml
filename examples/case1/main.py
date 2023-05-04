import numpy as np
import scipy.sparse as sps

import porepy as pp
import pygeon as pg

import sys

sys.path.insert(0, "src/")
from hodge_solver import HodgeSolver
from sampler import Sampler
from setup import *


def visualize(mdg, mu, r, sampler, file_name):
    # visualization of the results
    q_f = sampler.get_q_f(mu)
    q, p = sampler.hs.step3(q_f, r)

    # post process of r
    if mdg.dim_max() == 2:
        proj_r = pg.eval_at_cell_centers(mdg, pg.Lagrange1(sampler.keyword))
        cell_r = proj_r @ r
    if mdg.dim_max() == 3:
        proj_r = pg.eval_at_cell_centers(mdg, pg.Nedelec0(sampler.keyword))
        cell_r = (proj_r @ r).reshape((3, -1), order="F")

    # post process velocity
    proj_q = pg.eval_at_cell_centers(mdg, pg.RT0(sampler.keyword))
    cell_q = (proj_q @ q).reshape((3, -1), order="F")

    # post process pressure
    proj_p = pg.eval_at_cell_centers(mdg, pg.PwConstants(sampler.keyword))
    cell_p = proj_p @ p

    # save the solutions to be exported in the data dictionary of the mdg
    for _, data in mdg.subdomains(return_data=True):
        pp.set_solution_values("cell_r", cell_r, data, 0)
        pp.set_solution_values("cell_q", cell_q, data, 0)
        pp.set_solution_values("cell_p", cell_p, data, 0)

    # export the solutions
    save = pp.Exporter(mdg, file_name)
    save.write_vtu(["cell_r", "cell_q", "cell_p"])


def main():
    mdg = pg.grid_unitary(2, 0.125)
    mdg.compute_geometry()

    create_data(mdg)

    keyword = "flow"
    hs = HodgeSolver(mdg, pg.RT0(keyword))
    sampler = Sampler(hs)

    num_samples = 4

    r_samples = np.empty(num_samples, dtype=np.ndarray)
    mu_samples = np.empty((num_samples, 2))
    for idx, (mu, r) in enumerate(sampler.generate_set(num_samples, seed=1)):
        mu_samples[idx, :] = mu
        r_samples[idx] = r

    r_samples = np.vstack(r_samples)

    r_rand = np.random.rand(r_samples.shape[1])
    r_rand = r_samples[1, :]

    loss = sampler.compute_loss(mu_samples[0, :], r_samples[0, :], r_rand)
    print(loss)
    print(mu_samples)
    visualize(mdg, mu_samples[0, :], r_samples[0, :], sampler, "sol")

    # return

    # r, q_f = sampler.get_sample([0, 1])

    # face_mass = pg.face_mass(mdg)
    # div = pg.cell_mass(mdg) @ pg.div(mdg)
    # spp = sps.bmat([[face_mass, -div.T], [div, None]]).tocsc()
    # rhs = np.hstack((g, f))

    # qp_ref = sps.linalg.spsolve(spp, rhs)
    # q_ref = qp_ref[: g.size]
    # p_ref = qp_ref[g.size :]

    # err_p = p - p_ref
    # err_q = q - q_ref

    # print("err_p:", np.sqrt(err_p @ cell_mass @ err_p))
    # print("err_q:", np.sqrt(err_q @ face_mass @ err_q))


if __name__ == "__main__":
    main()
