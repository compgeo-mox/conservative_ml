import numpy as np
import scipy.sparse as sps

import porepy as pp
import pygeon as pg

import sys

sys.path.insert(0, "src/")
from hodge_solver import HodgeSolver
from sampler import Sampler
from setup import create_data


class SamplerSB(Sampler):
    def __init__(self, mdg, keyword):
        super().__init__(mdg, keyword)

        # mu[0] log of fracture permeability
        # mu[1:3] alpha for bc

        self.l_bounds = [-4, 0, 0]
        self.u_bounds = [4, 1, 1]

        self.num_param = len(self.l_bounds)

    def get_f(self, **kwargs):
        return np.zeros(self.mdg.num_subdomain_cells())

    def get_g(self, **kwargs):
        mu = kwargs.get("mu", None)

        p_bc = lambda x: np.dot(x[:2], mu[1:3])
        RT0 = pg.RT0("bc_val")

        bc_val = []
        for sd in self.mdg.subdomains():
            b_faces = sd.tags["domain_boundary_faces"]
            bc_val.append(-RT0.assemble_nat_bc(sd, p_bc, b_faces))

        return np.hstack(bc_val)

    def set_perm(self, mu):
        K = 10 ** mu[0]
        for sd, data in self.mdg.subdomains(return_data=True):
            if sd.dim < self.mdg.dim_max():
                specific_volumes = data[pp.PARAMETERS][self.keyword]["specific_volumes"]
                perm = pp.SecondOrderTensor(
                    K * specific_volumes * np.ones(sd.num_cells)
                )
                data[pp.PARAMETERS][self.keyword]["second_order_tensor"] = perm

        for _, data in self.mdg.interfaces(return_data=True):
            aperture = data[pp.PARAMETERS][self.keyword]["aperture"]
            kn = K / (aperture / 2)
            data[pp.PARAMETERS][self.keyword]["normal_diffusivity"] = kn

    def get_q0(self, mu):
        f = self.get_f()
        g = self.get_g(mu=mu)

        self.set_perm(mu)

        face_mass = pg.face_mass(self.mdg, keyword=self.keyword)
        spp = sps.bmat([[face_mass, -self.B.T], [self.B, None]]).tocsc()

        rhs = np.hstack((g, f))
        qp = sps.linalg.spsolve(spp, rhs)

        q = qp[: self.mdg.num_subdomain_faces()]

        return q - self.S_I(self.B @ q)

    def compute_qp(self, mu, q0):
        f = self.get_f(mu=mu)
        g = self.get_g(mu=mu)
        q_f = self.swp.sweep(f)

        q = q_f + q0

        self.set_perm(mu)
        face_mass = pg.face_mass(self.mdg, keyword=self.keyword)
        p = self.swp.sweep_transpose(face_mass @ q - g)

        return q, p


def main(mdg, keyword, num_samples, seed=1):
    sampler = SamplerSB(mdg, keyword)

    q0_samples = np.empty(num_samples, dtype=np.ndarray)
    mu_samples = np.empty((num_samples, sampler.num_param))
    for idx, (mu, q0) in enumerate(sampler.generate_set(num_samples, seed=seed)):
        mu_samples[idx, :] = mu
        q0_samples[idx] = q0

    q0_samples = np.vstack(q0_samples)

    S_0 = sampler.S_0(sps.eye(q0_samples.shape[1]))

    return sampler, mu_samples, q0_samples, pg.curl(mdg), S_0


if __name__ == "__main__":
    # step_size = float(input("Mesh stepsize: "))
    num_samples = int(input("Number of samples: "))

    step_size = 0.1
    #num_samples = 10

    mesh_kwargs = {"mesh_size_frac": step_size, "mesh_size_min": step_size / 10}
    mdg, _ = pp.md_grids_2d.seven_fractures_one_L_intersection(mesh_kwargs)
    pg.convert_from_pp(mdg)
    mdg.compute_geometry()

    keyword = "flow"
    create_data(mdg, keyword)

    sampler, mu, q0, curl, S_0 = main(mdg, keyword, num_samples)

    np.savez(
        "snapshots.npz",
        curl=curl.todense(),
        S_0=S_0.todense(),
        face_mass=sampler.face_mass.todense(),
        cell_mass=sampler.cell_mass.todense(),
        mu=mu,
        q0=q0,
        h=step_size,
    )
    print("Done.")
