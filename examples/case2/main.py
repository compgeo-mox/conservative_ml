import numpy as np
import scipy.sparse as sps

import porepy as pp
import pygeon as pg

import sys

sys.path.insert(0, "src/")
from sampler import Sampler
from setup import create_data, fracture_grid
from generator import generate_samples


class SamplerSB(Sampler):
    def __init__(self, mdg, keyword):
        super().__init__(mdg, keyword)

        # mu[0] log of fracture permeability
        # mu[1:4] alpha for bc
        # mu[4] f in the fractures

        self.l_bounds = [-4, 0, 0, 0, 1]
        self.u_bounds = [4, 1, 1, 1, 2]

        self.num_param = len(self.l_bounds)

    def get_f(self, **kwargs):
        mu = kwargs.get("mu", None)

        f = []
        for sd, data in self.mdg.subdomains(return_data=True):
            if sd.dim < self.mdg.dim_max():
                specific_volumes = data[pp.PARAMETERS][self.keyword]["specific_volumes"]
                f.append(mu[4] * specific_volumes * np.ones(sd.num_cells))
            else:
                f.append(np.zeros(sd.num_cells))

        return np.hstack(f)

    def get_g(self, **kwargs):
        mu = kwargs.get("mu", None)

        p_bc = lambda x: np.dot(x, mu[1:4])
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
        f = self.get_f(mu=mu)
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
        q_f = self.sptr.sweep(f)

        q = q_f + q0

        self.set_perm(mu)
        face_mass = pg.face_mass(self.mdg, keyword=self.keyword)
        p = self.sptr.sweep_transpose(face_mass @ q - g)

        return q, p


if __name__ == "__main__":
    num_samples = int(input("Number of samples: "))
    step_size = 0.1
    seed = 0  # seed for sampling

    mesh_kwargs = {"mesh_size_frac": step_size, "mesh_size_min": step_size}
    mdg = fracture_grid(mesh_kwargs)
    mdg.compute_geometry()

    keyword = "flow"
    create_data(mdg, keyword)

    sampler = SamplerSB(mdg, keyword)
    generate_samples(sampler, num_samples, step_size, "snapshots.npz", seed)

    print("Done.")
