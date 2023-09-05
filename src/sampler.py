import numpy as np
from scipy.stats import qmc
import scipy.sparse as sps

import porepy as pp
import pygeon as pg


class SpanningTrees:
    def __init__(self, mdg, weights, starting_faces=None) -> None:
        if starting_faces is None:
            num = np.asarray(weights).size
            starting_faces = self.find_starting_faces(mdg, num)

        self.sptrs = [pg.SpanningTree(mdg, f) for f in starting_faces]
        self.avg = lambda v: np.average(v, axis=0, weights=weights)

    def solve(self, f) -> np.ndarray:
        return self.avg([st.solve(f) for st in self.sptrs])

    def solve_transpose(self, rhs) -> np.ndarray:
        return self.avg([st.solve_transpose(rhs) for st in self.sptrs])

    def find_starting_faces(self, mdg, num):
        if isinstance(mdg, pp.Grid):
            sd = mdg
        elif isinstance(mdg, pp.MixedDimensionalGrid):
            # Extract the top-dimensional grid
            sd = mdg.subdomains()[0]
            assert sd.dim == mdg.dim_max()
        else:
            raise TypeError

        faces = np.where(sd.tags["domain_boundary_faces"])[0]
        return faces[np.linspace(0, faces.size, num, endpoint=False, dtype=int)]


class Sampler:
    def __init__(self, mdg, keyword, num_trees=5):
        self.mdg = mdg
        self.keyword = keyword

        self.cell_mass = pg.cell_mass(self.mdg, keyword="unit")
        self.face_mass = pg.face_mass(self.mdg, keyword="unit")

        self.sptr = SpanningTrees(mdg, [1 / num_trees] * num_trees)

        self.B = self.cell_mass @ pg.div(mdg)
        self.spp = sps.bmat([[self.face_mass, -self.B.T], [self.B, None]]).tocsc()

    def S_I(self, f):
        return self.sptr.solve(f)

    def S_0(self, r):
        return r - self.S_I(self.B @ r)

    def get_q0(self, mu):
        f = self.get_f(mu=mu)
        g = self.get_g(mu=mu)

        rhs = np.hstack((g, f))
        qp = sps.linalg.spsolve(self.spp, rhs)

        q = qp[: self.mdg.num_subdomain_faces()]

        return q - self.S_I(self.B @ q)

    def generate_set(self, num, seed=None):
        lhc = qmc.LatinHypercube(len(self.l_bounds), seed=seed)
        mu_samples = lhc.random(num)
        mu_samples = qmc.scale(mu_samples, self.l_bounds, self.u_bounds)

        for mu in mu_samples:
            yield mu, self.get_q0(mu), self.S_I(self.get_f(mu=mu))

    def compute_loss_r(self, q0_true, r):
        diff = q0_true - self.S_0(r)
        return np.sqrt(diff @ self.face_mass @ diff)

    def compute_loss_p(self, mu, q0_true, r):
        f = self.get_f(mu=mu)

        q_f = self.sptr.solve(f)
        p_true = self.sptr.solve_transpose(self.face_mass @ (q_f + q0_true))
        p = self.sptr.solve_transpose(self.face_mass @ (q_f + self.S_0(r)))

        diff = p_true - p

        return np.sqrt(diff @ self.cell_mass @ diff)

    def compute_loss(self, mu, q0_true, r, weights=[0.5, 0.5]):
        loss_r = self.compute_loss_r(q0_true, r)
        loss_p = self.compute_loss_p(mu, q0_true, r)

        return weights[0] * loss_r + weights[1] * loss_p

    def compute_qp(self, mu, q0):
        f = self.get_f(mu=mu)
        g = self.get_g(mu=mu)
        q_f = self.sptr.solve(f)

        q = q_f + q0
        p = self.sptr.solve_transpose(self.face_mass @ q - g)

        return q, p

    def visualize(self, mu, q0, file_name, folder=None, file_name_sptr=None):
        # visualization of the results
        q, p = self.compute_qp(mu, q0)

        # post process velocity
        proj_q = pg.eval_at_cell_centers(self.mdg, pg.RT0(self.keyword))
        cell_q = (proj_q @ q).reshape((3, -1), order="F")

        # post process pressure
        proj_p = pg.eval_at_cell_centers(self.mdg, pg.PwConstants(self.keyword))
        cell_p = proj_p @ p

        dofs = np.cumsum([sd.num_cells for sd in self.mdg.subdomains()])
        dofs = np.hstack(([0], dofs))

        # save the solutions to be exported in the data dictionary of the mdg
        for idx, (_, data) in enumerate(self.mdg.subdomains(return_data=True)):
            pp.set_solution_values(
                "cell_q", cell_q[:, dofs[idx] : dofs[idx + 1]], data, 0
            )
            pp.set_solution_values("cell_p", cell_p[dofs[idx] : dofs[idx + 1]], data, 0)

        # export the solutions
        save = pp.Exporter(self.mdg, file_name, folder_name=folder)
        save.write_vtu(["cell_q", "cell_p"])

        if file_name_sptr is not None:
            self.sptr.visualize_2d(self.mdg, file_name_sptr)
