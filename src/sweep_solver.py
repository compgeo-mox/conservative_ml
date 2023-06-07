import numpy as np
import scipy.sparse as sps

import porepy as pp
import pygeon as pg


class SweepSolver:
    def __init__(self, mdg: pg.MixedDimensionalGrid, discr):
        self.mdg = mdg
        self.discr = discr

        self.face_mass = pg.face_mass(self.mdg, discr=self.discr)

        self.swp = pg.Sweeper(mdg)
        self.SB = self.sweep_the_div()

    def sweep_the_div(self, tol=1e-10):
        div = pg.cell_mass(self.mdg) @ pg.div(self.mdg, discr=self.discr)
        SB = self.swp.sweep(div)

        SB.data[np.abs(SB.data) < tol] = 0
        SB.eliminate_zeros()

        return SB

    def solve(self, f, g):
        q_f = self.compute_q_f(f)
        q_0 = self.compute_q_0(q_f, g)
        return self.compute_qp(q_f, q_0, g)

    def compute_q_f(self, f):
        return self.swp.sweep(f)

    def compute_q_0(self, q_f, g):
        S_0 = sps.eye(*self.SB.shape, format="csc") - self.SB

        A = S_0.T @ self.face_mass @ S_0
        A += self.SB.T @ self.face_mass @ self.SB

        rhs = S_0.T @ (g - self.face_mass @ q_f)

        print("Swept problem is", A.shape, "with", A.nnz, "nonzeros")

        ls = pg.LinearSystem(A, rhs)

        return S_0 @ ls.solve()

    def compute_qp(self, q_f, q_0, g):
        q = q_f + q_0
        p = self.swp.sweep_transpose(self.face_mass @ q - g)

        return q, p
