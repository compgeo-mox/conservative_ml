import numpy as np
import scipy.sparse as sps

import porepy as pp
import pygeon as pg


class SweepSolver:
    def __init__(self, mdg: pg.MixedDimensionalGrid, discr):
        self.mdg = mdg
        self.discr = discr

        self.face_mass = pg.face_mass(self.mdg, discr=self.discr)

        self.swp = pg.SpanningTree(mdg)
        self.SB = self.sweep_the_div()

    def sweep_the_div(self, tol=1e-10):
        div = pg.cell_mass(self.mdg) @ pg.div(self.mdg, discr=self.discr)
        SB = self.swp.solve(div)

        SB.data[np.abs(SB.data) < tol] = 0
        SB.eliminate_zeros()

        return SB

    def solve(self, f, g):
        q_par = self.compute_q_par(f)
        q_hom = self.compute_q_hom(q_par, g)
        return self.compute_qp(q_par, q_hom, g)

    def compute_q_par(self, f):
        return self.swp.solve(f)

    def compute_q_hom(self, q_par, g):
        S_0 = sps.eye(*self.SB.shape, format="csc") - self.SB

        A = S_0.T @ self.face_mass @ S_0
        A += self.SB.T @ self.face_mass @ self.SB

        rhs = S_0.T @ (g - self.face_mass @ q_par)

        print("Swept problem is", A.shape, "with", A.nnz, "nonzeros")

        ls = pg.LinearSystem(A, rhs)

        return S_0 @ ls.solve()

    def compute_qp(self, q_par, q_hom, g):
        q = q_par + q_hom
        p = self.swp.solve_transpose(self.face_mass @ q - g)

        return q, p


class KrylovSweepSolver(SweepSolver):
    def __init__(self, mdg: pg.MixedDimensionalGrid, discr, tol):
        super().__init__(mdg, discr)
        self.tol = tol

    def compute_q_hom(self, q_par, g):
        S_0 = sps.eye(*self.SB.shape, format="csc") - self.SB

        A = S_0.T @ self.face_mass @ S_0

        rhs = S_0.T @ (g - self.face_mass @ q_par)
        ls = pg.LinearSystem(A, rhs)

        return S_0 @ ls.solve(solver=self.cg)

    def cg(self, A, b):
        return sps.linalg.cg(A, b, tol=self.tol)[0]
