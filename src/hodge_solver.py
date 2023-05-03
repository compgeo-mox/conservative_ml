import numpy as np
import scipy.sparse as sps

import porepy as pp
import pygeon as pg


class HodgeSolver:
    def __init__(self, mdg, discr):
        self.mdg = mdg
        self.discr = discr
        self.swp = pg.Sweeper(mdg)

        self.grad_op = pg.grad(mdg)
        self.curl_op = pg.curl(mdg)

        self.stab = self.stabilization()

    def stabilization(self):
        lumped_ridge_mass = pg.lumped_ridge_mass(self.mdg)
        inv_lumped_peak_mass = pg.lumped_peak_mass(self.mdg)
        inv_lumped_peak_mass.data = 1 / inv_lumped_peak_mass.data
        return (
            (lumped_ridge_mass @ self.grad_op)
            @ inv_lumped_peak_mass
            @ (lumped_ridge_mass @ self.grad_op).T
        )

    def solve(self, f, g, data):
        q_f = self.step1(f)
        r = self.step2(q_f, g, data)
        return self.step3(q_f, r, data)

    def step1(self, f):
        return self.swp.sweep(f)

    def step2(self, q_f, g):
        face_mass = pg.face_mass(self.mdg, discr=self.discr)
        A = pg.ridge_stiff(self.mdg) + self.stab
        rhs = self.curl_op.T @ (g - face_mass @ q_f)

        R = self.create_restriction(A)
        sol = sps.linalg.spsolve(R @ A @ R.T, R @ rhs)
        return R.T @ sol

    def step3(self, q_f, r):
        q = q_f + self.curl_op @ r

        face_mass = pg.face_mass(self.mdg, discr=self.discr)
        p = self.swp.sweep_transpose(face_mass @ q)

        return q, p

    def create_restriction(self, A):
        n = self.curl_op.shape[1]

        # If the constants are in the kernel, then we are in the 2D Dirichlet case
        if np.allclose(A @ np.ones(n), 0):
            # Create restriction that removes last dof
            R = sps.eye(n - 1, n)

        else:  # All other cases
            # Create restriction that removes tip dofs
            R = pg.remove_tip_dofs(self.mdg, 2)

        return R

    def copy(self):
        copy_self = HodgeSolver.__new__(HodgeSolver)

        for str, attr in self.__dict__.items():
            copy_self.__setattr__(str, attr)

        return copy_self
