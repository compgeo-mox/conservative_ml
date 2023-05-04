import numpy as np
from scipy.stats import qmc

import porepy as pp
import pygeon as pg


class Sampler:
    def __init__(self, hs):
        self.keyword = hs.discr.keyword

        self.hs = hs
        self.mdg = hs.mdg

        self.cell_mass = pg.cell_mass(self.mdg)
        self.face_mass = pg.face_mass(self.mdg)

    def get_f(self, mu):
        def source(x):
            return np.sin(2 * np.pi * mu[0] * x[0]) * np.sin(2 * np.pi * mu[1] * x[1])

        f = self.cell_mass @ pg.PwConstants(self.keyword).interpolate(
            self.mdg.subdomains()[0], source
        )
        return f

    def get_g(self):
        return np.zeros(self.mdg.num_subdomain_faces())

    def get_q_f(self, mu):
        f = self.get_f(mu)
        return self.hs.step1(f)

    def get_r(self, mu):
        g = self.get_g()
        q_f = self.get_q_f(mu)
        return self.hs.step2(q_f, g)

    def generate_set(self, num, seed=None):
        lhc = qmc.LatinHypercube(d=2, seed=seed)
        mu_samples = lhc.random(num)

        l_bounds = [0, 0]
        u_bounds = [4, 4]
        mu_samples = qmc.scale(mu_samples, l_bounds, u_bounds)

        for mu in mu_samples:
            yield mu, self.get_r(mu)

    def compute_loss_r(self, r_true, r):
        diff = self.hs.curl_op @ (r_true - r)
        return np.sqrt(diff @ self.face_mass @ diff)

    def compute_loss_p(self, mu, r_true, r):
        q_f = self.get_q_f(mu)

        _, p_true = self.hs.step3(q_f, r_true)
        _, p = self.hs.step3(q_f, r)
        diff = p_true - p

        return np.sqrt(diff @ self.cell_mass @ diff)

    def compute_loss(self, mu, r_true, r, weights=[0.5, 0.5]):
        loss_r = self.compute_loss_r(r_true, r)
        loss_p = self.compute_loss_p(mu, r_true, r)

        return weights[0] * loss_r + weights[1] * loss_p
