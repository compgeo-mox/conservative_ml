import time
import numpy as np
import scipy.sparse as sps

import pygeon as pg


def generate_samples(sampler, num_samples, step_size, file_name=None, seed=None):
    q0_samples = np.empty(num_samples, dtype=np.ndarray)
    qf_samples = np.empty(num_samples, dtype=np.ndarray)
    mu_samples = np.empty((num_samples, sampler.num_param))

    for idx, (mu, q0, qf) in enumerate(sampler.generate_set(num_samples, seed=seed)):
        mu_samples[idx, :] = mu
        q0_samples[idx] = q0
        qf_samples[idx] = qf

    q0_samples = np.vstack(q0_samples)
    qf_samples = np.vstack(qf_samples)

    identity = sps.eye(q0_samples.shape[1])
    S_0 = sampler.S_0(identity)
    S_I_transpose = sampler.sptr.solve_transpose(identity)

    if file_name is not None:
        div_op = pg.div(sampler.mdg)
        curl_op = pg.curl(sampler.mdg)
        Q_coords, R_coords = get_coords(sampler.mdg)
        np.savez(
            file_name,
            curl=curl_op.tocoo(),
            div=div_op.tocoo(),
            S_0=S_0.tocoo(),
            S_I_transpose=S_I_transpose.tocoo(),
            face_mass=sampler.face_mass.tocoo(),
            cell_mass=sampler.cell_mass.tocoo(),
            mu=mu_samples,
            q0=q0_samples,
            qf=qf_samples,
            h=step_size,
            Q_coords=Q_coords,
            R_coords=R_coords,
        )


def get_coords(mdg):
    Q_coords = []
    R_coords = []

    for sd in mdg.subdomains():
        Q_coords.append(sd.face_centers)

        if sd.dim == 3:
            R_coords.append(sd.nodes @ np.abs(sd.ridge_peaks) / 2)
        elif sd.dim == 2:
            R_coords.append(sd.nodes)

    Q_coords = np.hstack(Q_coords).T
    R_coords = np.hstack(R_coords).T

    return Q_coords, R_coords
