import numpy as np
import scipy.sparse as sps

import pygeon as pg


def generate_samples(sampler, num_samples, step_size, file_name=None, seed=1):
    q0_samples = np.empty(num_samples, dtype=np.ndarray)
    qf_samples = np.empty(num_samples, dtype=np.ndarray)
    mu_samples = np.empty((num_samples, sampler.num_param))
    for idx, (mu, q0, qf) in enumerate(sampler.generate_set(num_samples, seed=seed)):
        mu_samples[idx, :] = mu
        q0_samples[idx] = q0
        qf_samples[idx] = qf

    q0_samples = np.vstack(q0_samples)
    qf_samples = np.vstack(qf_samples)

    S_0 = sampler.S_0(sps.eye(q0_samples.shape[1]))

    if file_name is not None:
        div_op = pg.div(sampler.mdg)
        curl_op = pg.curl(sampler.mdg)
        np.savez(
            file_name,
            curl=curl_op.todense(),
            div=div_op.todense(),
            S_0=S_0.todense(),
            face_mass=sampler.face_mass.todense(),
            cell_mass=sampler.cell_mass.todense(),
            mu=mu_samples,
            q0=q0_samples,
            qf=qf_samples,
            h=step_size,
        )
