import numpy as np
import porepy as pp
import pygeon as pg


def create_data(mdg, keyword):
    p_bc = lambda x: x[1]
    RT0 = pg.RT0("bc_val")
    aperture = 1e-4

    for sd, data in mdg.subdomains(return_data=True):
        # Set up parameters
        specific_volumes = np.power(aperture, mdg.dim_max() - sd.dim)
        perm = pp.SecondOrderTensor(specific_volumes * np.ones(sd.num_cells))

        b_faces = sd.tags["domain_boundary_faces"]
        bc_val = -RT0.assemble_nat_bc(sd, p_bc, b_faces)

        param = {
            "second_order_tensor": perm,
            "bc_values": bc_val,
            "specific_volumes": specific_volumes,
        }
        pp.initialize_default_data(sd, data, keyword, param)

    for mg, data in mdg.interfaces(return_data=True):
        kn = 1 / (aperture / 2)
        param = {"normal_diffusivity": kn, "aperture": aperture}
        pp.initialize_data(mg, data, keyword, param)
