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


def fracture_grid(mesh_args=None):
    """Mixed-dimensional grid for a domain with a set of fractures."""

    bbox = {"xmin": 0, "xmax": 1, "ymin": 0, "ymax": 1, "zmin": 0, "zmax": 1}
    domain = pp.Domain(bbox)

    num_div = 12

    f0 = generate_fracture(8, 0.5, [0.25, 0.5, 0.45], [1, -1, 1])
    f1 = generate_fracture(num_div, 0.5, [0.75, 0.5, 0.5], [-1, -1, 1])
    f2 = generate_fracture(num_div, 0.25, [0.5, 0.6, 0.65], [0, 0.5, 1])
    f3 = generate_fracture(num_div, 0.35, [0.1, 0.25, 0.35], [0.1, 0, 1])
    f4 = generate_fracture(num_div, 0.15, [0.8, 0.45, 0.5], [1, -1, 1])

    network = pp.create_fracture_network([f0, f1, f2, f3, f4], domain)

    mdg = network.mesh(mesh_args)
    pg.convert_from_pp(mdg)

    return mdg


def generate_fracture(num_div, radius, center, rotation_vector):
    theta = np.linspace(0, 2 * np.pi, num_div + 1)[:-1]
    pts = np.array(
        [np.cos(theta), np.sin(theta), np.zeros_like(theta), np.ones_like(theta)]
    )

    # compute the transformation matrices
    S = scaling(radius, homogeneous=True)
    T = translation(center)
    R = rotation(rotation_vector, homogeneous=True)

    new_pts = T @ R @ S @ pts

    return pp.PlaneFracture(new_pts[:-1])


def rotation(vect, homogeneous=False):
    # Rotation matrix for a vector
    d = np.linalg.norm(vect)
    dx, dy, dz = vect

    dxy = dx * dx + dy * dy
    r0 = (dx * dx * dz / d + dy * dy) / dxy
    r1 = dx * dy * (dz / d - 1) / dxy
    r2 = (dy * dy * dz / d + dx * dx) / dxy

    mat = np.array([[r0, r1, -dx / d], [r1, r2, -dy / d], [dx / d, dy / d, dz / d]])
    return to_homogeneous(mat) if homogeneous else mat


def scaling(vect, homogeneous=False):
    # Scaling matrix
    mat = np.diag(vect * np.ones(3))
    return to_homogeneous(mat) if homogeneous else mat


def translation(vect):
    mat = np.array(
        [[1, 0, 0, vect[0]], [0, 1, 0, vect[1]], [0, 0, 1, vect[2]], [0, 0, 0, 1]]
    )
    return mat


def to_homogeneous(vect):
    vect = np.vstack((vect, np.zeros(3)))
    col = np.array([[0, 0, 0, 1]]).T
    return np.hstack((vect, col))
