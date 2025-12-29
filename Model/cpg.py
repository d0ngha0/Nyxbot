import numpy as np



def rotation_matrix_2d(theta_deg):
    theta_rad = np.deg2rad(theta_deg)
    return np.array([
        [np.cos(theta_rad), -np.sin(theta_rad)],
        [np.sin(theta_rad),  np.cos(theta_rad)]
    ])



def generate_cpg(cpg_out_size, mi):
    start_vector = np.array([1, 1])
    cpg_out = np.zeros((cpg_out_size, 2))
    before_activated = np.zeros((cpg_out_size, 2))
    before_activated[0, :] = 1.01 *start_vector
    cpg_out[0, :] = np.tanh(before_activated[0, :])
    # rotation matrix
    rot = 1.01 * rotation_matrix_2d(mi)
    
    for i in range(cpg_out_size - 1):
        # pre-activated
        before_activated[i+1, :] = rot @ cpg_out[i, :]
        # after-activated
        cpg_out[i + 1, :] = np.tanh(before_activated[i+1, :])

    return np.round(cpg_out,3)
