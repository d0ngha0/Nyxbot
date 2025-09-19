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

def generate_cpg_output(cpg_out_size, mi_schedule):
    """
    Generate CPG output with time-varying mi parameter.
    
    Args:
        cpg_out_size (int): total number of time steps N.
        mi_schedule (list of tuples): (start, end, mi) intervals where mi is active.
            Each interval is [start, end) over time indices [0, N).
    
    Returns:
        np.ndarray: cpg_out array of shape (cpg_out_size, 2), rounded to 3 decimals.
    """
    alpha = 1.01
    start_vector = np.array([1, 1])
    
    cpg_out = np.zeros((cpg_out_size, 2))
    before_activated = np.zeros((cpg_out_size, 2))
    
    # initialize at time 0
    before_activated[0] = alpha * start_vector
    cpg_out[0] = np.tanh(before_activated[0])
    
    def get_mi(t):
        """Find mi for time t, default to 0 if not in any interval."""
        for (s, e, mi) in mi_schedule:
            if s <= t < e:
                return mi
        return 0.0
    
    for i in range(cpg_out_size - 1):
        mi = get_mi(i)
        rot = alpha * rotation_matrix_2d(mi)
        before_activated[i + 1] = rot @ cpg_out[i]
        cpg_out[i + 1] = np.tanh(before_activated[i + 1])
    
    return np.round(cpg_out, 3)