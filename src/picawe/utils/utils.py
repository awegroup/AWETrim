import casadi as ca

def skew_symmetric(v):
    return ca.vertcat(
        ca.horzcat(0, -v[2], v[1]),
        ca.horzcat(v[2], 0, -v[0]),
        ca.horzcat(-v[1], v[0], 0)
    )

def calculate_angle_2vec(vector_a, vector_b, reference_vector=None):


    dot_product = ca.dot(vector_a, vector_b)
    magnitude_a = ca.norm_2(vector_a)
    magnitude_b = ca.norm_2(vector_b)

    cos_theta = dot_product / (magnitude_a * magnitude_b)
    angle_rad = ca.arccos(cos_theta)

    return angle_rad