import casadi as ca

def transformation_AZR_from_W(azimuth, elevation):
    phi = azimuth
    beta = elevation
    # Create the transformation matrix
    transformation = ca.vertcat(
        ca.horzcat(-ca.sin(phi), ca.cos(phi), 0),
        ca.horzcat(-ca.sin(beta) * ca.cos(phi), -ca.sin(beta) * ca.sin(phi), ca.cos(beta)),
        ca.horzcat(ca.cos(beta) * ca.cos(phi), ca.cos(beta) * ca.sin(phi), ca.sin(beta))
    )
    return transformation

def transformation_C_from_AZR(chi):
    # Directly create the transformation matrix using CasADi
    transformation = ca.vertcat(
        ca.horzcat(ca.sin(chi), ca.cos(chi), 0),
        ca.horzcat(-ca.cos(chi), ca.sin(chi), 0),
        ca.horzcat(0, 0, 1)
    )
    return transformation

def transformation_C_from_A(theta_a, chi_a, roll):
    # Define the Pitch matrix
    Pitch = ca.vertcat(
        ca.horzcat(ca.cos(theta_a), 0, ca.sin(theta_a)),
        ca.horzcat(0, 1, 0),
        ca.horzcat(-ca.sin(theta_a), 0, ca.cos(theta_a))
    )

    # Define the Yaw matrix
    Yaw = ca.vertcat(
        ca.horzcat(ca.cos(chi_a), -ca.sin(chi_a), 0),
        ca.horzcat(ca.sin(chi_a), ca.cos(chi_a), 0),
        ca.horzcat(0, 0, 1)
    )

    # Define the Roll matrix
    Roll = ca.vertcat(
        ca.horzcat(1, 0, 0),
        ca.horzcat(0, ca.cos(roll), -ca.sin(roll)),
        ca.horzcat(0, ca.sin(roll), ca.cos(roll))
    )

    # Compute the transformation matrix T using the @ operator
    T = Yaw @ Pitch @ Roll

    return T

def transformation_C_from_K(pitch, roll, yaw = 0):

    # Define the Pitch matrix
    Pitch = ca.vertcat(
        ca.horzcat(ca.cos(pitch), 0, ca.sin(pitch)),
        ca.horzcat(0, 1, 0),
        ca.horzcat(-ca.sin(pitch), 0, ca.cos(pitch))
    )

    # Define the Yaw matrix
    Yaw = ca.vertcat(
        ca.horzcat(ca.cos(yaw), -ca.sin(yaw), 0),
        ca.horzcat(ca.sin(yaw), ca.cos(yaw), 0),
        ca.horzcat(0, 0, 1)
    )

    # Define the Roll matrix
    Roll = ca.vertcat(
        ca.horzcat(1, 0, 0),
        ca.horzcat(0, ca.cos(roll), -ca.sin(roll)),
        ca.horzcat(0, ca.sin(roll), ca.cos(roll))
    )

    T = Yaw @ Pitch @ Roll
    return T

def transformation_C_from_W(azimuth, elevation, course):
    # Create the transformation matrix
    return transformation_C_from_AZR(course) @ transformation_AZR_from_W(azimuth, elevation)
