# Copyright (c) 2023-2026 Oriol Cayon, Delft University of Technology
#
# Licensed under the EUPL, Version 1.2 or - as soon they will be approved by
# the European Commission - subsequent versions of the EUPL (the "Licence");
# You may not use this work except in compliance with the Licence.
# You may obtain a copy of the Licence at:
#
#     https://joinup.ec.europa.eu/software/page/eupl
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the Licence is distributed on an "AS IS" basis,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the Licence for the specific language governing permissions and
# limitations under the Licence.
#
# SPDX-License-Identifier: EUPL-1.2

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


def transformation_Wind_from_W(direction_wind):
    return ca.vertcat(
        ca.horzcat(ca.cos(-direction_wind), -ca.sin(-direction_wind), 0),
        ca.horzcat(ca.sin(-direction_wind), ca.cos(-direction_wind), 0),
        ca.horzcat(0, 0, 1),
    )


def transformation_Wind_from_C(azimuth, elevation, course, direction_wind):
    return transformation_Wind_from_W(direction_wind) @ ca.transpose(
        transformation_C_from_W(azimuth, elevation, course)
    )


def transformation_C_from_Wind(azimuth, elevation, course, direction_wind):
    return ca.transpose(
        transformation_Wind_from_C(azimuth, elevation, course, direction_wind)
    )
