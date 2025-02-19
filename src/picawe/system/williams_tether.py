from picawe.system.tether import Tether
import casadi as ca
from picawe.utils.utils import calculate_angle_2vec
import numpy as np



class WilliamsTether(Tether):

    def __init__(self, E=132e9, diameter=0.01, density=970, n_elements=10, elastic=False, cf = 0.01):
        super().__init__(E, diameter, density)
        self.elevation_first_element = ca.SX.sym("elevation_first_element")
        self.azimuth_first_element = ca.SX.sym("azimuth_first_element")
        self.tether_length = ca.SX.sym("tether_length")
        self.n_elements = n_elements
        self.elastic = elastic
        self.cf = cf


    @property
    def force_tether_at_kite(self):
        return self.tether_shape_symbolic()["tether_force_kite"]
    

    def tether_shape_symbolic(self):
        """Calculate tether shape using symbolic expressions"""

        elevation_0 = self.elevation_first_element  
        azimuth_0 = self.azimuth_first_element
        tether_length = self.tether_length
        tension_ground = self.tension_tether_ground
        v_kite = self.velocity_kite
        diameter = self.diameter_tether
        density = self.density_tether
        cdt = self.drag_coefficient_tether

        l_unstrained = tether_length / self.n_elements
        m_s = ca.pi * diameter**2 / 4 * l_unstrained * density

        n_elements = self.n_elements


        uf = self.speed_friction

        omega = self.velocity_rotation_course_frame
        
        tensions = ca.SX.zeros((n_elements, 3))
        tensions[0, 0] = ca.cos(elevation_0) * ca.cos(azimuth_0) * tension_ground
        tensions[0, 1] = ca.cos(elevation_0) * ca.sin(azimuth_0) * tension_ground
        tensions[0, 2] = ca.sin(elevation_0) * tension_ground

        positions = ca.SX.zeros((n_elements + 1, 3))
        if self.elastic:
            l_s = (tension_ground / (self.EA) + 1) * l_unstrained
        else:
            l_s = l_unstrained

        positions[1, 0] = ca.cos(elevation_0) * ca.cos(azimuth_0) * l_s
        positions[1, 1] = ca.cos(elevation_0) * ca.sin(azimuth_0) * l_s
        positions[1, 2] = ca.sin(elevation_0) * l_s

        velocities = ca.SX.zeros((n_elements + 1, 3))
        velocities_apparent_wind = ca.SX.zeros((n_elements + 1, 1))
        angle_va_tether = ca.SX.zeros((n_elements + 1, 1))
        accelerations = ca.SX.zeros((n_elements + 1, 3))

        drag_tether = 0
        stretched_tether_length = l_s  # Stretched
        for j in range(n_elements):  # Iterate over point masses.
            last_element = j == n_elements - 1

            # Determine kinematics at point mass j.
            vj = ca.cross(omega, positions[j + 1, :].T)
            velocities[j + 1, :] = vj
            aj = ca.cross(omega, vj)
            accelerations[j + 1, :] = aj
            delta_p = positions[j + 1, :] - positions[j, :]
            ej = delta_p.T / ca.norm_2(delta_p)  # Axial direction of tether element
            vwj = uf/self.kappa * ca.log(positions[j + 1, 2] / self.z0) * ca.vertcat(1, 0, 0)  

            # Determine flow at point mass j.
            vaj = vwj -vj  # Apparent wind velocity

            vajp = ca.dot(vaj, ej) * ej  # Parallel to tether element
            # TODO: check whether to use vajn
            vajn = vaj - vajp  # Perpendicular to tether element

            vaj_sq = ca.norm_2(vaj)**2

            # Determina angle between  va and tether
            theta = calculate_angle_2vec(vaj, ej)
            cd_t = cdt * ca.sin(theta) ** 3 + ca.pi*self.cf*ca.cos(theta)**3
            cl_t = cdt * ca.sin(theta) ** 2 * ca.cos(theta)-ca.pi*self.cf*ca.sin(theta)*ca.cos(theta)**2
            dir_D = vaj / ca.norm_2(vaj) # Drag direction
            dir_L = -(ej - ca.dot(ej, dir_D) * dir_D) # Lift direction
            dynamic_pressure_area = 0.5 * self.rho * ca.norm_2(vaj) ** 2 * l_unstrained * diameter

            # Save va norm and angle with tether
            velocities_apparent_wind[j + 1, :] = ca.norm_2(vaj)
            angle_va_tether[j + 1, :] = theta

            # Calculate lift and drag using the common factor
            lift_j = dynamic_pressure_area * cl_t * dir_L
            drag_j = dynamic_pressure_area * cd_t * dir_D

            # Determine drag at point mass j.
            if self.n_elements == 1:
                faj = 0.5 * lift_j + 0.5 * drag_j
            elif last_element:
                faj = 0.5 * drag_j + 0.5 * lift_j
            else:
                faj = 0.5 * drag_j + 0.5 * lift_j
            
  
            
            if last_element:
                point_mass = m_s / 2 + self.mass_wing
            else:
                point_mass = m_s
            

            # Use force balance to infer tension on next element.
            fgj = ca.SX.zeros((3))
            fgj[2] = -point_mass * self.g
            if not last_element:
                next_tension = (
                    point_mass * aj + tensions[j, :].T - fgj - faj
                )  # a_kite gave better fit
                tensions[j + 1, :] = next_tension

            if not last_element:
                if self.elastic:
                    l_s = (ca.norm_2(tensions[j + 1, :]) / self.EA + 1) * l_unstrained
                else:
                    l_s = l_unstrained
                stretched_tether_length += l_s
                positions[j + 2, :] = (
                    positions[j + 1, :]
                    + tensions[j + 1, :] / ca.norm_2(tensions[j + 1, :]) * l_s
                )
            elif last_element:
                next_tension = tensions[j, :].T - fgj - faj  # a_kite gave better fit
                aerodynamic_force = next_tension

        va = vwj - vj
        ez_bridle = -tensions[-1, :].T / ca.norm_2(
            tensions[-1, :]
        )  # Bridle direction, pointing down
        ey_bridle = ca.cross(ez_bridle, -va) / ca.norm_2(
            ca.cross(ez_bridle, -va)
        )  # y-axis of bridle frame, perpendicular to va
        ex_bridle = ca.cross(
            ey_bridle, ez_bridle
        )  # x-axis of bridle frame, perpendicular ex and ey
        dcm_b2w = ca.horzcat(ex_bridle, ey_bridle, ez_bridle)

        ez_bridle = -tensions[-1, :].T / ca.norm_2(
            tensions[-1, :]
        )  # Bridle direction, pointing down
        ey_bridle = ca.cross(ez_bridle, v_kite) / ca.norm_2(
            ca.cross(ez_bridle, v_kite)
        )  # y-axis of bridle frame, perpendicular to va
        ex_bridle = ca.cross(
            ey_bridle, ez_bridle
        )  # x-axis of bridle frame, perpendicular ex and ey
        dcm_b2vel = ca.horzcat(ex_bridle, ey_bridle, ez_bridle)

        # Tether frame at the kite
        ez_tether = -tensions[-2, :].T / ca.norm_2(tensions[-2, :])
        ey_tether = ca.cross(ez_tether, -va) / ca.norm_2(ca.cross(ez_tether, -va))
        ex_tether = ca.cross(ey_tether, ez_tether)
        dcm_t2w = ca.horzcat(ex_tether, ey_tether, ez_tether)

        tension_kite = tensions[-1, :].T
        
        cd_tether = drag_tether / (0.5 * self.rho * ca.norm_2(vaj) ** 2 * self.area_wing)



        res = {
            "kite_position": ca.vertcat(positions[-1, :]),
            "tether_force_kite": tension_kite,
        }

        return res

    def objective_function(self, r_kite):
        """Objective function for optimization"""

        return ca.vertcat(r_kite).T - ca.vertcat(self.tether_shape_symbolic()["kite_position"])

    def solve_tether_shape(self, r_kite):
        """Solve tether shape using optimization"""

        f = self.objective_function(r_kite)
        x = ca.symvar(f)
        x_names = [var.name() for var in x]
        nlp = {"x": ca.vertcat(*x), "f": 0, "g": f}
        solver = ca.nlpsol("solver", "ipopt", nlp)

    
        return solver, x_names