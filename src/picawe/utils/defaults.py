import numpy as np
DEFAULT_BOUNDS = {
    "tension_tether_ground": [0, 1e7],
    "input_steering": [-10, 10],
    "s_dot": [0, 50],
    "s_ddot": [-50, 50],
    "speed_tangential": [0, 150],
    "angle_roll": [-np.pi/4, np.pi/4],
    "timeder_angle_course": [-np.pi, np.pi],
    "angle_pitch": [-np.pi/4, np.pi/4],
    "angle_yaw": [-np.pi/4, np.pi/4],
    "angle_elevation": [0, np.pi],

}

PLOT_LABELS = {
    'acceleration_normal': '$a_n$ [$m/s^2$]',
    'acceleration_radial': '$a_r$ [$m/s^2$]',
    'acceleration_tangential': '$a_{\\tau}$ [$m/s^2$]',
    'angle_course': '$\\chi$ [$^\\circ$]',
    'angle_flight_path_aerodynamic': '$\\gamma_a$ [$^\\circ$]',
    'angle_heading_aerodynamic': '$\\chi_a$ [$^\\circ$]',
    'steering_input': '$u_s$',
    'dchi_ds': '$d\\chi / ds$',
    'distance_radial': '$r$ [m]',
    'force_tether_talmar': '$F_{t, \\mathrm{Talmar}}}$',
    'phi_unwrapped': '$\\phi_a$ [$^\\circ$]',
    'ratio_kinematic': '$\\kappa$',
    'ratio_tether': '$\\frac{F_t}{F_{t, \\mathrm{Talmar}}}}$',
    's': 's [-]',
    's_dot': '$\\dot{s}$ [-]',
    's_ddot': '$\\ddot{s}$ [-]',
    'speed': '$v_k$ [m/s]',
    'speed_radial': '$v_r$ [m/s]',
    'speed_tangential': '$v_\\tau$ [m/s]',
    'speed_wind_true': '$v_{w,true}$ [m/s]',
    'speed_wind_apparent': '$v_{w,app}$ [m/s]',
    'tension_tether_ground': '$F_{t,g}$ [N]',
    'tension_kite': '$F_{t,k}$ [N]',
    'power_ground': '$P_g$ [W]',
    'kite_angle_of_attack': '$\\alpha$ [$^\\circ$]',
    'time': '$t$ [s]',
    'timeder_angle_course': '$\\dot{\\chi}$ [$^\\circ$/s]',
    'timeder_speed_radial': '$\\dot{v}_r$ [$m/s^2$]',
    'timeder_speed_tangential': '$\\dot{v}_\\tau$ [$m/s^2$]',
    'x': 'x [m]',
    'y': 'y [m]',
    'z': 'z [m]',
    "input_steering": "$u_s$",
}

PLOT_PARAMETERS = [
                'speed_tangential', 'speed_radial',"input_steering", "tension_tether_ground"
            ]