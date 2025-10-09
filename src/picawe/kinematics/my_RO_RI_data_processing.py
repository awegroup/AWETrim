from picawe.kinematics.my_Lisajous_data_processing import Lisajous_data_processing
import numpy as np
import matplotlib.pyplot as plt

''' Going from RO to RI, so we are looking at the points from the end of the consistent reel-out phase to the start of RI phase. '''

class RO_RI_data_processing(Lisajous_data_processing):
    def __init__(self, file_path_full, file_path_cycle, cyc_idx=0):
        super().__init__(file_path_full=file_path_full, file_path_cycle=file_path_cycle, cyc_idx=cyc_idx)

        self.RO_RI_idx0 = None  # Start index of the RO to RI transition (relative to cycle start)
        self.RO_RI_idxf = self.ri_idx0  # End index of the RO to RI transition (relative to cycle start)

        # This class has to find the start index of the RO to RI transition

    def find_start_RO_RI_idx(self):
        ''' Create a logic that finds the start index of the RO to RI transition
        Start of RO to RI transition is when the trajectory reaches a point of 
        minimum altitude still along a consistent Lisajous shape (bottom right of a Lisajous shape). '''
        
        for i in range(self.Lisajous_idxf, self.ri_idx0):
            if self.az_cyc[i] > 0.1 and self.del_RO[i] > 0 and self.daz_RO[i] < 0 and self.el_cyc[i] < 0.25:
                self.RO_RI_idx0 = i
                break
        
        if self.RO_RI_idx0 is None:
            raise ValueError("No valid start point found for the RO to RI transition in the reel-out data.")
        
        self.RO_RI_p0_sph = (self.az_cyc[self.RO_RI_idx0], self.el_cyc[self.RO_RI_idx0])
        self.RO_RI_pf_sph = (self.az_cyc[self.RO_RI_idxf], self.el_cyc[self.RO_RI_idxf])

        self.RO_RI_p0_cart = (self.x_cyc[self.RO_RI_idx0], self.y_cyc[self.RO_RI_idx0], self.z_cyc[self.RO_RI_idx0])
        self.RO_RI_pf_cart = (self.x_cyc[self.RO_RI_idxf], self.y_cyc[self.RO_RI_idxf], self.z_cyc[self.RO_RI_idxf])

        self.RO_RI_crs0 = self.crs_cyc[self.RO_RI_idx0]
        self.RO_RI_crsf = self.crs_cyc[self.RO_RI_idxf]

        self.RO_RI_v0 = (self.dx_cyc[self.RO_RI_idx0], self.dy_cyc[self.RO_RI_idx0], self.dz_cyc[self.RO_RI_idx0])
        self.RO_RI_vf = (self.dx_cyc[self.RO_RI_idxf], self.dy_cyc[self.RO_RI_idxf], self.dz_cyc[self.RO_RI_idxf])

        self.RO_RI_sph = (self.az_cyc[self.RO_RI_idx0:self.RO_RI_idxf+1], self.el_cyc[self.RO_RI_idx0:self.RO_RI_idxf+1])
        self.RO_RI_cart = (self.x_cyc[self.RO_RI_idx0:self.RO_RI_idxf+1], self.y_cyc[self.RO_RI_idx0:self.RO_RI_idxf+1], self.z_cyc[self.RO_RI_idx0:self.RO_RI_idxf+1])

    def plot_RO_RI_path3D(self):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot(self.x_cyc, self.y_cyc, self.z_cyc, label='Cycle Path')
        ax.plot(self.x_cyc[self.RO_RI_idx0:self.RO_RI_idxf], self.y_cyc[self.RO_RI_idx0:self.RO_RI_idxf], self.z_cyc[self.RO_RI_idx0:self.RO_RI_idxf], color='orange', linestyle='--', label='RO to RI Transition')
        ax.scatter(*self.RO_RI_p0_cart, color='g', label='RO Start')
        ax.scatter(*self.RO_RI_pf_cart, color='r', label='RI End')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.legend()
        plt.show()

if __name__ == "__main__":
    full_path = "/home/theophile/src/Simulation_Results/trial_Uri_valid_2/ProtoLogger_csv/2025-09-25_11-48-58_ProtoLogger.csv"
    cycle_path = "/home/theophile/src/Simulation_Results/trial_Uri_valid_2/cycles/cycle_data_sheet_lines.csv"

    obj = RO_RI_data_processing(file_path_full=full_path, file_path_cycle=cycle_path, cyc_idx=0)
    obj.find_start_RO_RI_idx()
    obj.plot_RO_RI_path3D()