from picawe.kinematics.ReelInBspline_data_processing import ReelInBspline_data_processing
from picawe.kinematics.Lisajous_data_processing import Lisajous_data_processing
import numpy as np
import matplotlib.pyplot as plt

''' Going from RI to RO but because RI ends at the end of the cycle, 
the points that we actually look at here are the RO points at the start of the cycle. '''

class RI_RO_data_processing(Lisajous_data_processing):
    def __init__(self, file_path_full, file_path_cycle, cyc_idx=0):
        super().__init__(file_path_full=file_path_full, file_path_cycle=file_path_cycle, cyc_idx=cyc_idx)
    
        self.RI_RO_idx0 = 0  # Start index of the RI to RO transition (relative to cycle start)
        self.RI_RO_idxf = None  # End index of the RI to RO transition (relative to cycle start)

        # This class has to find the end index of the RI to RO transition

    def find_end_RI_RO_idx(self):
        for i in range(self.Lisajous_idx0):
            if self.az_cyc[i] < 0 and self.del_RO[i] < 0 and self.daz_RO[i] < 0:
                self.RI_RO_idxf = i
                break
        
        if self.RI_RO_idxf is None:
            raise ValueError("No valid end point found for the RI to RO transition in the reel-out data.")

        self.RI_RO_p0_sph = (self.az_cyc[self.RI_RO_idx0], self.el_cyc[self.RI_RO_idx0])
        self.RI_RO_pf_sph = (self.az_cyc[self.RI_RO_idxf], self.el_cyc[self.RI_RO_idxf])

        self.RI_RO_p0_cart = (self.x_cyc[self.RI_RO_idx0], self.y_cyc[self.RI_RO_idx0], self.z_cyc[self.RI_RO_idx0])
        self.RI_RO_pf_cart = (self.x_cyc[self.RI_RO_idxf], self.y_cyc[self.RI_RO_idxf], self.z_cyc[self.RI_RO_idxf])

        self.RI_RO_sph = (self.az_cyc[self.RI_RO_idx0:self.RI_RO_idxf+1], self.el_cyc[self.RI_RO_idx0:self.RI_RO_idxf+1])
        self.RI_RO_cart = (self.x_cyc[self.RI_RO_idx0:self.RI_RO_idxf+1], self.y_cyc[self.RI_RO_idx0:self.RI_RO_idxf+1], self.z_cyc[self.RI_RO_idx0:self.RI_RO_idxf+1])

    def plot_RI_RO_path3D(self):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot(self.x_cyc, self.y_cyc, self.z_cyc, label='Cycle Path')
        ax.plot(self.x_cyc[:self.RI_RO_idxf], self.y_cyc[:self.RI_RO_idxf], self.z_cyc[:self.RI_RO_idxf], color='orange', linestyle='--', label='RI to RO Transition')
        ax.scatter(*self.RI_RO_p0_cart, color='g', label='RI Start')
        ax.scatter(*self.RI_RO_pf_cart, color='r', label='RO End')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.legend()
        plt.show()

if __name__ == "__main__":
    full_path = "/home/theophile/src/Simulation_Results/trial_Uri_valid_2/ProtoLogger_csv/2025-09-25_11-48-58_ProtoLogger.csv"
    cycle_path = "/home/theophile/src/Simulation_Results/trial_Uri_valid_2/cycles/cycle_data_sheet_lines.csv"

    obj = RI_RO_data_processing(file_path_full=full_path, file_path_cycle=cycle_path, cyc_idx=0)
    obj.find_end_RI_RO_idx()
    obj.plot_RI_RO_path3D()