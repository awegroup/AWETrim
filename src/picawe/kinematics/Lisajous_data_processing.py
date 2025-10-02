from picawe.kinematics.RI_data_processing import RI_data_processing as ribdata
import numpy as np
import matplotlib.pyplot as plt

class Lisajous_data_processing(ribdata):
    def __init__(self, file_path_cycle=None, file_path_full=None, cyc_idx=0):
        super().__init__(file_path_cycle=file_path_cycle, file_path_full=file_path_full, cyc_idx=cyc_idx)

        self.az_RO = self.az_cyc[:self.ri_idx0]
        self.el_RO = self.el_cyc[:self.ri_idx0]

        self.daz_RO = np.gradient(self.az_RO)
        self.del_RO = np.gradient(self.el_RO)

        self.x_RO, self.y_RO, self.z_RO = self.x_cyc[:self.ri_idx0], self.y_cyc[:self.ri_idx0], self.z_cyc[:self.ri_idx0]

        self.ID_lisajous_start_end()

    # 2D Plot of the Reel-Out Path (Azimuth vs Elevation)
    def plot_reel_out_path2D(self):

        plt.figure()
        plt.plot(self.az_Lisajous, self.el_Lisajous)
        plt.scatter(self.Lisajous_p0[0][0], self.Lisajous_p0[0][1], color='green', label='Lisajous Start Point')
        plt.scatter(self.Lisajous_pf[0][0], self.Lisajous_pf[0][1], color='red', label='Lisajous End Point')
        plt.xlabel('Azimuth (rad)')
        plt.ylabel('Elevation (rad)')
        plt.title('Kite Path During Reel-Out Phase')
        plt.legend()
        plt.show()
    
    # 3D Plot of the Reel-Out Path (X, Y, Z)
    def plot_reel_out_path3D(self):

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot(self.x_RO, self.y_RO, self.z_RO, label='Reel-Out Path')
        ax.scatter(self.x_RO[self.Lisajous_idx0], self.y_RO[self.Lisajous_idx0], self.z_RO[self.Lisajous_idx0], color='green', label='Lisajous Start Point')
        ax.scatter(self.x_RO[self.Lisajous_idxf], self.y_RO[self.Lisajous_idxf], self.z_RO[self.Lisajous_idxf], color='red', label='Lisajous End Point')
        ax.scatter(self.x_RO[0], self.y_RO[0], self.z_RO[0], color='blue', label='RO Start Point')
        ax.scatter(self.x_RO[-1], self.y_RO[-1], self.z_RO[-1], color='orange', label='RO End Point')
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        ax.set_title('3D Kite Path During Reel-Out Phase')
        ax.legend()
        plt.show()

    def ID_lisajous_start_end(self):
        # Truncate the RO data to find the CST Lissajous pattern parameters
        
        self.Lisajous_p0 = []
        self.Lisajous_pf = []

        start_point_found = False

        for i in range(len(self.daz_RO)):

            if self.daz_RO[i] > 0 and self.del_RO[i] > 0 and self.az_RO[i] <= 0.01 and self.az_RO[i] >= -0.01 and self.el_RO[i] <= 0.5 and not start_point_found:
                self.Lisajous_idx0 = i
                self.Lisajous_p0.append((self.az_RO[i], self.el_RO[i]))
                start_point_found = True

            elif self.daz_RO[i] > 0 and self.del_RO[i] > 0 and self.az_RO[i] <= 0.01 and self.az_RO[i] >= -0.01 and self.el_RO[i] <= 0.5 and start_point_found and i > self.Lisajous_idx0 + 10:
                self.Lisajous_idxf = i
                self.Lisajous_pf.append((self.az_RO[i], self.el_RO[i]))
        
        self.az_Lisajous = self.az_RO[self.Lisajous_idx0:self.Lisajous_idxf+1]
        self.el_Lisajous = self.el_RO[self.Lisajous_idx0:self.Lisajous_idxf+1]
        
        if start_point_found == False:
            raise ValueError("No valid start point found for the CST Lissajous pattern in the reel-out data.")

if __name__ == "__main__":
    full_path = "/home/theophile/src/Simulation_Results/trial_Uri_valid_2/ProtoLogger_csv/2025-09-25_11-48-58_ProtoLogger.csv"
    cycle_path = "/home/theophile/src/Simulation_Results/trial_Uri_valid_2/cycles/cycle_data_sheet_lines.csv"

    obj = Lisajous_data_processing(file_path_cycle=cycle_path, file_path_full=full_path, cyc_idx=0)
    obj.plot_reel_out_path2D()
    obj.plot_reel_out_path3D()