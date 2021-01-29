import matplotlib.animation as animation
import mpl_toolkits.mplot3d.axes3d as axes3d
import matplotlib.pyplot as plt
import numpy as np
from FileOperation import FileOperation as FileIO
from Animation import Animation

 

class Visualization :

    def __init__ (self, timesteps: int, save_data_itr: int, hdf5_file_name: str, x_upper: float, y_upper:float, z_upper: float ) :

        self.timesteps = timesteps
        self.save_data_itr = save_data_itr
        self.hdf5_file_name = hdf5_file_name
        self.x_upper = x_upper
        self.y_upper = y_upper
        self.z_upper = z_upper
        
        # calculate the number of water molecules in the simulation box
        self.molecule_count= int (FileIO.return_molecule_count(hdf5_file_name))
        self.position_array = np.array(FileIO.extract_data_to_np_array(self.timesteps, self.save_data_itr, self.molecule_count, self.hdf5_file_name)[0])
        print (self.position_array)
        # the plot and figure are created, using the data from the hdf5 file

        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111, projection='3d')
        title = self.ax.set_title("Visualization of water molecules")
        self.ax.set_xlabel("X - position")
        self.ax.set_ylabel("Y - position")
        self.ax.set_zlabel("Z - position")



    def __call__ (self) :



        # create the graph with scatter points and animate the simulation using a repeated call to the update_graph method

        graph = self.graph_spec()

        anim_func = animation.FuncAnimation(self.fig, Animation.update_graph, self.timesteps // self.save_data_itr, interval=1, blit=False)

        # finally show the animation

        plt.show()


    def graph_spec (self) :


        # now the nice slicing of the positional array comes into play;
        # the axes of the 3D-plot are scaled based on the maximum and minimum x, y and z - coordinates of all timesteps,
        # meaning that the -simulated molecules will always stay in the animation box
        # (not to be confused with the simulation box !)

        # as an example:
        # x_position_range = (np.amax(position_array[:][:][..., 0]), np.amin(position_array[:][:][..., 0]))
        # this gives a tuple containing the maximum (np.amax()) and minimum (np.amin()) values
        # of all available x-coordinates from all timesteps

        x_position_range = (np.amin(self.position_array[:][:][..., 0]), np.amax(self.position_array[:][:][..., 0]))
        y_position_range = (np.amin(self.position_array[:][:][..., 1]), np.amax(self.position_array[:][:][..., 1]))
        z_position_range = (np.amin(self.position_array[:][:][..., 2]), np.amax(self.position_array[:][:][..., 2]))
        self.ax.set_xlim(0, self.x_upper)
        self.ax.set_ylim(0, self.y_upper)
        self.ax.set_zlim(0, self.z_upper)

        # for animation, the atom-specific colours are now defined
        rgb_array_oxygen = np.array([180, 225, 0])
        rgb_array_hydrogen = np.array([40, 40, 255])
        colours = Animation.define_colour_array(self.molecule_count, rgb_array_oxygen, rgb_array_hydrogen)

        # for animation, the atom-specific sizes (based roughly on VDW-radii) are now defined

        sizes_of_atoms = Animation.define_atom_size_array(self.molecule_count, 62.7, 2)

        graph = self.ax.scatter(self.position_array[0][:][..., 0], self.position_array[0][:][..., 1], self.position_array[0][:][..., 2],
                        c=colours/255, s=sizes_of_atoms)

        return graph

if __name__=="__main__":

    hdf5_file_name = "data.hdf5"
    timesteps = 4000
    save_data_itr = 5
    x_upper = 10
    y_upper = 10
    z_upper = 10
    viusl_obj = Visualization (timesteps, save_data_itr, hdf5_file_name, x_upper, y_upper, z_upper)
    res = viusl_obj()


