# 16.01.2021
# Author: Florian Anders

import matplotlib.animation as animation
import mpl_toolkits.mplot3d.axes3d as axes3d
import matplotlib.pyplot as plt
import numpy as np
from FileOperation import FileOperation as FileIO


class Animation:

    @staticmethod
    def update_graph(step: int):

        graph._offsets3d = (position_array[step][:][..., 0], position_array[step][:][..., 1], position_array[step][:][..., 2])

    @staticmethod
    def define_colour_array(molecule_count: int, rgb_value_array_oxygen: np.ndarray, rgb_value_array_hydogen: np.ndarray):

        colour_array = np.zeros(shape=(molecule_count * 3, 3))

        for i in range(molecule_count * 3):

            if i % 3 == 0 or i == 0:

                colour_array[i, 0] = rgb_value_array_oxygen[0]
                colour_array[i, 1] = rgb_value_array_oxygen[1]
                colour_array[i, 2] = rgb_value_array_oxygen[2]

            else:

                colour_array[i, 0] = rgb_value_array_hydogen[0]
                colour_array[i, 1] = rgb_value_array_hydogen[1]
                colour_array[i, 2] = rgb_value_array_hydogen[2]

        return colour_array

    @staticmethod
    def define_atom_size_array(molecule_count: int, size_of_hydrogen: float, size_ratio_oxygen_hydrogen: float):

        size_array = np.zeros(shape=(molecule_count * 3))

        for i in range(molecule_count * 3):

            if i % 3 == 0 or i == 0:

                size_array[i] = size_of_hydrogen * size_ratio_oxygen_hydrogen

            else:

                size_array[i] = size_of_hydrogen

        return size_array


if __name__ == "__main__":

    timesteps = 100
    time_interval = 2
    molecule_count = 10

    hdf5_file_name = "data.hdf5"

    FileIO.save_data(timesteps, time_interval, molecule_count, "data.hdf5")
    FileIO.write_hdf5_txt('data.hdf5')

    position_array = np.array(FileIO.extract_data_to_np_array(timesteps, time_interval, molecule_count, hdf5_file_name)[0])

    rgb_array_oxygen = np.array([255, 0, 0])
    rgb_array_hydrogen = np.array([40, 40, 120])
    colours = Animation.define_colour_array(molecule_count, rgb_array_oxygen, rgb_array_hydrogen)

    sizes_of_atoms = Animation.define_atom_size_array(molecule_count, 30, 3)

    fig = plt.figure()

    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel("X - position")
    ax.set_ylabel("Y - position")
    ax.set_zlabel("Z - position")

    x_position_range = (-2, 2)
    y_position_range = (-2, 2)
    z_position_range = (-2, 2)

    ax.set_xlim(x_position_range)
    ax.set_ylim(y_position_range)
    ax.set_zlim(z_position_range)

    title = ax.set_title("3D Test")

    graph = ax.scatter(position_array[0][:][..., 0], position_array[0][:][..., 1], position_array[0][:][..., 2],
                       c=colours/255, s=sizes_of_atoms)

    animation = animation.FuncAnimation(fig, Animation.update_graph, timesteps // time_interval, interval=200, blit=False)

    plt.show()
