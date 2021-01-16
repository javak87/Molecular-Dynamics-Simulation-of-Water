import matplotlib.animation as animation
import mpl_toolkits.mplot3d.axes3d as p3
import matplotlib.pyplot as plt
import numpy as np
import FileOperation.FileOperation

def animate_scatters(iteration, data, scatters1, scatters2, scatters3):
    """
    Update the data held by the scatter plot and therefore animates it.
    Args:
        iteration (int): Current iteration of the animation
        data (list): List of the data positions at each iteration.
        scatters (list): List of all the scatters (One per element)
    Returns:
        list: List of scatters (One per element) with new coordinates
    """
    data1, data2, data3 = data
    for i in range(data1[0].shape[0]):
        scatters1[i]._offsets3d = (data1[iteration][i,0:1], data1[iteration][i,1:2], )
        scatters2[i]._offsets3d = (data2[iteration][i,0:1], data2[iteration][i,1:2], data2[iteration][i,2:])
        scatters3[i]._offsets3d = (data3[iteration][i,0:1], data3[iteration][i,1:2], data3[iteration][i,2:])
    return scatters1, scatters2, scatters3


def main():
    """
    def main(data, save=False):
    Creates the 3D figure and animates it with the input data.
    Args:
        data (list): List of the data positions at each iteration.
        save (bool): Whether to save the recording of the animation. (Default to False).
    """

    t_steps = 100
    t_interval = 2

    # t_steps should be a multiple of t_interval, to include the last calculated data;
    # otherwise there will be unnecessary calculations, where the results won't be saved

    mol_count = 5

    # FileOperation.save_data(t_steps, t_interval, mol_count, "data.hdf5")
    # FileOperation.write_hdf5_txt('data.hdf5')

    pos_array = np.array(FileOperation.FileOperation.FileOperation.extract_data_to_np_array(t_steps, t_interval, mol_count, "data.hdf5")[0])
    vel_array = np.array(FileOperation.FileOperation.FileOperation.extract_data_to_np_array(t_steps, t_interval, mol_count, "data.hdf5")[1])

    # Attaching 3D axis to the figure
    fig = plt.figure()
    ax = p3.Axes3D(fig)
    
    
    # Initialize scatters

    array_h1 = np.zeros((len(pos_array[0]), len(pos_array[0][:]), 3))
    array_h2 = np.zeros((len(pos_array[0]), len(pos_array[0][:]), 3))
    array_o = np.zeros((len(pos_array[0]), len(pos_array[0][:]), 3))

    for i in range(len(pos_array[0][:][..., 0])):

        count_o = 0
        count_h1 = 0
        count_h2 = 0

        if i % 3 == 0 or i == 0:

            array_o[][count_o] = [ax.scatter(pos_array[0][i][0], pos_array[0][i][1], pos_array[0][i][2], color=(1, 0, 0, 1), s=250)]
            count_o += 1

        elif i % 2 == 0 and (count_h2 + 1) % 2 != 0:

            scatters_h2[count_h2] = [ax.scatter(pos_array[0][i][0], pos_array[0][i][1], pos_array[0][i][2], c='xkcd:pinkish red', s=50)]
            count_h2 += 1

        elif i % 2 != 0 and (count_h2 + 1) % 2 == 0:

            scatters_h2[count_h2] = [ax.scatter(pos_array[0][i][0], pos_array[0][i][1], pos_array[0][i][2], c='xkcd:pinkish red', s=50)]
            count_h2 += 1

        else:

            scatters_h1[count_h1] = [ax.scatter(pos_array[0][1][..., 0], pos_array[0][i][1], pos_array[0][i][2], c='xkcd:pinkish red', s=50)]
            count_h1 += 1

    print(scatters_o)

    # Number of iterations
    iterations = len(pos_array[:])

    # Setting the axes properties
    ax.set_xlim3d([-3*10**(-10), 3*10**(-10)])
    ax.set_xlabel('X')

    ax.set_ylim3d([-3*10**(-10), 3*10**(-10)])
    ax.set_ylabel('Y')

    ax.set_zlim3d([-3*10**(-10), 3*10**(-10)])
    ax.set_zlabel('Z')

    # Provide starting angle for the view.
    ax.view_init(25, 10)

    animation.FuncAnimation(fig, animate_scatters, iterations, fargs=(data, scatters1, scatters2, scatters3),
                                       interval=50, blit=False)
    
    """
    if save:
        Writer = animation.writers['ffmpeg']
        writer = Writer(fps=200, metadata=dict(artist='Me'), bitrate=-1, extra_args=['-vcodec', 'libx264'])
        ani.save('3d-scatted-animated.mp4', writer=writer)
    """
    plt.show()




    # pos_array[0] gives all atom - positions for the first time step

    # pos_array[0][1] gives the position of the second atom ([1]) for the first time step ([0])

    # pos_array[0][1][2] gives the z-coordinate ([2]) of the second atom ([1]) of the first time step ([0])

    # more examples with slicing:

    # pos_array[0][:][..., 0] will give all x-coordinates from all atom positions of the first time step
    # pos_array[0][:][..., 1] will give all y-coordinates from all atom positions of the first time step
    # pos_array[0][:][..., 2] will give all z-coordinates from all atom positions of the first time step

    # the same holds true for the vel_array

main()