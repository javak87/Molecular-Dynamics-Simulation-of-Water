import matplotlib.animation as animation
import mpl_toolkits.mplot3d.axes3d as p3
import matplotlib.pyplot as plt


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
        scatters1[i]._offsets3d = (data1[iteration][i,0:1], data1[iteration][i,1:2], data1[iteration][i,2:])
        scatters2[i]._offsets3d = (data2[iteration][i,0:1], data2[iteration][i,1:2], data2[iteration][i,2:])
        scatters3[i]._offsets3d = (data3[iteration][i,0:1], data3[iteration][i,1:2], data3[iteration][i,2:])
    return scatters1, scatters2, scatters3

def main(data):
    """
    def main(data, save=False):
    Creates the 3D figure and animates it with the input data.
    Args:
        data (list): List of the data positions at each iteration.
        save (bool): Whether to save the recording of the animation. (Default to False).
    """

    # Attaching 3D axis to the figure
    fig = plt.figure()
    ax = p3.Axes3D(fig)
    
    
    # Initialize scatters
    data1 = data[0]
    scatters1 = [ ax.scatter(data1[0][i,0:1], data1[0][i,1:2], data1[0][i,2:],c='xkcd:pinkish red', s = 50) for i in range(data1[0].shape[0]) ]
    data2 = data[1]
    scatters2 = [ ax.scatter(data2[0][i,0:1], data2[0][i,1:2], data2[0][i,2:],c='xkcd:pinkish red', s = 50) for i in range(data2[0].shape[0]) ]
    data3 = data[2]
    scatters3 = [ ax.scatter(data3[0][i,0:1], data3[0][i,1:2], data3[0][i,2:],c='xkcd:greenish blue', s=250) for i in range(data3[0].shape[0]) ]
    
    # Number of iterations
    iterations = len(data1)

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



