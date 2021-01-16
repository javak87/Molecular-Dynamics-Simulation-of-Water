import numpy as np
import matplotlib.pyplot as plt
plt.style.use('seaborn-dark')
import animation

#Simulation Parameters
grid_dims = np.array([2,2,2]) 
#num_molecules = np.prod(grid_dims)
timesteps = 5
num_molecules = 2
num_atoms = 3 #number of atoms in one molecule
dims = 3 #number of spatial dimensions 
# (the rest of the code is not written to handle anything but two dimensions, though you could change that)
# bond angle of H-O-H
bond_angle = 104.5*np.pi/180 # radians
bond_length = 9.7e-11 #[m] the distance between H and O in H2O
# http://hyperphysics.phy-astr.gsu.edu/hbase/molecule/vibrot.html#c3
side_lengths = [2*grid_dims[0]*bond_length, 2*grid_dims[1]*bond_length, 2*grid_dims[2]*bond_length]
# Determines the spacing of the molecules in x and y direction
k_spring = 458.9 # bond strength of H2O
#http://hyperphysics.phy-astr.gsu.edu/hbase/molecule/vibrot.html#c2




def make_liquid(num_molecules, dims, bond_length, bond_angle, grid_dims, side_lengths):
    """
    pos lets one atom spawn a bond_length away from the other atom, 
    in a uniformly distributed random direction

    numpy.random.normal(loc=0.0, scale=1.0, size=None) 
    loc: Mean (“centre”) of the distribution.
    scale:  Standard deviation of the distribution. Must be non-negative.
    size: output shape

    """
    pos = np.random.normal(0,1,size=(num_molecules,dims)) 
    
    
    """
    create an array to hold the x and y values of the hydrogen atom positions, 
    relative to the molecule's location
    
    default norm in linalg.norm
    matrix norm of an matrix defined as the square root of 
    the sum of the absolute squares of its elements
    """
    r = np.linalg.norm(pos, axis = 1)  


    #determine the length of the random spacing between the atoms
    pos = pos/r.reshape(num_molecules,1)*bond_length 
    #scale the positions by the random spacing length to normalize
    """Note: reshape allows the array of shape (num_molecules, dims) to be
    divided by an array (tuple) with shape (num_molecules, )
    by giving it shape (num_molecules, 1), both the x and y component will then be divided by r. """
    #print(pos.shape)
    
    molecule = np.zeros((num_molecules, dims)) 
    #array to hold the positions of the different atoms
    

    """
    np.meshgrid()
    """
    #initialize the molecules in a square grid, with random rotation
    xx = np.linspace(-side_lengths[0]/2, side_lengths[0]/2, grid_dims[0]) #x dimension for meshgrid
    yy = np.linspace(-side_lengths[1]/2, side_lengths[1]/2, grid_dims[1]) #y dimension for meshgrid
    zz = np.linspace(-side_lengths[2]/2, side_lengths[2]/2, grid_dims[2]) #z dimension for meshgrid
    mesh_x, mesh_y, mesh_z = np.meshgrid(xx,yy,zz,sparse=True) #creates three meshgrids 
    
    #ravel flattens the 3d array into a 1d array
    molecule[:,0] = np.ravel(mesh_x) # Sets the x-position 
    molecule[:,1] = np.ravel(mesh_y) # Sets the y-position
    molecule[:,2] = np.ravel(mesh_z) # Sets the z-position
    # The centre of the molecules will be in a grid formation
    
    def z_rotation(vector,theta):
        """Rotates 3-D vector around z-axis"""
        R = np.array([[np.cos(theta), -np.sin(theta),0],[np.sin(theta), np.cos(theta),0],[0,0,1]])
        pos2 = vector
        for i in range(num_molecules):
            pos2[i] = np.dot(R,pos2[i,:])
        return pos2

    H1 = molecule + pos 
    H2 = molecule + z_rotation(pos, bond_angle)
    O = molecule
    
    return H1, H2, O





position_H1 = np.zeros((timesteps, num_molecules, dims))
position_H2 = np.zeros((timesteps, num_molecules, dims))
position_O = np.zeros((timesteps, num_molecules, dims))

for index in range(position_H1.shape[0]):
    position_H1[index], position_H2[index], position_O[index] = make_liquid(num_molecules, dims, bond_length, bond_angle, grid_dims, side_lengths) 


data = [position_H1, position_H2, position_O]
animation.main(data)