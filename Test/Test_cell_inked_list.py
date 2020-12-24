import numpy as np
from cell_linked_list import CellLinked
"""
This is a test for the "cell_linked_list" file.
Simulation box : 3*3*3
The number of cell per each axis : 3 
In this test, a simulation box is considered to initialize the position state of particles.
Each of them has been specified to a cell.
Before testing the CellLinked class, we know the position state of particles and their corresponding cell.
Firstly,  the primary_state of the particles is initialized.
Secondly, the head array and the list array are created using the CellLinked class.
Thirdly, the given head array and the list array have been used to obtain the secondary_state.
To satisfy all parts of the class,  primary_state  should be equal to secondary_state 
"""
# Initilialize the number of cells per each axis
_no_cells = 3
cells = []
primary_state = np.zeros((1,3))
for i in range (0, _no_cells) :
    for j in range (0, _no_cells) :
        for k in range (0, _no_cells) :

            # Initialize particale postion in x direction between i and i+1
            p0 = np.random.random_sample((np.random.randint(0,5), 1)) + i

            # Initialize particale postion in y direction between j and j+1
            p1 = np.random.random_sample((p0.shape[0], 1)) + j

            # Initialize particale postion in z direction between k and k+1
            p2 = np.random.random_sample((p0.shape[0], 1)) + k

            # Unified position of each particle in x, y , and z direction
            p = np.append(p0, p1, axis=1)
            p = np.append(p, p2, axis=1)

            # add new particle position to primary_state           
            primary_state = np.concatenate((primary_state,p))

            # add amount of particles generated in each step
            cells.append(p0.shape[0])

# omit the zero raw which is generated in the initialization step
primary_state = primary_state [1:, :]

# use CellLinked to generate a head array and a list array of all particles.
initial_parameter = CellLinked(3 , 1 , primary_state)
head_arr, list_arr = initial_parameter ()

assert len (head_arr) == _no_cells*_no_cells*_no_cells , "the length of head array is not equal to number of total cells"
assert max (head_arr) == len (list_arr)-1, "The maximum number in head array is not equal to the length of list array"

"""
In the following section, the primary position state of particles, head array, and list array are known.
Based on the head array and list array, the secondary position state has been obtained.
If head and list array are computed wrongly, secondary position state is not equal to primary position state.
"""

secondary_state = np.zeros((1,3))
for ii in head_arr:
    if ii > -1 :
        cell_state = np.zeros((1,3))
        cell_state = np.concatenate((cell_state, np.array ([primary_state[ii]])))
        while  list_arr[ii] != -1 :
            cell_state = np.concatenate((cell_state, np.array ([primary_state[list_arr[ii]]])))
            ii = list_arr[ii]

        cell_state = np.flip(cell_state[1:, :], 0)
        secondary_state = np.concatenate((secondary_state, cell_state))

assert (secondary_state[1:, :] == primary_state).all() , 'Linked-list is not connected properly'
"""
In this section, neighbor cells are calcualted based on the head array.
To test the _find_neighbor_cells function, consider a cube with 27 cells.
if cell index == 13, all cells in this cube are neighbor.
"""
# Compute the head and list array using simulation box length, cut-off radius, and the position state
position = CellLinked (3, 1, primary_state )
head_arr, list_arr = position ()

# calculate neighbor cells based on head array
neighbor_cells = position._find_neighbor_cells(head_arr)

# compare the 13th element of neighbor_cells (cube center) and the result
result = list(range(0, 27))
assert neighbor_cells [13] == result , 'neighbor indexes or neighbor cells calculate wrongly'

"""
In this section, particles in each cell and neighbor cells are identified. 
Consider a cube with 27 cells, if cell index == 13 (cube center), all cells in this cube are neighbor.
and all particles are neighbors of each other.
"""
# Initialize simulation box lenghth and cut-off radius
box_len=3
r_cut= 1

# Initialize postions of particles inside of the simulation box
postate = box_len * np.random.random_sample((50, 3))

# Compute head array and list array
model = CellLinked (box_len,  r_cut, postate)
head_arr, list_arr = model ()

# calculate neighbor cells based on head array
neighbor_cells = model._find_neighbor_cells(head_arr)

# Extract neighbor cells of the 13th cells (cell index) which all cells are neighbor of each other
neighbor_cells = neighbor_cells[13]

# Extract all particles positioned in the 13th cell and other neighbor cells
temporary_state = model._find_particles_inside_neighbor_cells (list_arr, head_arr, neighbor_cells)

# Check all particles which are calculate by "_find_particles_inside_neighbor_cells" function
mask = np.isin(postate, temporary_state)
assert np.all(mask == True) == True, 'particles in neighbor cells has been indetified wrongly'

print ('\n')
print ('--------------------------------------------------')
print ('the CellLinked class is verified by the given test')
print ('--------------------------------------------------')




    






             



