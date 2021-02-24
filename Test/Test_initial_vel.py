import numpy as np
import numpy.testing as npt
from initial_vel import InitialVelocity

def apply_initial_vel () :

    '''
    In this function, InitialVelocity class is tested. the Center Of Mass Velocity has been calculated.
    because the simulation box is not moving, the Center Of Mass Velocity should be close to zero.
    '''
    O_mass=16
    H_mass=1.00794
    Kb=0.001985875
    temp=298.15
    no_atoms = 10000

    vel_object = InitialVelocity (O_mass, H_mass, Kb, temp)
    all_velocities = vel_object (no_atoms)

    # Calculate MV (momenta)
    all_velocities[::3] *= O_mass
    all_velocities[1::3] *= H_mass
    all_velocities[2::3] *= H_mass

    # compute sum of total mass
    tot_mas = O_mass*(no_atoms/3) + 2*(no_atoms/3)*H_mass

    # calculate Velocity of Center Of Mass
    vel_center_mas = np.sum(all_velocities, axis=0)/(tot_mas)
    npt.assert_almost_equal(np.sum(vel_center_mas, axis=0), [1e-2, 1e-2, 1e-2], decimal=2)

    return (vel_center_mas)

if __name__=="__main__":
    apply_initial_vel ()



