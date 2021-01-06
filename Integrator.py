import numpy as np

class Integrator():

    def __init__ (self, dim, particles, mass, force, pos, vel, acc, dt,) :
        
        self.dim = dim
        self.particles = particles
        self.mass = mass
        self.pos = pos
        self.vel = vel
        self.acc = acc
        self.dt =dt
        self.force = force
        
    def update_pos(self,dim, particles ,pos, vel, acc, dt):
        """
        Update the particle positions.
        
        Parameters
        ----------
        dim: 
            Spatial dimension
        particles:
            Particle number
        pos:
            The positions of the particles
        vel:
            The velocities of the particles
        acc:
            The accelerations of the particles
        dt: float
            The timestep length
        """
       
        for i in range ( 0, dim ):
            for j in range ( 0, particles ):
                pos[i,j] = pos[i,j] + vel[i,j] * dt + 0.5 * acc[i,j] * dt * dt
        return pos

    def update_vel(self, dim, particles, mass, force, vel, acc, dt):
        
        #define reverse mass to make faster calculation (multiplication is faster than division)
        rev_mass = 1/mass

        """
        Update the particle velocities.
        
        Parameters
        ----------
        dim: 
            Spatial dimension
        particles:
            Particle number
        vel:
            The velocities of the particles
        acc:
            The accelerations of the particles
        force:
            Force acting on particles
        dt: 
            The timestep length
        """
        for i in range ( 0, dim ):
            for j in range ( 0, particles ):
                vel[i,j] = vel[i,j] + 0.5 * dt * ( force[i,j] / rev_mass + acc[i,j] )
        
        return vel

    def update_acc(self, dim, particles, mass, force, acc, dt):
        #define reverse mass to make faster calculation (multiplication is faster than division)
        rev_mass = 1/mass
        """
        Update the particle velocities.
        
        Parameters
        ----------
        dim: 
            Spatial dimension
        particles:
            Particle number
        acc:
            The accelerations of the particles
        dt: 
            The timestep length
        """
        for i in range ( 0, dim):
            for j in range ( 0, particles):
                acc[i,j] = force[i,j] * rev_mass
        return acc