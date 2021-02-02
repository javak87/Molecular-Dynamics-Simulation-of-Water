import itertools
import heapq
import numpy as np
from scipy.spatial import distance

def _gen_relevant_images(x, bounds, distance_upper_bound):
    # Map x onto the canonical unit cell, then produce the relevant
    # mirror images
    real_x = x - np.where(bounds > 0.0,
                          np.floor(x / bounds) * bounds, 0.0)
    #m = len(x)
    m= x.shape[0]
    
    xs_to_try = [real_x]
    for i in range(m):
        if bounds[i] > 0.0:
            disp = np.zeros(m)
            disp[i] = bounds[i]
            
            if distance_upper_bound == np.inf:
                xs_to_try = list(
                    itertools.chain.from_iterable(
                        (_ + disp, _, _ - disp) for _ in xs_to_try))
            else:
                extra_xs = []
                # Point near lower boundary, include image on upper side
                if abs(real_x[i]) < distance_upper_bound:
                    extra_xs.extend(_ + disp for _ in xs_to_try)
                    
                # Point near upper boundary, include image on lower side
                if abs(bounds[i] - real_x[i]) < distance_upper_bound:
                    extra_xs.extend(_ - disp for _ in xs_to_try)

                xs_to_try.extend(extra_xs)
    
    return xs_to_try

class GenerateImages :

    def __init__ (self, postate, bounds, distance_upper_bound) :

        self.postate = postate
        self.bounds = bounds
        self.distance_upper_bound = distance_upper_bound

        all_images = np.zeros((1,3))

        for i in range (0, self.postate.shape[0]) :

            images = np.array (_gen_relevant_images(self.postate[i], self.bounds, self.distance_upper_bound))

            all_images = np.vstack((all_images, images))
        self.all_images = all_images[1:]

    def expand_images (self) :

        return self.all_images



if __name__=="__main__":

    bounds = np.array([10, 10, 10])

    postate = np.random.randint(low=1, high=10, size = (100,3))

    P_ch = np.array ([[-0.834], [0.417], [0.417]])

    point_charge = np.zeros((postate.shape[0],1))
    point_charge[0::3] = P_ch[0]
    point_charge[1::3] = P_ch[1]
    point_charge[2::3] = P_ch[2]

    print (point_charge)

    image_obj = GenerateImages(postate, bounds, 10)
    images = image_obj.expand_images()

    print (images.shape)
    charge_array = np.zeros((images.shape[0],1))

    #print (charge_array)
    charge_array[0:int (images.shape[0]/3)] = P_ch[0]
    charge_array[int (images.shape[0]/3):int (2*images.shape[0]/3)] = P_ch[1]
    charge_array[int (2*images.shape[0]/3):int (images.shape[0])] = P_ch[2]

    #print (charge_array)

    r_vector =  postate[:, np.newaxis] - images[np.newaxis, :]
    #print(r_vector.shape)

    charges = point_charge[:, np.newaxis]*charge_array[np.newaxis, :]

    #print(charges.shape)


    sq_dist = r_vector**2
    norm_vector = np.sqrt (np.sum (sq_dist.T, axis=0).T.reshape(sq_dist.shape[0], sq_dist.shape[1], 1))

    #print (vector)
    force = charges*r_vector
    #print (force)
    #print ('force \n', force[0])
    x= force.sum(axis=1)
    print (x.sum(axis=0))

    