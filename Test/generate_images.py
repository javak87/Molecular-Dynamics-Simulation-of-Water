import itertools
import heapq
import numpy as np

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

    postate = np.arange(1, 7).reshape(2,3)
    print (postate)
    #np.array([0.2, 0.3, 0.4])
    #np.arange(12).reshape(4,3)

    image = GenerateImages(postate, bounds, 10)
    resul = image.expand_images()
    print (resul)