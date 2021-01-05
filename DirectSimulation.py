import numpy as np

class DirectSimulation():
    def __init__(self, r: np.ndarray, v: np.ndarray, N):
        """
            
        """
        self.r = r
        self.v = v
        self.N = N

        print("Positions:",r)
        print("Speed:",v)
        print("Atoms:", mass)


    def Calculus(self, r: np.ndarray, v: np.ndarray, N):
        self.r = r
        self.v = v
        self.N = N

        
        Vext = calc_Vext(r)
        Vint = calc_Vint(r)
        T = calc_T(v)

        H = T + Vext + Vint
        print("Energie:", H)

    def Update(self):
        pass


def randominitial (N):
        global r
        r = np.random.rand(N,3)     #position x,y,z of any Partical
        global v                    
        v = np.random.rand(N,3)     #speed x,y,z of an partical
        global mass    
        mass = np.random.randint(3, size=(N,1)) # 0, 1, 2
    

def calc_Vext (r):
    sigma = 3.166
    epsilon = 1.079
    Vext = np.zeros((N,1))
    Vext = 48 * epsilon * np.power(sigma, 12) / np.power(r, 13) -  24 * epsilon * np.power(sigma, 6) / np.power(r, 7) 
    return Vext


def calc_Vint (r):
    Vint = np.zeros((N,1))
    return Vint   

def calc_T (v):
    T = np.zeros((N,1))
    return T

if __name__=="__main__":
    global N
    N = 5               #number of particals
    randominitial(N)    # Iniatial random values vor postion and speed

    simu = DirectSimulation(r, v, N) 
    simu.Calculus(r,v,N)

    
    
