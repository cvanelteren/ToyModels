import numpy as np
import pylab as plt
import itertools
from tqdm import tqdm

class Ising:
    def __init__(self, shape = (10,10), temperature = 1/.4):
        self.spins            = np.random.randint(0, 2, size = (shape)) * 2 - 1
        self.interactionField = np.ones(shape = shape) 

        self.beta    = temperature
        self.size    = shape

        # change this to check sides of dimensions and or in
        self.checkNeigbors = np.array(
        list(itertools.product(*[[-1, 0, 1] for i in shape]))[1::2] # only check the cardinal directions
                                      )
    def simulate(self, steps = 10, buffer = .25):
        res = []
        for t in tqdm(range(steps)):
            # do simulate
            idx         = np.random.randint(0, len(self.spins.ravel()))
            coordinate  = np.unravel_index(idx, self.size)
            out         = self.updateStates(coordinate)
            if t % int(buffer * steps) == 0:
                print(t)
                res.append(out)
        self.res = np.array(res)
    def updateStates(self, coordinate):
        """
        Uses MCMC algorithm to sample the states
        """
        # update every spin
        # for coordinate, spin in np.ndenumerate(self.spins):
        neighbors   = self.getNeighbors(coordinate)
        spin        = self.spins[coordinate]
        # TODO: add external field
        J       = np.array([self.interactionField[tuple(neighbor)]  for neighbor in  neighbors])
        spins   = np.array([self.spins[tuple(neighbor)] for neighbor in neighbors])
        tmp     = J * spins
#        nonFlipEnergy = sum(tmp *(spin))
        nonFlipEnergy = sum(tmp * spin)
        flipEnergy    = sum(tmp * spin * -1)
        # print(flipEnergy - nonFlipEnergy)
        a = spin
        if 2 * nonFlipEnergy < 0:
            spin *= -1
        elif np.random.rand() < np.exp(-self.beta * nonFlipEnergy):
            spin *= -1
        self.spins[coordinate] =  spin
        return self.spins.copy() # prevent shallow copy


    def getNeighbors(self, coordinate):
        """
        Gets neighbors with circular bounds.
        """
        neighbors = (np.array(coordinate) + self.checkNeigbors) % self.size
        return neighbors

shape = (64,64)
steps = 100000
i = Ising(shape = shape)
i.simulate(steps, buffer = .25)
columns = len(i.res) // 2
# %%  visualize
rows    = len(i.res) // columns
fig, ax = plt.subplots(rows, columns)
for x, y in zip(ax.ravel(), i.res):
    x.imshow(y, aspect = 'auto')
plt.show()

