import numpy as np, matplotlib.pyplot as plt, time
from scipy.special import softmax
from matplotlib import style
style.use('seaborn-poster')
import matplotlib
import matplotlib.pylab as pylab
params = {'legend.fontsize': 'x-large',
          'figure.figsize': (15, 5),
         'axes.labelsize': 40,
         'axes.labelpad' : 20,
         'axes.titlesize':'x-large',
         'xtick.labelsize':'x-large',
         'ytick.labelsize':'x-large'}
plt.rcParams.update(params)
np.random.seed(1234)
plt.rc
class Mouse:
    def __init__(self,\
            gamma = .9975,\
            epsilon = .01,\
            beta    = 2,\
            nDirections = 8,\
            nNeurons = 493,\
            sigmaPC = .16,\
            sigmaHC = np.pi / 2,\
            speed = 0.3,\
            dt = .1,
            momentum = .9):

        # assign input to properties
        for name, var in locals().items():
            setattr(self, name, var)
        # SETUP ACTOR
        # setup directions
        r = np.arange(0, nDirections) * 2 * np.pi / nDirections
        self.dirs = r

        # SETUP CRITIC
        self.weights = np.zeros(self.nNeurons)
        # give place cells random position?

        # create regular grid of place cells
        radii = np.random.uniform(0, 1, size = nNeurons)
        theta = np.random.uniform(0, 2 * np.pi, size = nNeurons)


        self.placeCellPositions = (np.asarray([np.cos(theta), np.sin(theta)]) * np.sqrt(radii)).T
    def firingRatePC(self, currentPosition):
        """
        Computes the firing rate of placecells of the grid
        """

        delta = currentPosition - self.placeCellPositions
        firingRates = np.exp(-(delta**2).sum(axis = -1)/ (2 * self.sigmaPC**2))
        return firingRates

    def firingRateHC(self, currentAngle):
        """
        returns firing rate of the head direction cells
        """

        # compute difference centered at the current angle
        delta = currentAngle - self.dirs
        return np.exp(-delta**2 / (2 * self.sigmaHC**2))

    def addMomentum(self, newDirection):
        """
        Adds momentum to the old direction
        correct for wrapping of phase
        """
        return ((1 - self.momentum) * newDirection+ \
    self.momentum * self.direction) % (2 * np.pi) - np.pi

    def updatePosition(self, currentPosition, probs):
        # sample new direction
        idx = np.random.choice(self.nDirections, p = probs)
        headDirection = self.dirs[idx]

        # print(idx, headDirection)
        # headDirection = np.random.choice(self.dirs, p = probs)
        # headDirection = self.addMomentum(self.dirs[idx])
        proposition   = np.array([np.cos(headDirection), np.sin(headDirection)])

        newPos = currentPosition + proposition * self.speed * self.dt
        # check if mouse is out of bounds
        radius = (newPos**2).sum()
        counter = 0
        while radius > 1:
            # headDirection += np.pi
            idx = np.random.choice(self.nDirections)
            headDirection = self.dirs[idx]
            # and move him back inside TODO: replace
            proposition = np.array([np.cos(headDirection), np.sin(headDirection)])
            newPos = currentPosition + proposition * self.speed * self.dt
            radius = (newPos**2).sum()
            counter += 1
            if counter > 100:
                print(proposition, probs)
        # update the old head direction
        self.direction = headDirection
        return newPos, idx



# construct a grid
nGrid = 25
r  = np.linspace(-1, 1, nGrid)
X, Y = np.meshgrid(r, r)

coords = np.asarray([X.ravel(), Y.ravel()]).T

# print(coords.shape, X.shape)


# platform location plus size
platformTheta =  0
platformRadius= .5
platformLocation =  platformRadius * \
            np.asarray([np.cos(platformTheta), np.sin(platformTheta)])
threshold = .05
from mpl_toolkits.mplot3d import Axes3D
def simulate(nTrials = 100, nTime = 10000, dt = .1):
    mouse = Mouse()
    weightsPerTrial = np.zeros((nTrials, mouse.nNeurons, mouse.nDirections))

    # weights from place cell to action cells
    actionWeights = np.zeros((mouse.nNeurons, mouse.nDirections, mouse.nDirections))
    criticWeights = np.zeros((mouse.nNeurons, mouse.nDirections))
    positions = np.empty((nTrials, nTime, 2))

    # make plot
    fig = plt.figure(figsize = (15, 30), constrained_layout = 1)
    gs = fig.add_gridspec(ncols = 2, nrows = 1, width_ratios = [1, 1.5])
    ax = fig.add_subplot(gs[0])
    ax.scatter(*platformLocation, marker = '*', s = 5000)
    tmp = np.linspace(0, np.pi * 2)
    ax.plot(np.cos(tmp), np.sin(tmp))
    props = dict(xlabel = 'x', ylabel = 'y')
    ax.set(**props)
    ax.axis('square')

    h, = ax.plot([0], [0])
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.scatter(*mouse.placeCellPositions.T, alpha = .1)

    ## add critic field
    right = fig.add_subplot(gs[1], projection = '3d')

    mainax = fig.add_subplot(111, frameon = False,\
    xticks = [], yticks = [])
    global coords
    # coords = coords.reshape(nGrid, nGrid, -1)
    Z   = np.zeros(X.shape[:2])
    right.plot_surface(X, Y, Z)
    right.set_zlim(0, 1)
    props['zlabel'] = 'Critic score'
    # criticContour = right.imshow(Z)
    right.set(**props)
    fig.subplots_adjust(wspace = 0)
    fig.show()

    goals = 0
    buffer = np.ma.array(np.empty((nTime, 2)))
    actionFiringRate = np.zeros(mouse.nDirections)
    buffer[nTime - 1] = np.ma.masked

    for trial in range(nTrials):
        thetaStart = np.random.uniform(0, 2*np.pi)
        start = np.asarray([np.cos(thetaStart), np.sin(thetaStart)]) * .9
        reward = 0 # rest reward
        buffer[:] = np.nan # rest buffer
        # setup rat location and direction
        goalReached = False
        positions[trial, 0] = start
        # swim towards the center
        headDirection = thetaStart + np.pi

        # update the old direction
        mouse.direction = headDirection
        for t in range(1, nTime):
            # perform action
            firingRatesPC = mouse.firingRatePC(positions[trial, t - 1])
            firingRatesHC = mouse.firingRateHC(mouse.direction)
            # determine action cell firing rates
            for idx in range(mouse.nDirections):
                tmp = firingRatesPC.dot(actionWeights[..., idx])
                tmp = tmp.dot(firingRatesHC)
                actionFiringRate[idx] = tmp


            # compute firing rate probabilities
            probs = np.exp(\
            (actionFiringRate - np.max(actionFiringRate) * mouse.beta))
            # print(probs, actionFiringRate.max())
            probs /= probs.sum()
            # probs = softmax(actionFiringRate * mouse.beta)
            # probs = np.exp(actionFiringRate * mouse.beta)
            # probs = probs / probs.sum()

            positions[trial, t], idx  = mouse.updatePosition(\
                     positions[trial, t - 1], \
                     probs)

            distanceToPlatform = ((platformLocation - positions[trial, t])**2).sum()
            if distanceToPlatform < threshold:
                reward = 1
                # print('Goal reached')
                goals += 1

            # compute the expecation of the current position
            criticValueNow         = firingRatesPC.dot(criticWeights).dot(\
                                    firingRatesHC)
            # compute the expecation of the new location
            criticValueExpectation =  mouse.firingRatePC(positions[trial, t]).dot( \
                                    criticWeights.dot(mouse.firingRateHC(mouse.direction)))


            # Prediction error
            delta = reward + mouse.gamma * criticValueExpectation - criticValueNow
            # print(delta, criticValueExpectation, positions[trial, t])

            criticWeights += mouse.epsilon * delta * firingRatesPC[:, None].dot(firingRatesHC[None, :])
            # print(firingRatesPC.shape, firingRatesHC.shape)
            actionWeights[..., idx] += mouse.epsilon * delta * \
            firingRatesPC[:, None].dot(firingRatesHC[None,:])

            # print(delta)
            # print(actionWeights.max())
            # if actionWeights.max() > 0:
            #     actionWeights /= actionWeights.max()
            #     criticWeights /= criticWeights.max()
            # print(positions[trial, t])
            # show visuals
            buffer[t] = positions[trial, t]
            # if t:
            # if not trial % 5 and not t % 100:
            # if not trial % 5 and not t % 100:
            # if not t % (nTime // 3):
            if not t % 100:
                h.set_data(buffer.T)
                mainax.set_title(f'Trial {trial}; goals {goals}; reward {reward} t = {t}', fontsize = 40)
                fig.canvas.flush_events()
                fig.canvas.draw()

            if reward:
                h.set_data(buffer.T)
                mainax.set_title(f'Trial {trial}; goals {goals}; reward {reward} t = {t}', fontsize = 40)
                fig.canvas.flush_events()
                fig.canvas.draw()
                break
        # compute the critic  estimation
        for idx, xy in enumerate(coords.reshape(-1, 2)):
            maxAngle = 0
            PC = mouse.firingRatePC(xy)
            for jdx, angle in enumerate(mouse.dirs):
                tmp = PC.dot(\
                                criticWeights).dot(mouse.firingRateHC(angle))
                maxAngle = max([maxAngle, tmp])
            Z.ravel()[idx] = maxAngle


        # right.cla()
        # criticContour = right.plot_surface(*coords.T, Z = Z)
        u, v = np.gradient(Z)
        # print(u.shape, X.shape, v.shape)
        right.cla()
        # print(coords.shape, X.shape, Z.shape)
        right.plot_surface(X, Y, Z)
        if Z.max() > 1:
            right.autoscale()
        else:
            right.set_zlim(0, 1)
        # right.streamplot(X, Y, u[:, 0].reshape(X.shape), u[:, 1].reshape(X.shape))
        # right.streamplot(X, Y, v[:, 0].reshape(X.shape), v[:, 1].reshape(X.shape))
        right.set(**props)
        # criticContour.set_data(Z)
        # criticContour.autoscale()




if __name__ == '__main__':
    simulate()
    plt.show()
