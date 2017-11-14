from pylab import *
from numpy import *
# TODO: the update rule does not work as intended; i.e. the voltage can be higher than the threshold

class Izikevich:
    def __init__(self, v, u, a, b, c, d, dt = 1, threshold = 30, type = 1):
        # assign everything to class attributes
        [setattr(self, key, value) for key, value in locals().items() if key != 'self']
        self.spiked = 0

    def updateState(self, I = 0):
        if self.v >= self.threshold:
            self.v = self.c
            self.u += self.d
            self.spiked = 1 * self.type # output to other neurons
        else:
            dv = (.04 * self.v +  5 )* self.v + 140 - self.u + I
            du = self.a * (self.b * self.v - self.u)
            self.v += dv * self.dt
            self.u += du * self.dt
            self.spiked = 0

        if self.v >= self.threshold:
            self.v = self.c
            self.u += self.d
            self.spiked = 1 * self.type # output to other neurons

        return [self.v, self.u, self.spiked]

if __name__ == '__main__':
    # TYPICAL VALUES FROM Izikevich paper
    v = -70 # mV
    a =  .02 # time scale recovery variable
    b = .2  # sensitivity of the recovery variable to u
    c = -65 # mV  afterspike reset value of membrane potential caused by K+ influx
    d = 2   # describes after spike reset of the recovery variable u caused by slow-high treshold of Na+ and K+
    u = b * v

    nSteps = 1000
    dt     = .5
    time   = arange(0, nSteps, dt)
    I      =  random.randn(len(time)) * 15
    neuron = Izikevich(v, u, a, b, c, d, dt)
    uv     = array([neuron.updateState(i) for i in I])
    fig, ax = subplots()
    # ax.plot(time, uv[:,0])
    ax.plot(time, uv[:,0])
    setp(ax, 'xlabel', 'time[ms]', 'ylabel', 'Membrane potential[mv]')
    show()
