
from pip import main
with open('requirements.txt', 'r') as f:
    for line in f:
        print(line)
        main(('install ' + line).split(' '))
import networkx
from pylab import *
from numpy import *
from izhikevich import Izikevich
n = 100
k = 30
p = .6

G = networkx.random_graphs.connected_watts_strogatz_graph(n, k, p)
fig, ax = subplots()
networkx.draw(G, pos = networkx.draw_circular(G))
print(G.edges())
show()

# %%
# - make graph
# - for every node in the graph make a neural object with random values
# - simulate and record the spikes

def createVertices(G, v = -70, u = -65, a = .02, b = .2, c = -65, d = 2):
    vertices = {}
    for node in G.nodes():
        v = random.rand() * -70 + 50
        u = random.rand() * -65 + 15
        d = random.rand() * 2
        vertices[node] = Izikevich(v, u, a, b, c, d)
    return vertices
# %%

def simulate(G, vertices, nSteps = 1000, dt = .5):
    time = arange(0, nSteps, dt)
    states = []
    for t in time:
        nextState = []
        for node in G.nodes():
            # TODO: clean this up
            if t == 0:
                vertices[node].dt = dt
            neighbors = G[node].keys()
            neighborCurrent = 0
            # get the external input from its neighbors
            for neighbor in neighbors:
                neighborCurrent += int(vertices[node].spiked)
            # add random noise + external current
            state = vertices[node].updateState(neighborCurrent + random.rand() * 15)
            # print(state)
            nextState.append(state)
        states.append(nextState)
    return states

vertices = createVertices(G)
r = array(simulate(G, vertices))
cfg = {'xlabel' : 'time[ms]', 'ylabel' : 'neuron [spike]'}
fig, ax = subplots();
ax.imshow(r[...,-1].T, aspect ='auto', cmap = 'gray_r')
setp(ax, **cfg)
savefig('example.png')
fig, ax = subplots()
ax.plot(r[...,0])
show()
