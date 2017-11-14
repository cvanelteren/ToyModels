import networkx
from pylab import *
from numpy import *
from izhikevich import Izikevich

# %%
# - make graph
# - for every node in the graph make a neural object with random values
# - simulate and record the spikes

def createVertices(G, v = -70, u = -65, a = .02, b = .2, c = -65, d = 2, state = 1):
    vertices = {}
    for node in G.nodes():
        v = (random.rand() * 70  - 20)*-1
        u = random.rand() * -65 + 15
        d = random.rand() * 2
        if random.rand() > .9:
            state = -1
        else:
            state = 1
        vertices[node] = Izikevich(v, u, a, b, c, d, type = state)
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
            state = vertices[node].updateState(neighborCurrent + random.randn()*15)
            # print(state)
            nextState.append(state)
        states.append(nextState)
    return states


n = 500
m = 1
k = 5
p = .9

G = networkx.random_graphs.connected_watts_strogatz_graph(n, k, p)
# G = networkx.complete_graph(200)
# G  = networkx.random_graphs.erdos_renyi_graph(n,m)
fig, ax = subplots()
networkx.draw(G) #, pos = networkx.draw_circular(G))
# print(G.edges())

vertices = createVertices(G)
r = array(simulate(G, vertices))
cfg = {'xlabel' : 'time[ms]', 'ylabel' : 'neuron [spike]'}
fig, ax = subplots();
h = ax.imshow(r[...,-1].T, aspect ='auto')
setp(ax, **cfg)
savefig('example.png')
colorbar(h)
# fig, ax = subplots()
# ax.plot(r[...,0])
show()
