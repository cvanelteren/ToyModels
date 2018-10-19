from numpy import *
from matplotlib.pyplot import *
import tqdm
import itertools, functools, PyQt5
from vispy import scene, visuals, app
# TODO: clean up code, add fully featured data transport and vispy support (current beta)
# %%
class mainWindow(PyQt5.QtWidgets.QMainWindow):
    def __init__(self, data):
        canvas  = scene.SceneCanvas()
        view    = canvas.central_widget.add_view()

        ncolors   = 5
        spacing   = linspace(0, 1, ncolors + 1)
        self.cmaps = cm.viridis(spacing)

        xr =  array([0, data.T.shape[0]]); yr = array([0, data.T.shape[1]])
        data = self.cmaps[data, :]
#        assert 0
        image   = scene.visuals.Image(data = data, parent = view.scene)

        view.camera = 'panzoom'
        view.camera.viewbox.camera.set_range(x = xr, y = yr)

        self.view  = view
        self.image = image
        canvas.show()

    def update(self, data):
        self.image.set_data(self.cmaps[data.clip(0, len(self.cmaps) - 1), :])
        self.image.draw()
        self.view.update()

# %%
class Sandbox:
    def __init__(self, shape, threshold = 4, parent = None):
        self.threshold = threshold
        self.shape = shape
        self.state = zeros(shape, dtype = int)
        self.state = random.randint(3, 4, size = self.shape)
        self.N     = len(self.state.ravel())
        self.neighbors = array(
                            list(
                            itertools.product(*[range(-1,2) for _ in self.shape])
                                )
                            )[1::2] # cardinal directions
        # create parent window
        if parent == None:
            self.parent = mainWindow(self.state)


    def checkAvalanche(self, alpha = True):
        idx = [unravel_index(idx, self.shape) for idx, i in enumerate(self.state.ravel()) if i > self.threshold]
        # idx = array(where(self.state >= self.threshold)).T # list per location
        for coordinate in idx:
            self.state[tuple(coordinate)] -= self.threshold
            neighbors = coordinate + self.neighbors
            for neighbor in neighbors:
                if all(neighbor // self.shape == 0):
                    self.state[tuple(neighbor)] += 1
        self.parent.update(self.state)
    def step(self, ev):
        coordinate = random.randint(0, self.N)
        coordinate = unravel_index(coordinate, self.shape)
        self.state[coordinate] += 1
        self.checkAvalanche()
        self.parent.update(self.state)

    def simulate(self, steps = 1e9, buffer = 1, plot = False):
        output = []
        for t in tqdm.tqdm(range(int(steps))):
#            print(t)
            if t % int(buffer * steps) == 0:
                if plot: self.plot()
            self.step()
        return output
N = 100
shape = (N,N)
sandbox = Sandbox((shape))
timer = app.Timer(connect = sandbox.step)
timer.start()
if __name__ == '__main__':
    app.run()
