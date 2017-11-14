from pylab import *
from numpy import *

from sklearn import datasets
from sklearn.naive_bayes import GaussianNB
from pylab import *
from mpl_toolkits.mplot3d import axes3d

iris = datasets.load_iris()
parameters = []
for classs in unique(iris.target):
    idx = where(iris.target == classs)[0]
    data = iris.data[idx, :]
    mu, sigma = data.mean(axis = 0), data.std(axis = 0)

    parameters.append((mu, sigma))
parameters = array(parameters)

# %%
def gausProb(x, mu, sigma):
    norm  =  (sqrt(2*pi) * sqrt(sigma.T.dot(sigma)))**(-1)
    p     = exp(-((x - mu).T.dot(x-mu)) / sigma.T.dot(sigma))
    return p * norm

x = iris.data[0,:]
mu, sigma = parameters[0,...]
res = [ [] for i in range(len(parameters))]
print(len(res))
for x, label in zip(iris.data, iris.target):
    y = gausProb(x, *parameters[label, :])
    print(y)
    res[label].append(y)
res = array(res)

print(res.shape)
# %%
fig, ax = subplots();
ax.plot(res.T.max(1))
show()
