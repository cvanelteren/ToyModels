import networkx as nx
from time import sleep


def search(path, graph, prev, target):
    current = path[-1]
    if current == target:
        return True
    elif current in path[:-1]:
        return False
    print(path)
    for neighbor in graph.neighbors(current):
        if neighbor != prev and neighbor not in path:
            # try-out new member
            path.append(neighbor)
            # go into the deep
            if search(path, graph, current, target):
                return path
            path.pop()
            print(path)

    return False
import random
N = 5
graph = nx.cycle_graph(N)
s = random.randint(0, N)
t= random.randint(0, N)
s,t = 1, 10
print(s, t)

path = search([s], graph, None, 2)
print("Returning", path)
print(tuple(graph.neighbors(s)))
