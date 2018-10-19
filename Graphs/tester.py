import numpy as np


# window = np.array([(i,j) for i in range(-1, 2) for j in range(-1, 2)])
#
# someBin = np.ones((100,100))
# idxs = np.array(np.where(someBin > 0)).T
# for idx in idxs:
#     if ((idx + window)  // someBin.shape == 0 ).all():
#         if (someBin[idx + window] > 0).all():
#             print(f"Center {idx} contains all zeros")


# a   = np.random.rand(4,3,10)
# b   = np.zeros(a.shape)
# idx = [np.ravel_multi_index(i, a.shape) for i in zip(*np.where(a < .1))]
# print(a, idx)
# for i in idx:
#     a.ravel()[i] = b.ravel()[i]
# print(a)

def power(x, base = 2):
    if x <= base:
        return 1
    return power(x // base, base) + 1


print (power(int(1e9)))
