import numpy as np

k = 10
random_points = np.random.choice(np.arange(50), k, replace=False)[:, None]
# print(random_points)

a = []
b1 = [1, 2]
b2 = [3, 4]
b3 = [5, 6]
a.append(b1)
a.append(b2)
a.append(b3)
print(a)
print(np.mean(a, axis=0))