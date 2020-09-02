import matplotlib.pyplot as plt
import numpy as np

x = np.arange(0., 10., 0.2)
print(len(x))
y = np.random.random(50)
print(x)
print(y)
print(x * y)
plt.scatter(x, y*x)
plt.show()
