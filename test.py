import matplotlib.pyplot as plt
import numpy as np

data = np.load('expert_returns.npy')
plt.plot(data)
plt.show()