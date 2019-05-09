
# python file to generate graphs

import numpy as np
import matplotlib.pyplot as plt

#data_in = np.fromfile("gmm_out.txt", dtype=np.float64)
#data_in = np.reshape(data_in, (2,-1))
#print(data_in)

data = np.loadtxt("europediff.txt")
mu = np.loadtxt("gmm_out.txt")
l = plt.scatter(data[:,0],data[:,1])
plt.scatter(mu[:,0],mu[:,1])

plt.show()

#print(mu)
