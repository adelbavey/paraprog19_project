
# python file to generate graphs

import numpy as np

#data_in = np.fromfile("gmm_out.txt", dtype=np.float64)
#data_in = np.reshape(data_in, (2,-1))
#print(data_in)

data_in = np.loadtxt("gmm_out.txt")

print(data_in)
