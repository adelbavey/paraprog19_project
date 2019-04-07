

import numpy as np
import random
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal



def expectation_step(data, priors, mu, Sigma):
	N = data.shape[0];
	K = mu.shape[0];
	posteriors = np.zeros((N,K))
	for i in range(N):
		row_sum = 0
		for k in range(K):
			posteriors[i][k] = priors[k]*multivariate_normal.pdf(data[i],mean=mu[k], cov=Sigma[k])
			row_sum += posteriors[i][k]
		for k in range(K):
			posteriors[i][k] /= row_sum
	return posteriors


def maximization_step(data, posteriors):

	N = data.shape[0]
	K = posteriors.shape[1]

	#new priors
	priors = np.sum(posteriors,axis=0)/N

	#new means
	mu = np.dot(posteriors.T,data)/np.dot(posteriors.T, np.ones((N,1)))

	#new vars
	Sigma = [np.zeros((dims, dims)) for k in range(K)]
	for k in range(K):
		data_nomean = data-mu[k]
		data_nomean_weighted = np.multiply(data_nomean,np.matrix(posteriors[:,k]).T)
		cov_mat = np.dot(data_nomean_weighted.T, data_nomean)
		cov_mat/= np.sum(np.matrix(posteriors[:,k]))
		Sigma[k] = cov_mat

	return priors, mu, Sigma


#load data
data = np.loadtxt("s1.txt")
N = data.shape[0]
dims = data.shape[1]



K = 20

#init priors
priors = np.full((K),1/K)

#init means, randomly assign to data points
mu = np.zeros((K, dims))
for k in range(K):
	mu[k] = data[random.randint(0,N)]

#init variances, same for all components
sample_mean = np.mean(data, 0)
sum_init_vars = np.zeros((dims, dims))
for point in (data-sample_mean):
	sum_init_vars += np.matrix(point).T*np.matrix(point)
sum_init_vars/=N

Sigma = [sum_init_vars for k in range(K)]

#EM algorithm
posteriors = np.zeros((N,K))
for i in range(1,100):
	
	print(i)
	#E step
	posteriors = expectation_step(data, priors, mu, Sigma)
	#M step
	priors, mu, Sigma = maximization_step(data, posteriors)

#print(mu)
#print(Sigma)

l = plt.scatter(data[:,0],data[:,1])
plt.scatter(mu[:,0],mu[:,1])


plt.show()