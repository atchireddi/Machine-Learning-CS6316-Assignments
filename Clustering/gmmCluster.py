__author__ = 'zhangyin'

import numpy as np

class gmmCluster:
    def __init__(self,k,maxIter,covType):
        self.k = k
        self.maxIter = maxIter
        self.covType = covType
        if covType not in ["diag","full"]:
            print "Warning: covType should be either 'diag' or 'full' "

    def _initialize(self,X):
        xn,xp = X.shape
        mu = X[np.random.choice(range(xn),self.k,replace=False)]
        pi = np.ones(self.k)/self.k
        return mu,pi


    def _estep(self, X, mu, Sigma, pi):  # mu, pi, Sigma should be np.array,
        xn,xp = X.shape
        gamma = np.zeros((xn,self.k))
        for i in range(xn):
            x = X[i]
            temp = np.zeros(self.k)
            for h in range(self.k):
                temp[h] = np.exp(-0.5* np.inner(np.inner(x-mu[h], np.linalg.inv(Sigma)),x-mu[h]) )*pi[h]
            gamma[i] = temp/sum(temp)
        return gamma

    def _mstep(self,X,gamma):  # gamma should be np.array
        xn,xp = X.shape
        mu = np.zeros((self.k,xp))
        pi = np.zeros(self.k)
        for h in range(self.k):
            top=np.zeros(xp)
            bottom=0
            for i in range(xn):
                top = top + gamma[i,h]*X[i]
                bottom = bottom + gamma[i,h]
            mu[h] = top/bottom
            pi[h] = bottom/xn
        return mu,pi

    def run(self,X):
        if self.covType == "diag":
            Sigma = np.diag(np.diag(np.cov(X.T)))
        elif self.covType == "full":
            Sigma = np.cov(X.T)
        else:
            print "error"

        self.mu = None
        self.labels = None

        n,p = X.shape
        mu,pi = self._initialize(X)
        iter = 0
        converge = False
        while iter < self.maxIter and not converge:
            old_mu = mu.copy()
            old_pi = pi.copy()
            gamma = self._estep(X, old_mu, Sigma, old_pi)
            mu,pi = self._mstep(X,gamma)
            if np.sum(abs(old_mu-mu))/np.sum(abs(old_mu))<0.001:
                converge=True
                print("GMM algorithm converges in "+str(iter+1)+" iterations")
            iter = iter + 1
        if iter == self.maxIter:
            print("GMM algorithm fails to converge in "+str(iter)+" iterations")

        labels = [np.argmax(g) for g in gamma]
        self.mu = mu
        self.labels =labels