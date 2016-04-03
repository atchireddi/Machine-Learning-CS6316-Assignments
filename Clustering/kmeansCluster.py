__author__ = 'zhangyin'
import numpy as np


class kmeansCluster:
    def __init__(self,k,maxIter):
        self.k = k
        self.maxIter = maxIter

    def _dist(self,x,y): # define the distance between two points
        return sum((x-y)**2)**0.5

    def _reset(self,k,maxIter):
        self.k = k
        self.maxIter = maxIter
        self.clusters = None
        self.xlabel = None

    def _init(self,X):
        n,p = X.shape
        return X[np.random.choice(range(n),self.k,replace=False)] # random initialization

    def run(self,X):
        n,p = X.shape
        clusters = self._init(X)
        iter = 0
        converge = False
        while iter < self.maxIter and not converge:
            old_clusters = clusters.copy()
            xlabel=[]
            for x in X:
                xlabel.append(np.argmin([self._dist(x,c) for c in old_clusters]))
            for i in range(self.k):
                clusters[i] = sum(X[np.array(xlabel) == i])/len(X[np.array(xlabel) == i])
            if np.all(clusters==old_clusters):
                converge=True
                print "kmeans algorithm converges in %d iterations" %(iter +1)
            iter += 1
        if iter == self.maxIter:
            print "kmeans algorithm fails to converges in %d iterations" %self.maxIter

        self.clusters =  clusters
        self.xlabel = np.array(xlabel)


