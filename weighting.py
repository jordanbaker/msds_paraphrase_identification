# -*- coding: utf-8 -*-
"""
Created on Wed May  3 21:21:40 2017

@author: Andrew Pomykalski
"""

from sklearn.feature_extraction.text import CountVectorizer
import pickle as cPickle
#from cPickle import load, dump
import numpy, gzip
import scipy.sparse as ssp

path = "C:/Users/Andrew Pomykalski/Desktop/Machine Learning/Final Project"
os.chdir(path)

class TFKLD(object):
    def __init__(self, ftrain, ftest):
        self.ftrain, self.ftest = ftrain, ftest
        self.trnM, self.trnL = None, None
        self.tstM, self.tstL = None, None
        self.weight = None

    def loadtext(self, fname):
        text, label = [], []
        with open(fname, 'r') as fin:
            for line in fin:
                items = line.strip().split("\t")
                label.append(int(items[0]))
                text.append(items[1])
                text.append(items[2])
        return text, label


    def createdata(self):
        trnT, trnL = self.loadtext(self.ftrain)
        tstT, tstL = self.loadtext(self.ftest)
        # Change the parameter setting in future
        countizer = CountVectorizer(dtype=numpy.float,
                                    ngram_range=(1,2))
        trnM = countizer.fit_transform(trnT)
        self.trnM, self.trnL = trnM, trnL
        tstM = countizer.transform(tstT)
        self.tstM, self.tstL = tstM, tstL
        self.trnM = ssp.lil_matrix(self.trnM)
        self.tstM = ssp.lil_matrix(self.tstM)


    def weighting(self):
        print('Create data matrix ...')
        self.createdata()
        print('Counting features ...')
        M = self.trnM.todense()
        print('type(M) = {}'.format(type(M)))
        L = self.trnL
        nRow, nDim = M.shape
        print('nRow, nDim = {}, {}'.format(nRow, nDim))
        # (0, F), (0, T), (1, F), (1, T)
        count = numpy.ones((4, nDim))
        for n in range(0, nRow, 2):
            if n % 1000  == 0:
                print('Process {} rows'.format(n))
            for d in range(nDim):
                label = L[n // 2]
                if ((M[n,d] > 0) and (M[n+1,d] == 0)) or ((M[n,d] == 0) and (M[n+1,d] > 0)):
                    # Non-shared
                    if label == 0:
                        # (0, F)
                        count[0,d] += 1.0
                    elif label == 1:
                        # (1, F)
                        count[2,d] += 1.0
                elif (M[n,d] > 0) and (M[n+1,d] > 0):
                    # Shared
                    if label == 0:
                        # (0, T)
                        count[1,d] += 1.0
                    elif label == 1:
                        # (1, T)
                        count[3,d] += 1.0
        # Compute KLD
        print('Compute KLD weights ...')
        weight = self.computeKLD(count)
        # Apply weights
        print('Weighting ...')
        self.__weighting()


    def computeKLD(self, count):
        # Smoothing
        count = count + 0.05
        # Normalize
        pattern = [[1,1,0,0],[1,1,0,0],[0,0,1,1],[0,0,1,1]]
        pattern = numpy.array(pattern)
        prob = count / (pattern.dot(count))
        #
        ratio = numpy.log((prob[0:2,:] / prob[2:4,:]) + 1e-7)
        self.weight = (ratio * prob[0:2,:]).sum(axis=0)
        print(self.weight.shape)


    def __weighting(self):
        weight = ssp.lil_matrix(self.weight)
        print('Applying weighting to training examples')
        for n in range(self.trnM.shape[0]):
            if n % 1000  == 0:
                print('Process {} rows'.format(n))
            self.trnM[n, :] = self.trnM[n, :].multiply(weight)
        print('Applying weighting to test examples')
        for n in range(self.tstM.shape[0]):
            if n % 1000  == 0:
                print('Process {} rows'.format(n))
            self.tstM[n, :] = self.tstM[n, :].multiply(weight)


    def save(self, fname):
        D = {'trnM':self.trnM, 'trnL':self.trnL,
             'tstM':self.tstM, 'tstL':self.tstL}
        with gzip.open(fname, 'w') as fout:
            cPickle.dump(D, fout)
        print('Done')


def main():
    ftrain = "C:/Users/Andrew Pomykalski/Desktop/Machine Learning/Final Project/train.txt"
    ftest = "C:/Users/Andrew Pomykalski/Desktop/Machine Learning/Final Project/test.txt"
    tfkld = TFKLD(ftrain, ftest)
    tfkld.weighting()
    tfkld.createdata()
    tfkld.save("original-data.pickle.gz")

if __name__ == "__main__":
    main()