from scipy.sparse import vstack, lil_matrix
import pickle as cPickle
from scipy.sparse.linalg import svds
import numpy, gzip
import os

path = "C:/Users/Andrew Pomykalski/Desktop/Machine Learning/Final Project"
os.chdir(path)

class DimReduction(object):
    def __init__(self, M, K):
        self.M = M
        self.K = K


    def svd(self):
        U, s, Vt = svds(self.M, k=self.K)
        W = U.dot(numpy.diag(s))
        H = Vt
        return (W, H)


    def nmf(self):
        pass


def main(K=20):
    with gzip.open("original-data.pickle.gz") as fin:
        D = cPickle.load(fin)
        trnM, trnL = D['trnM'], D['trnL']
        tstM, tstL = D['tstM'], D['tstL']
    # Combine data together for DR
    M = vstack([trnM, tstM])
    print('M.shape = {}'.format(M.shape))
    dr = DimReduction(M, K)
    W, H = dr.svd()
    # Split data
    trnM = W[:(2*len(trnL)), :]
    tstM = W[(2*len(trnL)):, :]
    # Save data
    D = {'trnM':trnM, 'trnL':trnL,
         'tstM':tstM, 'tstL':tstL}
    print('Save data into file ...')
    with gzip.open("dr-data.pickle.gz", "w") as fout:
        cPickle.dump(D, fout)
    print('Done')
    

if __name__ == '__main__':
    main()