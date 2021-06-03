"""
    File name: block_diagonalize
    Author: Scott Wolf
    Date created: 2020/06/26
    Python Version: 3.7

    Reference: Berman, Gordon J., William Bialek, and Joshua W. Shaevitz. "Predictability and hierarchy in Drosophila behavior." Proceedings of the National Academy of Sciences 113.42 (2016): 11943-11948.


"""
import numpy as np
from pylab import figure, cm
from matplotlib.colors import LogNorm
from scipy import *

def IBF(transition_matrix, num_clusters=9, max_iters=10000, tau=1, beta=10):
    # trans_mat = (trans_mat / np.sum(trans_mat))
    t_mat = transition_matrix.copy()
    n = t_mat.shape[0]
    przx = np.random.rand(num_clusters,n)
    przx = np.multiply(przx,1/np.sum(przx,axis=0))

    prx = (np.mean(np.linalg.matrix_power(t_mat.T,5000),axis=0).T)

    prz = np.ones((num_clusters, 1)) / num_clusters

    pryx = np.linalg.matrix_power(t_mat,tau)

    pryz = np.matmul(pryx,np.multiply(przx,prx).T)

    for i in range(max_iters):
        prz = prz.reshape(num_clusters,1)
        DKL = np.subtract(np.sum(pryx*np.log(pryx),axis=0), np.matmul(np.log(pryz.T),pryx))
        lnprzx = -beta*DKL + np.log(prz)
        lnprzx = lnprzx - np.max(lnprzx,axis=0)
        przx = np.multiply(np.exp(lnprzx), 1 / np.sum(np.exp(lnprzx), axis=0))
        prz = np.matmul(przx, prx)
        pryz = np.matmul(pryx, np.multiply(przx, prx.T).T)
        pryz = np.multiply(pryz,1/ np.sum(pryz,axis=0))

    m = np.max(przx,axis=0)
    loc = np.argmax(przx,axis=0)
    c = np.unique(loc)
    NA = list()
    for k in c:
        NA.append(np.sum(np.count_nonzero(np.where(loc == k))))
    m = np.sort(NA)
    s = np.argsort(NA)

    P =list()
    cNdx = {}
    for k in range(len(c)):
        P.append(np.where(loc == k)[0].tolist())
        cNdx[k] = np.where(loc == k)
    P = [item for sublist in P for item in sublist]

    block_diag_trans_mat = t_mat[:, P][P]

    return P, block_diag_trans_mat, c, loc, cNdx
