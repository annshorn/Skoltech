# encoding: utf-8
# pset2.py

import numpy as np
# don't forget import packages, e.g. scipy
# but make sure you didn't put unnecessary stuff in here

# INPUT : diag_broadcast - list of diagonals value to broadcast,length equal to 3 or 5; n - integer, band matrix shape.
# OUTPUT : L - 2D np.ndarray, L.shape[0] depends on bandwidth, L.shape[1] = n-1, do not store main diagonal, where all ones;                  add zeros to the right side of rows to handle with changing length of diagonals.
#          U - 2D np.ndarray, U.shape[0] = n, U.shape[1] depends on bandwidth;
#              add zeros to the bottom of columns to handle with changing length of diagonals.
def band_lu(diag_broadcast, n):
    
    if len(diag_broadcast) == 3:
        L_2 = [0]
        U_dig = []
        U_up = []
        U_dig.append(diag_broadcast[1])
        for i in range(1, n):
            L_2.append(diag_broadcast[0]/U_dig[i-1])
            U_up.append(diag_broadcast[2])
            U_dig.append(diag_broadcast[1] - L_2[i]*U_up[i-1])
        
        L = [L_2[1:], np.ones([1, n])]

        U_up.append(0)
        U = np.vstack((U_dig, U_up))
        U = U.T
        
    if len(diag_broadcast) == 5:
        
        L_l = [diag_broadcast[1]/diag_broadcast[2]]
        U_k = [diag_broadcast[3]]
        U_diag = [diag_broadcast[2], diag_broadcast[2] - L_l[0]*U_k[0]]
        L_m = [diag_broadcast[0]/U_diag[0], diag_broadcast[0]/U_diag[1]]
        
        U_d = np.array([])
        
        for i in range(1, n):
            U_k.append(diag_broadcast[3] -  L_l[i-1]* diag_broadcast[4] )
            U_diag.append( diag_broadcast[2] - L_m[i-2]*diag_broadcast[4] - L_l[i-1]*U_k[i-1])
            
            U_d= np.append(U_d, diag_broadcast[4])
            
            L_l.append( (diag_broadcast[1] - L_m[i-1]*U_k[i-1])/U_diag[i] )
            
            L_m.append( diag_broadcast[0]/U_diag[i] )
            
        #print(len(L_l), len(U_k), len(U_diag), len(L_m))
        
        L_l.append(0)
        
        L = [L_m, L_l, np.ones([1, n])]

        U_k.append(0)
        
        
        U = np.vstack((U_diag, U_k))
        U_d = np.append(U_d, 0)
        U_d = np.append(U_d, 0)
        
        U_d = U_d.reshape([np.shape(U_d)[0], 1])
        
        
        U = np.vstack((U, U_d.T))
        U = U.T
        

    return L, U 


# INPUT : rectangular matrix A
# OUTPUT: matrices Q - orthogonal and R - upper triangular such that A = QR
def gram_schmidt_qr(A):
    
    Q = []
    A = A.transpose()
    m = A.copy()
    u = A

    summ = 0
    for i in range(1, A.shape[0]):
        for j in range(0, i):
            summ = summ + (np.dot(A[i], u[j])/np.dot(u[j],u[j]))*u[j]
        for s in range(0, i):
            u[i] = u[i] - summ
        summ = 0
            
    for k in range(u.shape[0]):
        Q.append(u[k]/(np.linalg.norm(u[k], ord=2)))
    
    R = np.dot(Q, m.transpose())
    
    return np.transpose(Q), R

# INPUT : rectangular matrix A
# OUTPUT: matrices Q - orthogonal and R - upper triangular such that A = QR
def modified_gram_schmidt_qr(A):
    
    Q = []
    A = A.transpose()
    m = A.copy()
    u = A
    
    for i in range(1, A.shape[0]):
        for j in range(0, i):
            r = np.dot(A[i], u[j])/np.dot(u[j],u[j])
            u[i] = u[i] - r*u[j]
    
    for k in range(u.shape[0]):
        Q.append(u[k]/(np.linalg.norm(u[k], ord=2)))
        
    R = np.dot(Q, m.transpose())
    
    return np.transpose(Q), R


# INPUT : rectangular matrix A
# OUTPUT: matrices Q - orthogonal and R - upper triangular such that A=QR
def householder_qr(A):
    
    A = A.T
    
    Q = np.eye(A.shape[1])
    #P = np.eye(A.shape[0], A.shape[1])
    
    R = A.copy()
    
    AA = A.copy()
    
    for i in range(R.shape[0]-1):
        
        P = np.eye(R.shape[1]) #1])
        I = np.eye(R.shape[1]) #1])
        R = R[i:, i:]
        alfa = np.linalg.norm(R[:1], ord=2)

        e = np.zeros([np.shape(R)[1],])#len(R),])
        e[0] = alfa
        u = R[:1] - np.sign(alfa)*e
        v = u/np.linalg.norm(u, ord=2)
        v = v.reshape((np.shape(v)[1], 1))
        
        P[i:, i:] = I[i:, i:] - 2*np.dot(v, v.T)
        
        R = np.dot(P, AA.T)
        R = R.T
        AA = R.copy()
    
        Q = np.dot(Q, P)
    
    return Q, R.transpose()


# INPUT:  G - np.ndarray
# OUTPUT: A - np.ndarray (of size G.shape)
def pagerank_matrix(G): # 5 pts
    # enter your code here
    return A


# INPUT:  A - np.ndarray (2D), x0 - np.ndarray (1D), num_iter - integer (positive) 
# OUTPUT: x - np.ndarray (of size x0), l - float, res - np.ndarray (of size num_iter + 1 [include initial guess])
def power_method(A, x0, num_iter): # 5 pts
    # enter your code here
    return x, l, res


# INPUT:  A - np.ndarray (2D), d - float (from 0.0 to 1.0), x - np.ndarray (1D, size of A.shape[0/1])
# OUTPUT: y - np.ndarray (1D, size of x)
def pagerank_matvec(A, d, x): # 2 pts
    # enter your code here
    return y


def return_words():
    # insert the (word, cosine_similarity) tuples
    # for the words 'numerical', 'linear', 'algebra' words from the notebook
    # into the corresponding lists below
    # words_and_cossim = [('word1', 'cossim1'), ...]
    
    numerical_words_and_cossim = []
    linear_words_and_cossim = []
    algebra_words_and_cossim = []
    
    return numerical_words_and_cossim, linear_words_and_cossim, algebra_words_and_cossim