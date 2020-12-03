# replication of the algorithm from here -- An iterative algorithm for singular valuedecomposition on noisy incomplete matrices https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.231.5583&rep=rep1&type=pdf 

import numpy as np
import numpy.linalg as LA

# create ground truth SVD matrix
X = np.abs(np.random.normal(0.0,5,size=(5,5)))
print(X)
true_U, true_S, true_V = LA.svd(X)
print("True U: ", true_U)
print("True S : ", true_S)
print("True V : ", true_V)
# initialize our Us, Vs, Ys
U = np.abs(np.random.normal(0.0,0.001,size=(5,5)))
V = np.abs(np.random.normal(0.0,0.001,size=(5,5)))
Y = np.abs(np.random.normal(0.0,0.001,size=(5,5)))
num_iters = 0
lr = 0.1
N_iters = 1000
# orthogonalize U and V
U, _ = LA.qr(U)
V,_ = LA.qr(V)
#print(U.T @ U) # check it is orthogonal
S = U.T @ Y @ V
penalization_param = 0.0
# begin loop
for i in range(N_iters):
  U += lr * (((Y @ V) + (U @ V.T @ Y.T @ U)) @ S)
  V += lr * (((Y.T @ U) + (V @ U.T @ Y @ V)) @ S)
  #U = U * np.sign(S)
  S = U.T @ Y @ V 
  S = np.abs(S)
  Y += lr * ((U @ S @ V.T )- Y) #- (penalization_param * (X - Y)))
  #U, _ = LA.qr(U)
  #V,_ = LA.qr(V)
  if np.isnan(U).any():
    stop
print("U :", U)
print("V : ", V)
print("S : ", S)
print("DIFFS: ", U - true_U)
print("Diffs V", V - true_V)
print("DIFFS S", S - true_S)
