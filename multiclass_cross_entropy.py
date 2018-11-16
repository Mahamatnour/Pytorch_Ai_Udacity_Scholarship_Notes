import numpy as np
# the formula for cross-entropy 
# Write a function that takes as input two lists Y, P,
# and returns the float corresponding to their cross-entropy.
def cross_entropy(Y, P, Z):
    Y = np.float_(0.7)
    P = np.float_(0.3)
    Z = np.float_(0.4)
    return -np.sum(Y * np.log(P) + (1 - Y) * np.log(1 - P)+ (1 - Y) * np.log(1 - Z))
    