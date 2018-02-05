# -*- coding: utf-8 -*-
"""
Created on Mon Jan 29 11:14:57 2018

@author: avila3
"""

import numpy as np
import matplotlib.pyplot as plt 

def CtoX(C,nodeCoordsLeft,nodeCoordsRight):

    x = (C+1)/2*(nodeCoordsRight-nodeCoordsLeft)+nodeCoordsLeft
    return x

def jacobian(nodeCoordsLeft,nodeCoordsRight):

    return np.diag((nodeCoordsRight-nodeCoordsLeft)/2)


    
if __name__  == "__main__":
#    CtoX(np.array([[-1,1,0],[1,.5,0]]),np.array([[3,2,1],[1,1,1]]),np.array([[2,3,2],[3,2,1]]))
    jac = jacobian(np.array([-1,2,3]),np.array([1,1,2]))

