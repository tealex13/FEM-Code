# -*- coding: utf-8 -*-
"""
Created on Mon Jan 29 11:14:57 2018

@author: avila3
"""

import numpy as np
import matplotlib.pyplot as plt 

def CtoX(C,nodeCoordsLeft,nodeCoordsRight):

    x = (C+1)/2*(nodeCoordsRight-nodeCoordsLeft)+nodeCoordsLeft
    print(x)

if __name__  == "__main__":
    CtoX(np.array([[-1,1,0],[1,.5,0]]),np.array([[3,2,1],[1,1,1]]),np.array([[2,3,2],[3,2,1]]))