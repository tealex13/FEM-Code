# -*- coding: utf-8 -*-
"""
Created on Wed Apr 18 11:49:00 2018

@author: avila3
"""
 
import numpy as np
import matplotlib as plt
import meshAssemble as mas
import gaussAssemble as gas
import basisAssemble as bas
import geometryAssemble as geas
import fextAssemble as fext
import loadAssemble as load

def defGrad(basisdXArray, ui): 
    return np.matmul(ui.transpose(),basisdXArray)

if __name__ == "__main__":
    defGrad(1, 1, 1, 10, 1)
#    defGrad(basisArray, hardCodedJac, ui, ele, eleNodeArray)