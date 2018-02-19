# -*- coding: utf-8 -*-
"""
Created on Sat Feb 17 13:52:16 2018

@author: Avila
"""
import numpy as np
import loadAssemble as load
import meshAssemble as mas
import gaussAssemble as gas
import basisAssemble as bas
import CtoX as cx

dim = 2
numEle = [1,1]
eleSize = [1,1]
endPoints = np.array([-1,1])
cPoints = mas.cartesian(np.matlib.repmat(endPoints,dim,1))

numBasis = 2
gaussPoints = endPoints.tolist() 
(gPArray,gWArray, mCount, mSize) = gas.parEleGPAssemble(dim,gaussPoints =gaussPoints)
basisArray = bas.basisArrayAssemble(dim,numBasis,gaussPoints,gPArray,gWArray, mCount, mSize)

(nodeCoords,eleNodesArray,edgeNodesArray) = mas.meshAssemble(numEle,eleSize)

temp = np.zeros([dim,dim])
temp2 = np.zeros([dim,1])
k = 0 #iterate through each element
for i in range(dim): #iterate through each of the dimensions
    for j in range(dim): #iterate through each dimsion in parent element
        temp2 = basisArray[np.arange(len(basisArray[:,0])),np.arange(len(basisArray[:,0]))*(dim+1)+1+j] 
        tempPoints = cx.CtoX(cPoints[:,i],nodeCoords[eleNodesArray[0,k],:],nodeCoords[eleNodesArray[-1,k],:])
        temp[i,j] = np.sum(tempPoints*basisArray[:,1+j])