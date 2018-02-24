# -*- coding: utf-8 -*-
"""
Created on Fri Feb 23 15:59:18 2018

@author: avila3
"""

import numpy as np
import meshAssemble as mas
import gaussAssemble as gas
import basisAssemble as bas
import geometryAssemble as geas
import fintAssemble as fint

def indexAssemble(eleNodesArray,dim):
    temp = eleNodesArray*dim
    temp2 = temp+dim
    index= np.zeros(len(temp)*dim)
    for i in range(len(temp)):
        index[i*dim:(i+1)*dim] = range(temp[i],temp2[i])
    index = np.matlib.repmat(index,len(index),1).astype(int)
    return(index)
        

if __name__ == "__main__":

    dim = 3
    numEle = [1]*dim
    eleSize = [1]*dim
    numBasis = 2
    gaussPoints = [-1/np.sqrt(3),1/np.sqrt(3)]
    E = 200*10**9 #modulus
    v = 0 #poisons ratio
    if dim == 1:
        constit = np.array([[E]])
    if dim == 2:
        constit = fint.stressAssemble(np.identity(3),E,v,dim)
    if dim == 3:
        constit = fint.stressAssemble(np.identity(6),E,v,dim)
    (gPArray,gWArray, mCount, mSize) = gas.parEleGPAssemble(dim,gaussPoints =gaussPoints)
    basisArray = bas.basisArrayAssemble(dim,numBasis,gaussPoints,gPArray, mCount, mSize)
    
    (nodeCoords,eleNodesArray,edgeNodesArray) = mas.meshAssemble(numEle,eleSize)
    
    detArray = geas.detAssemble(dim,mCount)
    
    K = np.zeros([len(nodeCoords[:,0])*dim,len(nodeCoords[:,0])*dim])
    for i in range(np.prod(numEle)): #Iterate through each element
        S = 0 # (S=0 for internal)
        memDim = geas.memDim(S,dim,mCount)
        tempK = np.zeros([len(basisArray[:,0])*dim,len(basisArray[:,0])*dim])
        for j in range(mSize[-memDim]): #Iterate through each gauss point 

            # Construct basis at GP
            (intScalFact,hardCodedJac) = geas.gaussJacobian(S,i,j,dim,basisArray,mCount,mSize,detArray,nodeCoords,eleNodesArray)
            basisdXArray = geas.basisdX(S,j,dim,basisArray,mCount,mSize,hardCodedJac)
            
            
            for k in range(len(basisdXArray[:,0])): #iterate through each basis
                Ba1 = fint.bAssemble(basisdXArray[k,:])
                for l in range(len(basisdXArray[:,0])): #iterate through each basis
                    Ba2 = fint.bAssemble(basisdXArray[l,:])
                    tempK[dim*k:dim*(k+1),dim*l:dim*(l+1)] = (np.matmul(np.matmul(np.transpose(Ba1),constit),Ba2)
                                    *intScalFact*gWArray[j]+tempK[dim*k:dim*(k+1),dim*l:dim*(l+1)])

        index = indexAssemble(eleNodesArray[:,i],dim)

        K[index,np.transpose(index)] = tempK + K[index,np.transpose(index)]