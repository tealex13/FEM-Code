# -*- coding: utf-8 -*-
"""
Created on Sat Feb 10 13:13:49 2018

@author: avila3
"""

import numpy as np
import matplotlib.pyplot as plt 
import meshAssemble as mas
import gaussAssemble as gas




def unpackBasis(basisFuncArray,pointsToEval):
    '''
    This function evaluates each basis function and derivative at a point
    '''
    
    tempArray = np.zeros(basisFuncArray.shape)
    for i in range(len(basisFuncArray[:,0])): #Iterate through each combination of basis funcitions
        for j in range(len(basisFuncArray[0,:])): #Iterate through basis in the combination
            tempArray[i,j] = basisFuncArray[i,j](pointsToEval[j])
            
    return tempArray

def basisArrayPartAssemble(dim,dimMem,numBasis,gaussPoints,gPArray, mCount, mSize):
    '''
    This function assembles the matrix of basis functions and derivatives for a single dimension
    and calles 'unpackBasis' for each set
    '''
    
    b0 = lambda x:1/2*(1-x)
    b1 = lambda x:1/2*(1+x)
    db0 = lambda x:-1/2
    db1 = lambda x:1/2
        
    basisArrayPart = np.zeros([numBasis**dim,(1+dim)*mCount[-dimMem]*len(gaussPoints)**dimMem])
#    print("\n",mCount[-(dimMem)],"\n")
    
    for k in range(mCount[-(dimMem)]): #Iterate through each member in group
    
        startingPoint = np.sum(mCount[:-dimMem]*mSize[:-dimMem]).astype(int)+mSize[-dimMem]*k
    
        pointsToEval = gPArray[startingPoint:mSize[-dimMem]+startingPoint,:]
        startingPoint = startingPoint+mSize[-dimMem]
        basisCartesianArray = np.matlib.repmat([b0,b1],dim,1)
        #basisFuncArray = mas.cartesian(basisCartesianArray)
#        print(pointsToEval,"\n")
        
        for i in range(len(pointsToEval[:,0])): #Iterate through each gauss point set
            basisCartesianArrayTemp = basisCartesianArray
            basisFuncArray = mas.cartesian(basisCartesianArrayTemp)
            basisArrayPart[:,(i+k*mSize[-dimMem])*(dim+1)]=np.prod(unpackBasis(basisFuncArray,pointsToEval[i,:]),axis=1)

            for j in range(dim): #Iterate through each partial derivatives for each dimensions
                basisCartesianArrayTemp = np.array(basisCartesianArray)
                basisCartesianArrayTemp[j,:] = [db0,db1]
                basisFuncArray = mas.cartesian(basisCartesianArrayTemp)
                basisArrayPart[:,(i+k*mSize[-dimMem])*(dim+1)+j+1] = np.prod(unpackBasis(basisFuncArray,pointsToEval[i,:]),axis=1)

    return basisArrayPart


def basisArrayAssemble(dim,numBasis,gaussPoints,gPArray, mCount, mSize):
    '''
    This function computes the values of the basis functions and derivatives of the 
    basis functions at each gaussPoint.
    It iterates calls 'basisArrayPartAssemble' for each dimension and assembles the matrices. 
    
    OUTPUT:
        basisArray- is the matrix of basis functions evaluated at the gauss point
            form: [N0(GP0),dN0(GP0[0]),dN0(GP0[1]),...N0(GP1),dN0(GP1[0]),dN0(GP1[1])...],...
                [N1(GP1),dN1(GP1[0]),dN1(GP1[1]),...N1(GP1),dN1(GP1[0]),dN1(GP1[1])...],...
                ...]
                This is repeated for each member
    '''
    
    basisArray = np.zeros([numBasis**dim,(1+dim)*sum(mCount*mSize).astype(int)])
    startingPoint = 0
    for i in range(dim,0,-1): #Iterate through each group of members
        basisArray[:,startingPoint:startingPoint+(1+dim)*mCount[-(i)]*mSize[-(i)]] = basisArrayPartAssemble(
                dim,i,numBasis,gaussPoints,gPArray, mCount, mSize)
        startingPoint = startingPoint + (1+dim)*mCount[-(i)]*mSize[-(i)]
    return basisArray  
 
if __name__ == "__main__":
    dim = 2

    numBasis = 2
    gaussPoints = [-1/np.sqrt(3),1/np.sqrt(3)]  
    (gPArray,gWArray, mCount, mSize) = gas.parEleGPAssemble(dim,gaussPoints =gaussPoints)
    basisArray = basisArrayAssemble(dim,numBasis,gaussPoints,gPArray, mCount, mSize)