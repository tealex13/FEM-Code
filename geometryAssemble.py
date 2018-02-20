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

def memDim(S,dim,mCount):
#    Sdim = 0
    for i in range(dim):
        if 0 > S - np.sum(mCount[:i+2]):
            Sdim = dim-i
            break
        elif i == dim-1:
            Sdim = 0
    return Sdim   

def basisSubsetAssemble(S,dim,basisArray,mCount,mSize,divDim = 0):
#    basisSubset = np.zeros([len(basisArray[:,0]),mSize])
    sDim = memDim(S,dim,mCount)
    startingPoint = np.sum(mSize[:-sDim])*(dim+1)+(S-np.sum(mCount[:-sDim]))*(dim+1)
    endingPoint = startingPoint+mSize[-sDim]*(dim+1)
    print(np.arange(startingPoint,endingPoint,dim+1))
     
    basisSubset = basisArray[:,np.arange(startingPoint+divDim,endingPoint+divDim,dim+1)]
        
    return(basisSubset)


def CtoX(C,eleNodesArray,nodeCoords):
    dim = len(C)
    b0 = lambda x:1/2*(1-x)
    b1 = lambda x:1/2*(1+x)
    
    basisArray = mas.cartesian(np.matlib.repmat([b0,b1],dim,1).tolist())
    x = np.zeros([1,dim])
    for i in range(len(basisArray)):
        temp = 1
        for j in range(dim):
            temp = basisArray[i,j](np.array(C))*temp
        x = temp*nodeCoords[eleNodesArray[i],:]+x
    return x

def gausstoX(basisSubset,eleNodesArray,nodeCoords):
    '''
    OUTPUT:
        x- the coordinate of the gauss points captured in basisSubset in the xyz frame
            format: [[GP1],[GP2],[GP3],...] where GP are gauss points [x,y,z,...]
    '''
    x = np.zeros([len(basisSubset[0,:]),dim])
    for i in range(len(basisSubset[0,:])): #Step through set of basis for each Gauss Point   
        x[i,:] = np.sum(basisSubset[:,i].reshape(len(nodeCoords[:,0]),1)*nodeCoords[eleNodesArray[:,0],:],0) #Sum each row
    return x

def combonator(evalDim,dim):
    '''
    INPUTS:
        evalDim- the dimensions of S
        dim- dimensions of element
    OUTPUTS:
        temp- the combinations of 0s and 1s
    '''
    temp = mas.cartesian(np.matlib.repmat([1,0],dim,1))
    temp = temp[np.sum(temp,1)==evalDim,:]
    return(temp)
    
def detAssemble(evalDim,dim,mCount):
    temp = combonator(evalDim,dim)
    
    reps = [1]+(mCount[2:]/dim).astype(int).tolist()
    np.repeat(temp, reps, axis=0)
    
    
#def gaussJacobian(S,dim,bassisArray,mCount,mSize,divDim = 0):
#    sDim = memDim(S,dim,mCount)
#    for i in range(mSize[-sDim]): #step through each gauss point
#        for j in range(): #step through each dimension
#    basisSubset = basisSubsetAssemble(0,dim,basisArray,mCount,mSize, divDim = 0)
#    x = gausstoX(basisSubset,eleNodesArray,nodeCoords)
    
dim = 3
numEle = [1]*dim
eleSize = [1]*dim
endPoints = np.array([-1,1])
cPoints = mas.cartesian(np.matlib.repmat(endPoints,dim,1))

numBasis = 2
gaussPoints = [-1/np.sqrt(3),1/np.sqrt(3)]
#gaussPoints = [-1,1]
(gPArray,gWArray, mCount, mSize) = gas.parEleGPAssemble(dim,gaussPoints =gaussPoints)
basisArray = bas.basisArrayAssemble(dim,numBasis,gaussPoints,gPArray,gWArray, mCount, mSize)

(nodeCoords,eleNodesArray,edgeNodesArray) = mas.meshAssemble(numEle,eleSize)

#print(CtoX([0,0],eleNodesArray,nodeCoords))
combonator(2,3)
#basisSubset = basisSubsetAssemble(0,dim,basisArray,mCount,mSize,divDim = 1)
#print("\n",basisSubset,'\n')
#x = gausstoX(basisSubset,eleNodesArray,nodeCoords)
#print(x)

#temp = np.zeros([dim,dim])
#temp2 = np.zeros([dim,1])
#k = 0 #iterate through each element
#for i in range(dim): #iterate through each of the dimensions
#    for j in range(dim): #iterate through each dimsion in parent element
#        temp2 = basisArray[np.arange(len(basisArray[:,0])),np.arange(len(basisArray[:,0]))*(dim+1)+1+j] 
#        tempPoints = cx.CtoX(cPoints[:,i],nodeCoords[:,eleNodesArray[0,k]],nodeCoords[:,eleNodesArray[-1,k]])
#        temp[i,j] = np.sum(tempPoints*basisArray[:,1+j])