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
import loadAssemble as load

def indexAssemble(eleNodesArray,dim):
    temp = eleNodesArray*dim
    temp2 = temp+dim
    index= np.zeros(len(temp)*dim)
    for i in range(len(temp)):
        index[i*dim:(i+1)*dim] = range(temp[i],temp2[i])
    index = np.matlib.repmat(index,len(index),1).astype(int)
    return(index)
        
def kAssemble(dim,numEle,constit,gWArray,mCount, mSize, basisArray,nodeCoords,eleNodesArray):
    detArray = geas.detAssemble(dim,mCount)#used for selecting jacobians
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
    return(K)
if __name__ == "__main__":
    # Parameters
    dim = 2
    numEle = [2]*dim
    eleSize = [1]*dim
    numBasis = 2
    gaussPoints = [-1/np.sqrt(3),1/np.sqrt(3)]
    E = 200*10**9 #modulus
    v = 0 #poisons ratio
    
    # Assmble the constitutive matrix from strain to stress
    constit = fint.constitAssemble(E,v,dim)
    #Assemble gause point arrays
    (gPArray,gWArray, mCount, mSize) = gas.parEleGPAssemble(dim,gaussPoints =gaussPoints)
    #Assemble the basis Array
    basisArray = bas.basisArrayAssemble(dim,numBasis,gaussPoints,gPArray, mCount, mSize) 
    #Assembe the mesh
    (nodeCoords,eleNodesArray,edgeNodesArray) = mas.meshAssemble(numEle,eleSize)
    
    # Construct force field
    numNodes = 2**dim*np.prod(numEle)
#    nodeArray = np.array([1,3])
#    dimArray = np.array([0,0])
#    values = np.array([1,1])*10**8
    nodeArray = np.array([1])
    dimArray = np.array([0])
    values = np.array([1])*10**8
    (constHas,constVal) = load.constraints(dim, numNodes, nodeArray, dimArray, values)
    
    E = 200*10**9 #modulus
    v = 0.3 #poisons ratio
    
    disp = np.ones([len(nodeCoords[:,0]),dim])*0
    
#    disp[[0,3],:] = 0
#    disp[[2,5],:] = .1
#    disp[:,1:] = 0
    
    disp[0:9:3,:] = 0
#    disp[1:9:3,:] = .05
    disp[2:9:3,:] = .1
    disp[:,1:] = 0
    
#    disp[[0,2,4,6],:] = 0
#    disp[[1,3,5,7],:] = .12
#    disp[:,1:] = 0
    
    constit = fint.constitAssemble(E,v,dim)

    nMax = 10
    n = 0 
    while n < nMax:
        n += 1
        Fext = FextAssemble(dim,numEle,gWArray, mCount, mSize,basisArray,nodeCoords,eleNodesArray,constHas,constVal)
    
        i = 0
        
    
    K = kAssemble(dim,numEle,constit,gWArray,mCount, mSize, basisArray,nodeCoords,eleNodesArray)
    
   