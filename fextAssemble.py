# -*- coding: utf-8 -*-
"""
Created on Thu Feb 22 09:13:31 2018

@author: avila3
"""

import numpy as np
import matplotlib.pyplot as plt 
import loadAssemble as load
import meshAssemble as mas
import gaussAssemble as gas
import basisAssemble as bas
import geometryAssemble as geas
import fintAssemble as fint

if __name__ == '__main__':
    dim = 2
    numEle = [1]*dim
    eleSize = [1]*dim
    numBasis = 2
    gaussPoints = [-1/np.sqrt(3),1/np.sqrt(3)]
    
    #gaussPoints = [-1,1]
    (gPArray,gWArray, mCount, mSize) = gas.parEleGPAssemble(dim,gaussPoints =gaussPoints)
    basisArray = bas.basisArrayAssemble(dim,numBasis,gaussPoints,gPArray, mCount, mSize)
    
    (nodeCoords,eleNodesArray,edgeNodesArray) = mas.meshAssemble(numEle,eleSize)
  
    detArray = geas.detAssemble(dim,mCount)

    # Construct force field
    numNodes = 2**dim*np.prod(numEle)
    nodeArray = np.array([1,3])
    dimArray = np.array([0,0])
    values = np.array([1,1])*10**8
    (constHas,constVal) = load.constraints(dim, numNodes, nodeArray, dimArray, values)
    
    
    Fext = np.zeros([len(nodeCoords[:,0]),dim])
    for i in range(np.prod(numEle)): #Iterate through each element
#        print('\n')
        tempNodes = eleNodesArray[:,i]
        for S in range(mCount[-dim]+mCount[-dim+1]): # iterate through each side. This will not work for 1 dimensional
            # Get the nodes from side
#            tempNodes = eleNodesArray[mas.sNodes(S,dim),i]
            
            memDim = geas.memDim(S,dim,mCount)
            
            tempHasForce = constHas[:,tempNodes]
            tempBodyForce = np.zeros(tempHasForce.shape)
            tempForce = constVal[:,tempNodes]
            
            startingGW = np.sum(mSize[-(memDim+1)])+(S-np.sum(mCount[-(memDim+1)]))*mSize[-memDim]
            
            Fa = np.zeros([len(tempNodes),dim])
            
            for j in range(mSize[-memDim]): #Iterate through each gauss point
                               
                # Construct basis at GP
                (intScalFact,hardCodedJac) = geas.gaussJacobian(S,i,j,dim,basisArray,mCount,mSize,detArray,nodeCoords,eleNodesArray)
                if S>0:
                    tempNormals = geas.stupidNormals(S,hardCodedJac,dim)
                basisdXArray = geas.basisdX(S,j,dim,basisArray,mCount,mSize,hardCodedJac)
                # Contruct Geometry at GP
                basisSubset = geas.basisSubsetGaussPoint(S,j,dim,basisArray,mCount,mSize)[:,0]
#                basisSubset = basisSubset[mas.sNodes(S,dim)]
                # Compute Current Strain
                for k in range(len(basisSubset)): #Iterate through basis
#                    print(i,S,j,k,'\t',len(basisSubset))
                    if S == 0: #Body force
                        a = 1 
                        Fa[k,:] = basisSubset[k]*tempBodyForce[:,k]*intScalFact*gWArray[startingGW+j]+Fa[k,:]

                    elif S > 0: #traction Force
                        for l in range(dim): #Iterate through dimensions
                            if tempHasForce[l,k] == 1: #pressure
                                Fa[k,l] = basisSubset[k]*tempForce[l,k]*tempNormals[l]*intScalFact*gWArray[startingGW+j]+Fa[k,l]
                            elif tempHasForce[l,k] == 0: #Direct Loads
                                Fa[k,l] = basisSubset[k]*tempForce[l,k]*intScalFact*gWArray[startingGW+j]+Fa[k,l]
#                                print(l,k)
                                    
                                
                                
            Fext[tempNodes,:] = Fa +Fext[tempNodes,:]
