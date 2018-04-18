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

def forceSorter(dim,forces,forceType,typeEvaluted):
    '''
    typeEvaluated:
        1 = body force
        2 = pressure
        3 = traction force
        
        For pressure only it only looks at the value in the x place
        
    OUTPUT:
        outForce- The forces in each dimension of each node.
    '''
    outForce = np.zeros(forces.shape)
    mask = np.repeat(forceType,dim,axis=1)==typeEvaluted
    outForce[mask] = forces[mask]
    return(outForce)

def fextAssemble(dim,numEle,gWArray, mCount, mSize,basisArray,nodeCoords,eleNodesArray,forces,forceType):
    bodyForce = forceSorter(dim,forces,forceType,1)
    pressForce = forceSorter(dim,forces,forceType,2)
    tractForce = forceSorter(dim,forces,forceType,3)
    
    Fext = np.zeros([len(nodeCoords[:,0]),dim])
    for i in range(np.prod(numEle)): #Iterate through each element
#        print('\n')
        tempNodes = eleNodesArray[:,i]
        
        if dim == 1:
            Srange = 1
        else:
            Srange = np.sum(mCount[:3])
        fa = np.zeros([len(tempNodes),dim])    
        for S in range(Srange): # iterate through each side
            # Get the nodes from side

            memDim = geas.memDim(S,dim,mCount)
            startingGW = np.sum(mSize[-(memDim+1)])+(S-np.sum(mCount[-(memDim+1)]))*mSize[-memDim]
            
            
            
            for j in range(mSize[-memDim]): #Iterate through each gauss point
                              
                # Construct basis at GP
                (intScalFact,hardCodedJac) = geas.gaussJacobian(S,i,j,dim,basisArray,mCount,mSize,nodeCoords,eleNodesArray)
                basisSubset = geas.basisSubsetGaussPoint(S,j,dim,basisArray,mCount,mSize)[:,0]
#                print("Gauss Point", j, "\n", intScalFact, "\n")
                if S == 0:
                    fa += np.outer(basisSubset,bodyForce[i,:dim])*intScalFact*gWArray[startingGW+j]
#                    print(fa)
                
                if S > 0:
                    if S == 1:
                        a = 1
                    # Pressure
                    tempNormal = geas.stupidNormals(S,hardCodedJac,dim)
#                    print(tempNormal[:dim],'\n',np.outer(basisSubset,pressForce[i,dim*S:dim*(S+1)]),'\n')
#                    tempNormal = np.matlib.repmat(tempNormal[:dim],len(basisSubset),1)
#                    fa += np.outer(basisSubset,pressForce[i,dim*S:dim*(S+1)])*tempNormal*intScalFact*gWArray[startingGW+j]

                    fa += np.outer(basisSubset*pressForce[i,dim*S],tempNormal[:dim])*intScalFact*gWArray[startingGW+j]
                    # Traction
                    fa += np.outer(basisSubset,tractForce[i,dim*S:dim*(S+1)])*intScalFact*gWArray[startingGW+j]
#                    print(hardCodedJac,'\n')
#                    print(np.outer(basisSubset*pressForce[i,dim*S],tempNormal[:dim])*intScalFact*gWArray[startingGW+j],'\n')
#                    print ("S", S, "\n normal \n", tempNormal,'\n')
#            if S > 0:
#                print (np.linalg.det(hardCodedJac[:2,:2]),'\n')
                
                
        Fext[tempNodes,:] += fa
    return(Fext)



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
    
    r = (nodeCoords[:,1]+1)
    theta = np.pi/2*(nodeCoords[:,0]/np.max(nodeCoords[:,0]))
    nodeCoords[:,0] =  r*np.sin(theta)
    nodeCoords[:,1] =  r*np.cos(theta)
    
    side = 1
    forceType = np.zeros([np.prod(numEle),np.sum(mCount)])
    forceType[0:numEle[0],side] = 2

    forces = np.zeros([np.prod(numEle),np.sum(mCount)*dim])
    forces[0:numEle[0],side*dim+0]=1
    
    
    Fext = fextAssemble(dim,numEle,gWArray, mCount, mSize,basisArray,nodeCoords,eleNodesArray,forces,forceType)
    
   