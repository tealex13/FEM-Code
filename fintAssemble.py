# -*- coding: utf-8 -*-
"""
Created on Wed Feb 21 12:15:07 2018

@author: avila3
"""

import numpy as np
import matplotlib.pyplot as plt 
import loadAssemble as load
import meshAssemble as mas
import gaussAssemble as gas
import basisAssemble as bas
import geometryAssemble as geas

def bAssemble(basisdX):
    dim = len(basisdX)
    if dim == 1:
        b = basisdX
    elif dim == 2:
        basisdX = np.insert(basisdX,0,0)
        b = np.array([[1,0],[0,2],[2,1]])
        b = basisdX[b]
    elif dim == 3:
        basisdX = np.insert(basisdX,0,0)
        b = np.array([[1,0,0],[0,2,0],[0,0,3],[0,3,2],[3,0,1],[2,1,0]])
        b = basisdX[b]
    return(b)
    
#def dispToStrain(disp):
    

if __name__ == '__main__':
    dim = 2
    numEle = [1]*dim
    eleSize = [1]*dim
    numBasis = 2
    gaussPoints = [-1/np.sqrt(3),1/np.sqrt(3)]
    E = 180*10**9 #modulus
    v = .3 #poisons ratio
    gams = E*v/((1+v)*(1-2*v))  
    G = E/(2*(1+v))
    boB = 2*G+gams
    
    #gaussPoints = [-1,1]
    (gPArray,gWArray, mCount, mSize) = gas.parEleGPAssemble(dim,gaussPoints =gaussPoints)
    basisArray = bas.basisArrayAssemble(dim,numBasis,gaussPoints,gPArray, mCount, mSize)
    
    (nodeCoords,eleNodesArray,edgeNodesArray) = mas.meshAssemble(numEle,eleSize)
    
    disp = np.ones([np.prod(numEle)*np.sum(mCount*mSize),dim])*.001
    disp[0,:] = 0
    disp[1:,1] = 0
    
    #print(CtoX([0,0],eleNodesArray,nodeCoords))
    detArray = geas.detAssemble(dim,mCount)
    
    
    Fint = np.zeros([mCount[-dim]*mSize[-dim]*np.prod(numEle)*len(basisArray[:,0]),dim])
    for i in range(np.prod(numEle)): #Iterate through each element
        S = 0 # (S=0 for internal)
        memDim = geas.memDim(S,dim,mCount)
        startingPoint = np.sum(mCount[-(memDim+1)]*mSize[-(memDim+1)])+S*mSize[-memDim]
#        print(startingPoint)
        tempDisp = disp[startingPoint:startingPoint+mSize[-memDim],:]
#        print(i)
        for j in range(mSize[-memDim]): #Iterate through each gauss point 
#            print(j)
            # Construct basis at GP
            (intScalFact,hardCodedJac) = geas.gaussJacobian(S,i,j,dim,basisArray,mCount,mSize,detArray,nodeCoords,eleNodesArray)
            basisdXArray = geas.basisdX(S,j,dim,basisArray,mCount,mSize,hardCodedJac)
            # Contruct Geometry at GP
            basisSubset = geas.basisSubsetGaussPoint(S,j,dim,basisArray,mCount,mSize)[:,0]
            # Compute Current Strain
            strain = 0
            for k in range(len(basisSubset)): #iterate through each basis
                Ba = bAssemble(basisdXArray[k,:])
#                print(Ba,'\n')
                strain = strain + np.matmul(Ba,tempDisp[k,:])

            # Calculate Stress
            if dim == 2:
                constit = np.array([[1,v,0],[v,1,0],[0,0,1-v]])*E/(1-v**2)
            elif dim == 3:
                constit = np.array([[boB,gams,gams,0,0,0],[gams,boB,gams,0,0,0],[gams,gams,boB,0,0,0],[0,0,0,G,0,0],[0,0,0,0,G,0],[0,0,0,0,0,G]])
            sigma = np.matmul(constit,strain)
            Fa = np.zeros([len(basisSubset),dim])
            for k in range(len(basisSubset)): #iterate through each basis
                Ba = bAssemble(basisdXArray[k,:])
#                for l in range(dim):
#                   a = np.matmul(np.identity(dim)[l,:],np.matmul(np.transpose(Ba),sigma))*intScalFact*gWArray[j]
#                   print(a)
                Fa[k,:] = np.matmul(np.transpose(Ba),sigma)*intScalFact*gWArray[j]+Fa[k,:]
            startingPointInternal = i*mCount[-dim]*mSize[-dim]*len(basisArray[:,0])+j*len(basisArray[:,0])
            Fint[startingPointInternal:startingPointInternal+len(basisArray[:,0]),:] = Fa 