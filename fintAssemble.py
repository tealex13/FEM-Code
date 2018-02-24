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

def stressAssemble(strain,E,v,dim):
    gams = E*v/((1+v)*(1-2*v))  
    G = E/(2*(1+v))
    boB = 2*G+gams
    if dim == 1:
        constit = np.array([[E]])
    if dim == 2:
        constit = np.array([[1,v,0],[v,1,0],[0,0,1-v]])*E/(1-v**2)
    elif dim == 3:
        constit = np.array([[boB,gams,gams,0,0,0],[gams,boB,gams,0,0,0],[gams,gams,boB,0,0,0],[0,0,0,G,0,0],[0,0,0,0,G,0],[0,0,0,0,0,G]])
    stress = np.matmul(constit,strain)
    return(stress)
#def dispToStrain(disp):

if __name__ == '__main__':
    dim = 2
    numEle = [2]*dim
    eleSize = [1]*dim
    numBasis = 2
    gaussPoints = [-1/np.sqrt(3),1/np.sqrt(3)]
    E = 200*10**9 #modulus
    v = 0 #poisons ratio

    stressAssemble(np.identity(dim),E,v,dim)
    
    #gaussPoints = [-1,1]
    (gPArray,gWArray, mCount, mSize) = gas.parEleGPAssemble(dim,gaussPoints =gaussPoints)
    basisArray = bas.basisArrayAssemble(dim,numBasis,gaussPoints,gPArray, mCount, mSize)
    
    (nodeCoords,eleNodesArray,edgeNodesArray) = mas.meshAssemble(numEle,eleSize)
    
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
    
    #print(CtoX([0,0],eleNodesArray,nodeCoords))
    detArray = geas.detAssemble(dim,mCount)
    
    
    Fint = np.zeros([len(nodeCoords[:,0]),dim])
    for i in range(np.prod(numEle)): #Iterate through each element
        S = 0 # (S=0 for internal)
        memDim = geas.memDim(S,dim,mCount)
        tempDisp = disp[eleNodesArray[:,i],:]
        Fa = np.zeros([len(basisArray[:,0]),dim])
        print('element', i)
        for j in range(mSize[-memDim]): #Iterate through each gauss point 

            # Construct basis at GP
            (intScalFact,hardCodedJac) = geas.gaussJacobian(S,i,j,dim,basisArray,mCount,mSize,detArray,nodeCoords,eleNodesArray)
            basisdXArray = geas.basisdX(S,j,dim,basisArray,mCount,mSize,hardCodedJac)
#            print(basisdXArray,'\n')
            # Contruct Geometry at GP
            basisSubset = geas.basisSubsetGaussPoint(S,j,dim,basisArray,mCount,mSize)[:,0]
            # Compute Current Strain
            strain = 0
#            print('Gauss Point',j)
            for k in range(len(basisSubset)): #iterate through each basis
                Ba = bAssemble(basisdXArray[k,:])
#                print(Ba,'\n')
                strain = strain + np.matmul(Ba,tempDisp[k,:])

#            print(strain,'\n')
#            print(basisdXArray,'\n')
            # Calculate Stress
            stress = stressAssemble(strain,E,v,dim)
            
            for k in range(len(basisSubset)): #iterate through each basis
                Ba = bAssemble(basisdXArray[k,:])
#                for l in range(dim):
#                   a = np.matmul(np.identity(dim)[l,:],np.matmul(np.transpose(Ba),stress))*intScalFact*gWArray[j]
#                   print(a)
                Fa[k,:] = np.matmul(np.transpose(Ba),stress)*intScalFact*gWArray[j]+Fa[k,:]
        startingPointInternal = i*mCount[-dim]*mSize[-dim]*len(basisArray[:,0])
#        print(Fa)
        Fint[eleNodesArray[:,i],:] = Fa+Fint[eleNodesArray[:,i],:]