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
import nonlinearFunctions as nlf

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

def constitAssemble(E,v,dim):
    gams = E*v/((1+v)*(1-2*v))  
    G = E/(2*(1+v))
    boB = 2*G+gams
    if dim == 1:
        constit = np.array([[E]])
    elif dim == 2:
        constit = np.array([[1,v,0],[v,1,0],[0,0,(1-v)/2]])*E/(1-v**2)
    elif dim == 3:
        constit = np.array([[boB,gams,gams,0,0,0],[gams,boB,gams,0,0,0],[gams,gams,boB,0,0,0],[0,0,0,G,0,0],[0,0,0,0,G,0],[0,0,0,0,0,G]])
    
    return(constit)
#def dispToStrain(disp):
def vonMises(s):
    s1 = s[0]
    s2 = s[1]
    s12 = s[2]
    vonM = (s1**2 - 1*s2 + s2**2 + 3*s12**2)**(1/2)
    return(vonM)
   
def fintAssemble(dim,numEle,gWArray, mCount, mSize,basisArray,nodeCoords,eleNodesArray,constit,disp):
    
    Fint = np.zeros([len(nodeCoords[:,0]),dim])
    tempNodeCoords = nodeCoords+disp
    vonM = np.zeros(np.prod(numEle))
    for i in range(np.prod(numEle)): #Iterate through each element
        S = 0 # (S=0 for internal)
        memDim = geas.memDim(S,dim,mCount)
        tempDisp = disp[eleNodesArray[:,i],:]
        Fa = np.zeros([len(basisArray[:,0]),dim])
        for j in range(mSize[-memDim]): #Iterate through each gauss point 
            
#            [temp1,temp2] = geas.gaussJacobian(S,i,j,dim,basisArray,mCount,mSize,disp,eleNodesArray)
            
            # Construct basis at GP
            (intScalFact,hardCodedJac) = geas.gaussJacobian(S,i,j,dim,basisArray,mCount,mSize,nodeCoords,eleNodesArray)
            basisdXArray = geas.basisdX(S,j,dim,basisArray,mCount,mSize,hardCodedJac)
            # Contruct Geometry at GP
            basisSubset = geas.basisSubsetGaussPoint(S,j,dim,basisArray,mCount,mSize)[:,0]
            # Compute Current Strain
            
            dUdX = nlf.partDeformationGrad(basisdXArray, tempDisp)
#            dUdX = temp2[:dim,:dim]
#            print(dUdX,'\n')
            [F,J] = nlf.deformationGrad(dUdX)
            gStrain = nlf.greenLagrangeStrain(dUdX)
            
            Cref = nlf.constitutiveCreater(F,J,constit)
            pkS = nlf.pkStress(gStrain,Cref)
#            stress = 
            stress = nlf.coachyStress(pkS,F,J)

            vonM[i] += vonMises(stress)*gWArray[j]
            (intScalFact,hardCodedJac) = geas.gaussJacobian(S,i,j,dim,basisArray,mCount,mSize,tempNodeCoords,eleNodesArray)
            basisdXArray = geas.basisdX(S,j,dim,basisArray,mCount,mSize,hardCodedJac)
            for k in range(len(basisSubset)): #iterate through each basis
                Ba = bAssemble(basisdXArray[k,:])
#                for l in range(dim):
#                   a = np.matmul(np.identity(dim)[l,:],np.matmul(np.transpose(Ba),stress))*intScalFact*gWArray[j]
#                   print(a)
                Fa[k,:] = np.matmul(np.transpose(Ba),stress)*intScalFact*gWArray[j]+Fa[k,:]
#        print(Fa)
        Fint[eleNodesArray[:,i],:] = Fa+Fint[eleNodesArray[:,i],:]
    return(Fint,vonM)

if __name__ == '__main__':
    dim = 2
    numEle = [2]*dim
    eleSize = [1]*dim
    numBasis = 2
    gaussPoints = [-1/np.sqrt(3),1/np.sqrt(3)]
    E = 200*10**9 #modulus
    v = 0 #poisons ratio
    
    #gaussPoints = [-1,1]
    (gPArray,gWArray, mCount, mSize) = gas.parEleGPAssemble(dim,gaussPoints =gaussPoints)
    basisArray = bas.basisArrayAssemble(dim,numBasis,gaussPoints,gPArray, mCount, mSize)
    
    (nodeCoords,eleNodesArray,edgeNodesArray) = mas.meshAssemble(numEle,eleSize)
    
    disp = np.zeros([len(nodeCoords[:,0]),dim])*0
    
    disp[0:9:3,:] = 0
    disp[1:9:3,0] = .05
    disp[2:9:3,0] = 0.1


    constit = constitAssemble(E,v,dim)
    
    Fin = fintAssemble(dim,numEle,gWArray, mCount, mSize,basisArray,nodeCoords,eleNodesArray,constit,disp)
    
