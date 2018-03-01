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
import fextAssemble as fext
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
    K = np.zeros([len(nodeCoords[:,0])*dim,len(nodeCoords[:,0])*dim])
    for i in range(np.prod(numEle)): #Iterate through each element
        S = 0 # (S=0 for internal)
        memDim = geas.memDim(S,dim,mCount)
        tempK = np.zeros([len(basisArray[:,0])*dim,len(basisArray[:,0])*dim])
        for j in range(mSize[-memDim]): #Iterate through each gauss point 

            # Construct basis at GP
            (intScalFact,hardCodedJac) = geas.gaussJacobian(S,i,j,dim,basisArray,mCount,mSize,nodeCoords,eleNodesArray)
            basisdXArray = geas.basisdX(S,j,dim,basisArray,mCount,mSize,hardCodedJac)
            
            
            for k in range(len(basisdXArray[:,0])): #iterate through each basis
                Ba1 = fint.bAssemble(basisdXArray[k,:])
                for l in range(len(basisdXArray[:,0])): #iterate through each basis
                    Ba2 = fint.bAssemble(basisdXArray[l,:])
#                    tempK[dim*k:dim*(k+1),dim*l:dim*(l+1)] += (np.matmul(np.matmul(np.transpose(Ba1[:dim,:dim]),constit[:dim,:dim]),Ba2[:dim,:dim])
#                                    *intScalFact*gWArray[j])
                    tempK[dim*k:dim*(k+1),dim*l:dim*(l+1)] += (np.matmul(np.matmul(np.transpose(Ba1),constit),Ba2)
                                    *intScalFact*gWArray[j])
#                    print(Ba1,'\n',Ba2,'\n')

        index = indexAssemble(eleNodesArray[:,i],dim)                     
        
        K[index,np.transpose(index)] += tempK
    return(K)
    
if __name__ == "__main__":
    # Parameters
    dim = 2
    numEle = [2]*dim
    eleSize = [2]*dim
    numBasis = 2
    gaussPoints = [-1/np.sqrt(3),1/np.sqrt(3)]
    E = 200*10**9 #modulus
    v = 0.3 #poisons ratio
    eps = 10**-5
###############################################################################    
    # Assmble the constitutive matrix from strain to stress
    constit = fint.constitAssemble(E,v,dim)
    #Assemble gause point arrays
    (gPArray,gWArray, mCount, mSize) = gas.parEleGPAssemble(dim,gaussPoints =gaussPoints)
    #Assemble the basis Array
    basisArray = bas.basisArrayAssemble(dim,numBasis,gaussPoints,gPArray, mCount, mSize) 
    #Assembe the mesh
    (nodeCoords,eleNodesArray,edgeNodesArray) = mas.meshAssemble(numEle,eleSize)
    
###############################################################################
    # Construct force field
    forceType = np.zeros([np.prod(numEle),np.sum(mCount)])
#    forceType[:,0] = 1
    forceType[[1,3],4] = 2
    forceType[[2,3],2] = 2
    forceType[3,4] = 3
    forces = np.ones([np.prod(numEle),np.sum(mCount)*dim])
    
    Fext = fext.fextAssemble(dim,numEle,gWArray, mCount, mSize,basisArray,nodeCoords,eleNodesArray,forces,forceType)
    
###############################################################################
    #Apply constraints
    disp = np.zeros([len(nodeCoords[:,0]),dim])
    
    constraintes = np.ones([len(nodeCoords[:,0]),dim]) 
    constraintes[0:9:3,:] = 0
#    constraintes[0:3,1] = 0

    contrainte2D = constraintes.astype(bool)
    constraintes = constraintes.flatten().astype(bool)    
    mask = np.outer(constraintes,constraintes)


###############################################################################
    #Iterate
    iMax = 10
    nMax = 1
    
    n = 0 
    while n < nMax:
        n += 1
#        Fext = fext.fextAssemble(dim,numEle,gWArray, mCount, mSize,basisArray,nodeCoords,eleNodesArray,constHas,constVal)
        
        Fint = fint.fintAssemble(dim,numEle,gWArray, mCount, mSize,basisArray,nodeCoords,eleNodesArray,constit,disp)
        R0 = Fext - Fint
    
        i = 0
        ui = disp
        while i < iMax:
            Fint = fint.fintAssemble(dim,numEle,gWArray, mCount, mSize,basisArray,nodeCoords,eleNodesArray,constit,ui)
#            print(Fext - Fint)
            Ri = (Fext - Fint)[contrainte2D]
            
            if np.linalg.norm(Ri) < eps:
                print(i,'solved','Ri =',np.linalg.norm(Ri))

                break
            else:
                print('Ri =',np.linalg.norm(Ri))
#                print(i)
                K = kAssemble(dim,numEle,constit,gWArray,mCount, mSize, basisArray,nodeCoords,eleNodesArray)

                deltU = np.linalg.solve(K[:,constraintes][constraintes,:],Ri)
#                print(i,deltU)
                ui[contrainte2D] = ui[contrainte2D] + deltU
#                print('ui =',ui[contrainte2D])
                i += 1

    
   