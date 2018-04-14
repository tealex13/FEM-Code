# -*- coding: utf-8 -*-
"""
Created on Fri Feb 23 15:59:18 2018

@author: avila3
"""

import numpy as np
import matplotlib as plt
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

def newtonRaph(dim,numEle,constit,gPArray,gWArray, 
               mCount, mSize,basisArray,nodeCoords,eleNodesArray,forces,forceType,disp,constraintes):
    eps = 10**-5
    Fext = fext.fextAssemble(dim,numEle,gWArray, mCount, mSize,basisArray,nodeCoords,eleNodesArray,forces,forceType)
    
    contrainte2D = constraintes.astype(bool)
    constraintes = constraintes.flatten().astype(bool) 
    
    #Iterate
    iMax = 10
    nMax = 1
    
    n = 0 
    while n < nMax:
        n += 1
#        Fext = fext.fextAssemble(dim,numEle,gWArray, mCount, mSize,basisArray,nodeCoords,eleNodesArray,constHas,constVal)
        
        R0 = np.linalg.norm(Fext)
        print(R0)
    
        i = 0
        ui = disp
        while i < iMax:
            Fint = fint.fintAssemble(dim,numEle,gWArray, mCount, mSize,basisArray,nodeCoords,eleNodesArray,constit,ui)
#            print(Fext - Fint)
            Ri = (Fext - Fint)[contrainte2D]
            
            if np.linalg.norm(Ri) < eps*R0:
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
    return(ui)

 
if __name__ == "__main__":
    #setup
    plt.pyplot.close('all')
    case = 'pressure'
#    case = 'patch'
    
    # Parameters
    dim = 2
    numEle = [5]*dim
    eleSize = [1]*dim
    numBasis = 2
    gaussPoints = [-1/np.sqrt(3),1/np.sqrt(3)]
    E = 200*10**9 #modulus
    E = 1#modulus
    v = 0.0 #poisons ratio
    
###############################################################################    
    # Assmble the constitutive matrix from strain to stress
    constit = fint.constitAssemble(E,v,dim)
    #Assemble gause point arrays
    (gPArray,gWArray, mCount, mSize) = gas.parEleGPAssemble(dim,gaussPoints =gaussPoints)
    #Assemble the basis Array
    basisArray = bas.basisArrayAssemble(dim,numBasis,gaussPoints,gPArray, mCount, mSize) 
    #Assembe the mesh
    (nodeCoords,eleNodesArray,edgeNodesArray) = mas.meshAssemble(numEle,eleSize)
 
    
#### Pressure Vessel
    if case ==  'pressure':
        r = (nodeCoords[:,1]+1)
        theta = np.pi/2*(nodeCoords[:,0]/np.max(nodeCoords[:,0]))
        nodeCoords[:,0] =  r*np.sin(theta)
        nodeCoords[:,1] =  r*np.cos(theta)
        # Construct force field
        side = 1
        forceType = np.zeros([np.prod(numEle),np.sum(mCount)])
        forceType[0:numEle[0],side] = 2
        forces = np.zeros([np.prod(numEle),np.sum(mCount)*dim])
        forces[0:numEle[0],side*dim+0]=.1
        #Apply constraints
        disp = np.zeros([len(nodeCoords[:,0]),dim])
        
        constraintes = np.ones([len(nodeCoords[:,0]),dim]) 
        constraintes[0:(numEle[0]+1)**2:numEle[0]+1,0] = 0 #mesh must be square
        constraintes[numEle[0]:(numEle[0]+1)**2:numEle[0]+1,1] = 0 #mesh must be square
        
    elif case == 'patch':
        nodeCoords[:,1] = nodeCoords[:,1]*((1-nodeCoords[:,0])*.4+1)
        # Construct force field
        side = 2
        forceType = np.zeros([np.prod(numEle),np.sum(mCount)])
        forceType[numEle[0]*(numEle[1]-1):numEle[0]*numEle[1],side] = 2 #dim must equal 2
        forces = np.zeros([np.prod(numEle),np.sum(mCount)*dim])
        forces[numEle[0]*(numEle[1]-1):numEle[0]*numEle[1],side*dim+0]=.1
        disp = np.zeros([len(nodeCoords[:,0]),dim])
        constraintes = np.ones([len(nodeCoords[:,0]),dim]) 
        constraintes[0:numEle[0]+1,1] = 0
        constraintes[0:np.prod(numEle)+numEle[0]+1:numEle[0]+1,0] = 0
    

#### 

###############################################################################

    ui = newtonRaph(dim,numEle,constit,gPArray,gWArray, 
               mCount, mSize,basisArray,nodeCoords,eleNodesArray,forces,forceType,disp,constraintes)
    mas.plotFigure(1,nodeCoords)
    mas.plotFigure(1,nodeCoords+ui)
    
    temp = ui[0:np.prod(numEle)+numEle[0]+1:numEle[0]+1,1]
    
   