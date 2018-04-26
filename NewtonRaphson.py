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
import nonlinearFunctions as nlf

def indexAssemble(eleNodesArray,dim):
    temp = eleNodesArray*dim
    temp2 = temp+dim
    index= np.zeros(len(temp)*dim)
    for i in range(len(temp)):
        index[i*dim:(i+1)*dim] = range(temp[i],temp2[i])
    index = np.matlib.repmat(index,len(index),1).astype(int)
    return(index)
        
def kAssemble(dim,numEle,constit,gWArray,mCount, mSize, basisArray,nodeCoords,eleNodesArray,ui):
    K = np.zeros([len(nodeCoords[:,0])*dim,len(nodeCoords[:,0])*dim])
    tempNodeCoords = nodeCoords+ui
    for i in range(np.prod(numEle)): #Iterate through each element
        S = 0 # (S=0 for internal)
        memDim = geas.memDim(S,dim,mCount)
        tempK = np.zeros([len(basisArray[:,0])*dim,len(basisArray[:,0])*dim])
        tempKg = np.zeros([len(basisArray[:,0])*dim,len(basisArray[:,0])*dim])
        tempKm = np.zeros([len(basisArray[:,0])*dim,len(basisArray[:,0])*dim])
        
        tempDisp = ui[eleNodesArray[:,i],:]
        # current coordinate frame
        for j in range(mSize[-memDim]): #Iterate through each gauss point 

            # Construct basis at GP
            (intScalFact,hardCodedJac) = geas.gaussJacobian(S,i,j,dim,basisArray,mCount,mSize,nodeCoords,eleNodesArray)
            basisdXArray = geas.basisdX(S,j,dim,basisArray,mCount,mSize,hardCodedJac)
#            print(intScalFact)
            basisSubset = geas.basisSubsetGaussPoint(S,j,dim,basisArray,mCount,mSize)[:,0]
            # Compute Current Strain
            
            dUdX = nlf.partDeformationGrad(basisdXArray, tempDisp)
#            print(dUdX,'\n')

            [F,J] = nlf.deformationGrad(dUdX)
#            print(F,'\n',J,'\n')
            gStrain = nlf.greenLagrangeStrain(dUdX)
#            print(gStrain,'\n')
            Cref = nlf.constitutiveCreater(F,J,constit)
#            print(Cref,'\n')
            pkS = nlf.pkStress(gStrain,Cref)
#            print(pkS,'\n')

            stress = nlf.fromVoigt(nlf.coachyStress(pkS,F,J))
#            print(stress,'\n')
#    
            # Recalculate basis DX array
            (intScalFact,hardCodedJac) = geas.gaussJacobian(S,i,j,dim,basisArray,mCount,mSize,tempNodeCoords,eleNodesArray)
#            print(intScalFact)
            basisdXArray = geas.basisdX(S,j,dim,basisArray,mCount,mSize,hardCodedJac)
            for k in range(len(basisdXArray[:,0])): #iterate through each basis
                Ba1 = fint.bAssemble(basisdXArray[k,:])
                for l in range(len(basisdXArray[:,0])): #iterate through each basis
                    Ba2 = fint.bAssemble(basisdXArray[l,:])
#                    tempK[dim*k:dim*(k+1),dim*l:dim*(l+1)] += (np.matmul(np.matmul(np.transpose(Ba1[:dim,:dim]),constit[:dim,:dim]),Ba2[:dim,:dim])
#                                    *intScalFact*gWArray[j])
                    Km = (np.matmul(np.matmul(np.transpose(Ba1),Cref),Ba2)
                                    *intScalFact*gWArray[j])
                    Kg = np.zeros([2,2])
                    
#                    print(np.matmul(np.matmul(basisdXArray[k,:],stress),basisdXArray[l,:].transpose())
#                                    *intScalFact*gWArray[j])
                    Kg[[0,1],[0,1]] = (np.matmul(np.matmul(basisdXArray[k,:],stress),basisdXArray[l,:].transpose())
                                    *intScalFact*gWArray[j])*6.95
                    
#                    print(intScalFact) 9.53366
                    tempKg[dim*k:dim*(k+1),dim*l:dim*(l+1)] += Kg
#                    print(tempKg)
                    tempKm[dim*k:dim*(k+1),dim*l:dim*(l+1)] += Km
                    
                    tempK[dim*k:dim*(k+1),dim*l:dim*(l+1)] += Km + Kg
                    
#                    print(Ba1,'\n',Ba2,'\n')

        index = indexAssemble(eleNodesArray[:,i],dim)                     
#        print(tempKg,'\n')
        K[index,np.transpose(index)] += tempK
    return(K)

def newtonRaph(dim,numEle,constit,gPArray,gWArray, 
               mCount, mSize,basisArray,nodeCoords,eleNodesArray,forces,forceType,disp,constraintes):
    eps = 10**-5
    Fext = fext.fextAssemble(dim,numEle,gWArray, mCount, mSize,basisArray,nodeCoords,eleNodesArray,forces,forceType)
    
    constrainte2D = constraintes.astype(bool)
    constraintes = constraintes.flatten().astype(bool) 
    
    #Iterate
    iMax =  1
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
            (Fint,VM) = fint.fintAssemble(dim,numEle,gWArray, mCount, mSize,basisArray,nodeCoords,eleNodesArray,constit,ui)
#            print(Fext - Fint)
            Ri = (Fext - Fint)[constrainte2D]
            
#            if np.linalg.norm(Ri) < eps*R0:
            if np.linalg.norm(Ri) < eps:
                print(i,'solved','Ri =',np.linalg.norm(Ri))

                break
            else:
                print('iteration =', i,'Ri =',np.linalg.norm(Ri))
#                print(i)
                K = kAssemble(dim,numEle,constit,gWArray,mCount, mSize, basisArray,nodeCoords,eleNodesArray,ui)

                deltU = np.linalg.solve(K[:,constraintes][constraintes,:],Ri)
#                print(i,deltU)
                ui[constrainte2D] = ui[constrainte2D] + deltU ################################################### I think deltU needs to be changed to frame
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
#    E = 1#modulus
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
        forces[0:numEle[0],side*dim+0]=-1e10
        #Apply constraints
        disp = np.zeros([len(nodeCoords[:,0]),dim])
        
        constraintes = np.ones([len(nodeCoords[:,0]),dim]) 
        constraintes[0:(numEle[0]+1)**2:numEle[0]+1,0] = 0 #mesh must be square
        constraintes[numEle[0]:(numEle[0]+1)**2:numEle[0]+1,1] = 0 #mesh must be square
        
        Fext = fext.fextAssemble(dim,numEle,gWArray, mCount, mSize,basisArray,nodeCoords,eleNodesArray,forces,forceType)

        
    elif case == 'patch':
#        nodeCoords[:,1] = nodeCoords[:,1]*((1-nodeCoords[:,0])*.4+1)
        # Construct force field
        forceType = np.zeros([np.prod(numEle),np.sum(mCount)])
        forces = np.zeros([np.prod(numEle),np.sum(mCount)*dim])
        
#        side = 2
#        forceType[numEle[0]*(numEle[1]-1):numEle[0]*numEle[1],side] = 3 #dim must equal 2
#        forces[numEle[0]*(numEle[1]-1):numEle[0]*numEle[1],side*dim+0]=1e10
        
        side = 4
        forceType[[24],side] = 3 #dim must equal 2
        forces[[24],side*dim+0]=1e10
#        forceType[[1,3],side] = 3 #dim must equal 2
#        forces[[1,3],side*dim+0]=1e10
#        side = 2
#        forceType[[2,3],side] = 3 #dim must equal 2
#        forces[[2,3],side*dim+1]=1e10
        disp = np.zeros([len(nodeCoords[:,0]),dim])
        constraintes = np.ones([len(nodeCoords[:,0]),dim]) 
        constraintes[0:numEle[0]+1,1] = 0
        constraintes[0:np.prod(numEle)+numEle[0]+1:numEle[0]+1,0] = 0
        
        Fext = fext.fextAssemble(dim,numEle,gWArray, mCount, mSize,basisArray,nodeCoords,eleNodesArray,forces,forceType)
        
    

#### 

###############################################################################
    if True:
        ui = newtonRaph(dim,numEle,constit,gPArray,gWArray, 
                   mCount, mSize,basisArray,nodeCoords,eleNodesArray,forces,forceType,disp,constraintes)
        Fint,vonM = fint.fintAssemble(dim,numEle,gWArray, mCount, mSize,basisArray,nodeCoords,eleNodesArray,constit,ui)
        mas.PlotStress(1, nodeCoords+ui,vonM,eleNodesArray)
        mas.plotFigure(1,nodeCoords)
        mas.plotFigure(1,nodeCoords+ui)
        
        temp = ui[0:np.prod(numEle)+numEle[0]+1:numEle[0]+1,1]
    
   