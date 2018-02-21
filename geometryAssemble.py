# -*- coding: utf-8 -*-
"""
Created on Sat Feb 17 13:52:16 2018

@author: Avila
"""
import numpy as np
from numpy import linalg as LA
import loadAssemble as load
import meshAssemble as mas
import gaussAssemble as gas
import basisAssemble as bas
import CtoX as cx

def CtoX(C,eleNodesArray,nodeCoords):
    '''
    Unlike gausstoX this function does all the calculations for determining x internally,
    instead of premptively. Just remember to change the basis if they are changed. 
    OUTPUT:
        x- the coordinate of the gauss points captured in basisSubset in the xyz frame
            format: [[GP1],[GP2],[GP3],...] where GP are gauss points [x,y,z,...]
    '''
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

def memDim(S,dim,mCount):
    '''
    Function returns the dimensions of each member S
    '''
#    Sdim = 0
    for i in range(dim):
        if 0 > S - np.sum(mCount[:i+2]):
            Sdim = dim-i
            break
        elif i == dim-1:
            print('error in memDim')
            Sdim = 0
    return Sdim   

def basisSubsetAssemble(S,dim,basisArray,mCount,mSize,divDim = 0):
    '''
    Function returns a single basis products from each gauss point in member S 
    '''
#    basisSubset = np.zeros([len(basisArray[:,0]),mSize])
    sDim = memDim(S,dim,mCount)
    startingPoint = np.sum(mSize[:-sDim])*(dim+1)+(S-np.sum(mCount[:-sDim]))*(dim+1)
    endingPoint = startingPoint+mSize[-sDim]*(dim+1)
#    print(np.arange(startingPoint,endingPoint,dim+1))
     
    basisSubsetGP = basisArray[:,np.arange(startingPoint+divDim,endingPoint+divDim,dim+1)]
        
    return(basisSubsetGP)
    
def basisSubsetGaussPoint(S,gaussPoint,dim,basisArray,mCount,mSize):
    '''
    Function returns all of the basis products of a gauss point in member S
    '''
#    basisSubset = np.zeros([len(basisArray[:,0]),mSize])
    sDim = memDim(S,dim,mCount)
    startingPoint = np.sum(mSize[:-sDim])*(dim+1)+gaussPoint*(dim+1)
    endingPoint = startingPoint+(dim+1)
#    print(np.arange(startingPoint,endingPoint,dim+1))
     
    basisSubset = basisArray[:,startingPoint:endingPoint]
        
    return(basisSubset)

def basisSelect(S,gaussPoint,dim,basisArray,mCount,mSize,divDim=0):
    '''
    Function returns just the basis column of the gaussPoint
    '''
    return(basisSubsetAssemble(S,dim,basisArray,mCount,mSize,divDim)[:,gaussPoint]) 


def gausstoX(basisSubset,ele,eleNodesArray,nodeCoords):
    '''
    OUTPUT:
        x- the coordinate of the gauss points captured in basisSubset in the xyz frame
            format: [[GP1],[GP2],[GP3],...] where GP are gauss points [x,y,z,...]
    '''
    x = np.zeros([len(basisSubset[0,:]),dim])
    for i in range(len(basisSubset[0,:])): #Step through set of basis for each Gauss Point   
        x[i,:] = np.sum(basisSubset[:,i].reshape(len(eleNodesArray[:,ele]),1)*nodeCoords[eleNodesArray[:,ele],:],0) #Sum each row
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
    
def detAssemble(dim,mCount):
    '''
    Function creats a matrix of the partial derivatives included in each side when 
    assembling the jacobian for each member.
    
    OUTPUT:
        detArray- a matrix of ones and zeros [[1,1,1],[0,1,1],[0,1,1],[1,0,1],[1,0,1],...]
            each row corresponds with a member S.
    '''
    
    detArray = np.empty((0,dim),int)
    reps = [1]+(mCount[2:]/dim).astype(int).tolist()
    for i in range(dim,0,-1):
        temp = combonator(i,dim)
        temp = np.repeat(temp,[reps[-i]]*len(temp[:,0]),axis=0)
        detArray = np.append(detArray,temp,axis=0)
    return detArray

    
def gaussJacobian(S,ele,gaussPoint,dim,basisArray,mCount,mSize,detArray):
    '''
    Returns the jacobian matrix of the gauss point for member S of elemenet ele
    INPUT:
        S- member number
        ele- element number
        gaussPoint- gauss point number
    OUTPUT:
        intScalFact- The Integral Scaling Factor
        hardCodedJac- The 3x3 jacobian matrix of the form [[dx/dn,dx/dm,dx/do],[dy/dn,dy/dm,dy/do],[dz/dn,dz/dm,dz/do]]
    '''
        
    intScalFact = np.zeros([dim,dim])
    hardCodedJac = np.zeros([3,3])
    for i in range(dim): #step through each parent dimension
        
        intScalFact[:,i] = np.sum(basisSelect(S,gaussPoint,dim,basisArray,mCount,mSize,i+1).reshape(len(eleNodesArray[:,ele]),1)*
                nodeCoords[eleNodesArray[:,ele],:],0)
#        hardCodedJac[:,i]
    hardCodedJac[:dim,:dim] = intScalFact
    intScalFact = intScalFact[detArray[S,:].astype(bool),:]
    intScalFact = LA.det(intScalFact[:,detArray[S,:].astype(bool)])
    return (intScalFact,hardCodedJac)
        
        
def stupidNormals(S,hardCodedJac,dim):
    '''
    Hard Coded normal calculator for the different members
    for dim 1 there are no normals, for dim = 2 the normals are to the lines
    for dim = 3 the normals are to the faces only
    
    OUTPUT:
        normal- is the normal vector to a member that has been normalized 
            it is an np.array of the form [x,y,z]
    '''
    
    if dim == 1:
        normal = []
        print('error: no normals for dim=1')
    elif dim == 2:
        if S == 0:
            normal = []
            print('error: no normals for S = 0')
        elif S == 1:
            temp = np.cross(np.array([0,0,1]),hardCodedJac[:,1])
            normal = 1/LA.norm(temp)*temp
        elif S == 2:
            temp = np.cross(hardCodedJac[:,1],np.array([0,0,1]))
            normal = 1/LA.norm(temp)*temp
        elif S == 3:
            temp = np.cross(hardCodedJac[:,0],np.array([0,0,1]))
            normal = 1/LA.norm(temp)*temp
        elif S == 4:
            temp = np.cross(np.array([0,0,1]),hardCodedJac[:,0])
            normal = 1/LA.norm(temp)*temp
        else:
            normal = []
            print('error: S outside of range')
    elif dim == 3:
        if S == 0:
            normal = []
            print('error: no normals for S = 0')
        elif S == 1:
            temp = np.cross(hardCodedJac[:,2],hardCodedJac[:,1])
            normal = 1/LA.norm(temp)*temp
        elif S == 2:
            temp = np.cross(hardCodedJac[:,1],hardCodedJac[:,2])
            normal = 1/LA.norm(temp)*temp
        elif S == 3:
            temp = np.cross(hardCodedJac[:,0],hardCodedJac[:,2])
            normal = 1/LA.norm(temp)*temp
        elif S == 4:
            temp = np.cross(hardCodedJac[:,2],hardCodedJac[:,0])
            normal = 1/LA.norm(temp)*temp
        elif S == 5:
            temp = np.cross(hardCodedJac[:,1],hardCodedJac[:,0])
            normal = 1/LA.norm(temp)*temp
        elif S == 6:
            temp = np.cross(hardCodedJac[:,0],hardCodedJac[:,1])
            normal = 1/LA.norm(temp)*temp
        else:
            normal = []
            print('error: S outside of range')
    return(normal)

def basisToX(S,gaussPoint,dim,basisArray,mCount,mSize,hardCodedJac):
    basisSubsetGP = basisSubsetGaussPoint(S,gaussPoint,dim,basisArray,mCount,mSize)
    basisToXArray = np.zeros(basisSubsetGP[:,1:].shape)
    for i in range(len(basisArray[:,0])):
        basisToXArray[i,:] = np.matmul(basisSubsetGP[i,1:],LA.inv(hardCodedJac[:dim,:dim]))
        print(basisSubsetGP[i,1:],'\n')
    return(basisToXArray)

dim = 2
numEle = [1]*dim
eleSize = [1]*dim
numBasis = 2
gaussPoints = [-1/np.sqrt(3),1/np.sqrt(3)]
#gaussPoints = [-1,1]
(gPArray,gWArray, mCount, mSize) = gas.parEleGPAssemble(dim,gaussPoints =gaussPoints)
basisArray = bas.basisArrayAssemble(dim,numBasis,gaussPoints,gPArray,gWArray, mCount, mSize)

(nodeCoords,eleNodesArray,edgeNodesArray) = mas.meshAssemble(numEle,eleSize)

#print(CtoX([0,0],eleNodesArray,nodeCoords))
detArray = detAssemble(dim,mCount)

ele = 0
S = 0
gaussPoint = 0


(intScalFact,hardCodedJac) = gaussJacobian(S,ele,gaussPoint,dim,basisArray,mCount,mSize,detArray)
temp = basisToX(S,gaussPoint,dim,basisArray,mCount,mSize,hardCodedJac)
print(temp)
#basisSubset = basisSubsetAssemble(0,dim,basisArray,mCount,mSize,divDim = 0)
#print(gausstoX(basisSubset,ele,eleNodesArray,nodeCoords))
#for i in range(0,8):
#    (intScalFact,hardCodedJac) = gaussJacobian(i,ele,gaussPoints[0],dim,basisArray,mCount,mSize,detArray)
#    print(intScalFact)
#
#    normal = stupidNormals(i,hardCodedJac,dim)
#    print(normal)
