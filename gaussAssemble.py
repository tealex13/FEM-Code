# -*- coding: utf-8 -*-
"""
Created on Tue Jan 30 13:08:29 2018

@author: avila3
"""

import numpy as np
import matplotlib.pyplot as plt 
import meshAssemble as mas
import CtoX as cx



def numCubeMembers(n):
    #This function returns a count list of each type of member 
    #(..., 4D-volume,volume lines,points) in each a n-dimensional cube.
    #Input: n is the dimension of the cube.
    #Output: list 
    memberCount = np.ones(n+1)
    memberCount[0] = 2
    for i in range(1,n):
        memberCount[:i+1] = memberCount[:i+1]*2+np.append(0,memberCount[:i])  
   
    return memberCount[:0:-1]

def parEleGPAssemble(dim, gaussPoints = [-1/np.sqrt(3),1/np.sqrt(3)], gaussWeights = [1,1] ):
    #generate Gauss points Array, weights array, the number of each member and the length of each member
#    dim = 2
    
    gaussPoints = np.array(gaussPoints)
    gaussWeights = np.array(gaussWeights)
    
    endPoints = np.array([-1,1])
    gaussPointNodes = np.arange(len(gaussPoints))
    
    numGP = len(gaussPointNodes)
    stackSize1 = numGP**(dim-np.arange(dim))*len(endPoints)**np.arange(dim)
    
    memberCount = numCubeMembers(dim) #The number of each members(...4D-vol,)
    memberSize = numGP**np.arange(dim,0,-1) #The number nodes to itegrate across for each member
    tempNumNodes = int(np.sum(memberCount*memberSize))
    
    gaussPointNodeArray = np.zeros([tempNumNodes,dim])
    #gaussWeightsArray = np.zeros([tempNumNodes,1])
    
    currentIndex = stackSize1[0] #Keep track of which cells we are filling
    gaussPointNodeArray[:currentIndex,:] = mas.cartesian(np.matlib.repmat(gaussPointNodes,dim,1))
    
    for i in range(1,dim): #Iterate through each dimension
        
        temp = [gaussPointNodes.tolist()]*(dim-i)
        temp.extend([np.arange(numGP,numGP+len(endPoints)).tolist()]*i)#Tack on node numbers for the endpoints
        temp = mas.cartesian(np.array(temp)) #Create a matrix with a missing row
        
        gaussPointNodeArray[currentIndex:currentIndex+stackSize1[i]]=temp #Record
        currentIndex = currentIndex+stackSize1[i]
        for j in range(dim-i): #iterate through each row
            
            for k in range(i): #iterate through each rearrangement
    
                temp[:,[k-(i+j),k-(i+j+1)]] = temp[:,[k-(i+j+1),k-(i+j)]]
                gaussPointNodeArray[currentIndex:currentIndex+stackSize1[i]]=temp
                currentIndex = currentIndex+stackSize1[i]
    
    # Assemble the guass Point Array and Weight Array using gaussPointNodeArray to index
    gaussPointsArray = np.append(gaussPoints,endPoints)[gaussPointNodeArray.astype(int)]
    gaussWeightsArray = np.product(np.append(gaussWeights,np.ones([1,len(endPoints)])) 
                                   [gaussPointNodeArray.astype(int)],1)
    #tack on a zero in front of memberCount and memberSize for easier refrencing
    memberCount = np.insert(memberCount,0,0).astype(int)
    memberSize = np.insert(memberSize,0,0).astype(int)
    return (gaussPointsArray,gaussWeightsArray,memberCount,memberSize)   

def gaussIntegrate(numEle,final,dimMem,func):
    """
    This function performs gauss quadriture integration
    
    INPUTS:
        numEle- number of elements in each dimension, list form
        final- size of space in each dimension, list
        dimMem- dimension of the members to integrate, int
        func- function to be integrated, inputs x for func(x) should be of the
                form [[x1,y1,z1,...],[x2,y2,z2,...],...]
    
    OUTPUTS:
        integral- sum of the all the integrated members across all elements
    """
    dim = len(numEle)
    (coords,eleNodes,edges) = mas.meshAssemble(numEle,final)
    (gPArray,gWArray, mCount, mSize) = parEleGPAssemble(dim)

    integral = 0
    for j in range(np.prod(numEle)): #Iterate through each element
        for i in range(mCount[-dimMem]): #Iterate through each member
            #select the correct gP set (this will change with each member)
            startingPoint = np.sum(mCount[:-dimMem]*mSize[:-dimMem]).astype(int)+mSize[-dimMem]*i
            pointsToEval = gPArray[startingPoint:mSize[-dimMem]+startingPoint,:]
            weights = gWArray[startingPoint:mSize[-dimMem]+startingPoint,]
            #get the left and right node values (this will also change with each member)
            
            nodeCoordsLeft = np.matlib.repmat(coords[eleNodes[0,j],:],len(pointsToEval[:,0]),1)
            nodeCoordsRight = np.matlib.repmat(coords[eleNodes[-1,j],:],len(pointsToEval[:,0]),1)
            #print(nodeCoordsLeft,'\n',nodeCoordsRight)
            xValues = cx.CtoX(pointsToEval,nodeCoordsLeft,nodeCoordsRight)
            integral  = sum(func(xValues)*weights)*np.linalg.det(cx.jacobian(nodeCoordsLeft[0,:],nodeCoordsRight[0,:]))+integral

    return integral




if __name__ == "__main__":
    memberCount = numCubeMembers(1)
    print(memberCount)
    (a,b,c,d) = parEleGPAssemble(3)
    
    numEle = [1,1,1]
    final = [1,1,1]
    func = lambda x: np.ones(len(x))
    #func = lambda x: x[:,0]**3
    #func = lambda x: np.prod(x,axis=1)
    dimMem = 3 # Dimension of member being integrated
    
    
    integral = gaussIntegrate(numEle,final,dimMem,func)