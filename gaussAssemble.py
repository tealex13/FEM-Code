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

#def guassIntegrate():

init = 0# not necissary all start at zero because of meshAssble code
numEle = [1,1,1]
final = [1,1,1]
dim = len(numEle)
#func = lambda x: np.ones(len(x))
func = lambda x: x[:,0]**3
dimMem = dim # Dimension of member being integrated



(coords,eleNodes,edges) = mas.meshAssemble(numEle,final)
(gPArray,gWArray, mCount, mSize) = parEleGPAssemble(dim)
sideNodes = mas.sideNodes(dim,dimMem)

tempSum = 0
#print (np.array(eleNodes[:,0]))
#for i in range(mCount[dimMem])
j = 0 #which element we are on
i = 0 #which member we are on
#select the correct gP set (this will change with each member)
startingPoint = np.sum(mCount[:-dimMem]*mSize[:-dimMem]).astype(int)+mSize[-dimMem]*i
pointsToEval = gPArray[startingPoint:mSize[-dimMem]+startingPoint,:]
weights = gWArray[startingPoint:mSize[-dimMem]+startingPoint,]
#get the left and right node values (this will also change with each member)

nodeCoordsLeft = np.matlib.repmat(coords[eleNodes[0,j],:],len(pointsToEval[:,0]),1)
nodeCoordsRight = np.matlib.repmat(coords[eleNodes[-1,j],:],len(pointsToEval[:,0]),1)
#print(nodeCoordsLeft,'\n',nodeCoordsRight)
xValues = cx.CtoX(pointsToEval,nodeCoordsLeft,nodeCoordsRight)
integral  = sum(func(xValues)*weights)*np.linalg.det(cx.jacobian(nodeCoordsLeft[0,:],nodeCoordsRight[0,:]))

#print(cx.CtoX(coords[np.array(eleNodes[:,0])]))

#for i in range(numEle):
#    tempSum = func(cx.CtoX(coords(eleNodes(i))))
#if __name__ == "__main__":
#    memberCount = numCubeMembers(1)
#    print(memberCount)
#    (a,b,c,d) = parEleGPAssemble(3)