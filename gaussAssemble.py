# -*- coding: utf-8 -*-
"""
Created on Tue Jan 30 13:08:29 2018

@author: avila3
"""

import numpy as np
import matplotlib.pyplot as plt 
import meshAssemble as mas



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
        
def func(x):
    return 1


#generate Gauss points
dim = 2

gaussPoints = np.array([-1/np.sqrt(3),1/np.sqrt(3)])
#gaussPointNode = np.array([-1/np.sqrt(3)])
gaussWeights = np.array([1,1])
endPoints = np.array([-1,1])
gaussPointNodes = np.arange(len(gaussPoints))

numGP = len(gaussPointNodes)
stackSize = len(endPoints)*numGP**(dim-1)
stackSize1 = numGP**(dim-np.arange(dim))*len(endPoints)**np.arange(dim)

#tempNumNodes = numGP**dim+dim*(dim-1)*stackSize



memberCount = numCubeMembers(dim) #The number of each members(...4D-vol,)
tempNumNodes = int(np.sum(memberCount*numGP**np.arange(dim,0,-1)))

gaussPointNodeArray = np.zeros([tempNumNodes,dim])
#gaussWeightsArray = np.zeros([tempNumNodes,1])

currentIndex = stackSize1[0] #Keep track of which cells we are filling
gaussPointNodeArray[:currentIndex,:] = mas.cartesian(np.matlib.repmat(gaussPointNodes,dim,1))

for i in range(1,dim): #Iterate through each dimension
    
    temp = [gaussPointNodes.tolist()]*(dim-i)
    temp.extend([np.arange(dim-1,dim+len(endPoints)-1).tolist()]*i)#Tack on node numbers for the endpoints
    temp = mas.cartesian(np.array(temp)) #Create a matrix with a missing row
    
    gaussPointNodeArray[currentIndex:currentIndex+stackSize1[i]]=temp #Record
    currentIndex = currentIndex+stackSize1[i]
#    print(currentIndex,numGP**dim+stackSize*dim*(i-1),"\n")
    for j in range(dim-i): #iterate through each row
        
        for k in range(i): #iterate through each rearrangement
#            print(i,j,k,"\n")
#            print(k-(i+j),k-(i+j+1))
            temp[:,[k-(i+j),k-(i+j+1)]] = temp[:,[k-(i+j+1),k-(i+j)]]
            gaussPointNodeArray[currentIndex:currentIndex+stackSize1[i]]=temp
            currentIndex = currentIndex+stackSize1[i]
#            print(currentIndex,numGP**dim+stackSize*(1+k+j+dim*(i-1)),"\n")


# Assemble the guass Point Array and Weight Array using gaussPointNodeArray to index
gaussPointsArray = np.append(gaussPoints,endPoints)[gaussPointNodeArray.astype(int)]
gaussWeightsArray = np.product(np.append(gaussWeights,np.ones([1,len(endPoints)])) 
                               [gaussPointNodeArray.astype(int)],1)
   

#def guassIntegrate():

#if __name__ == "__main__":
#    memberCount = numCubeMembers(1)
#    print(memberCount)