# -*- coding: utf-8 -*-
"""
Created on Fri Jan 19 16:53:15 2018

@author: avila3
"""

import numpy as np
import matplotlib.pyplot as plt 
import itertools

def binaryCounterMatrix(dim):
    binMat = np.zeros([2**dim,dim])
    count = 0
    for i in itertools.product([0,1],repeat=dim):
        binMat[count,:] = list(i)
        count = count+1
    
    return binMat.astype(int)

def eleNodes(arrays, out=None):
    """
    eleNodes assigns nodes to elements
    Input: array "arrays" [a, b, c, ... d] where a,b,c,d are the number 
    of elements in each of the dimension.
    Output: array where the coloums are the node values associated with each 
    element.
    """
    
    arrays = np.asarray(arrays)
    n = np.prod(arrays) # Total number of elements
    if out is None:
        out = np.zeros([2**len(arrays), n])
        
        
    m = int(n / arrays[-1]) #number of elements in previous dimensions
    o = np.prod(arrays[:-1]+1) #number of nodes in previous dimensions
    if any(arrays[0:-1]):
        eleNodes(arrays[:-1], out=out[:int(len(out)/2):,:m])
        for i in range(arrays[-1]):
            out[int(len(out)/2):,i*m:(i+1)*(m)] = out[:int(len(out)/2):,:m]+ (o)*(i+1)
            if i+1 != arrays[-1]:
                out[:int(len(out)/2),(i+1)*(m):(i+2)*(m)] = out[:int(len(out)/2):,:m]+ (o)*(i+1)
    else:
        out[:2,:arrays[0]+1] = np.vstack((np.arange(arrays[0]),np.arange(1,arrays[0]+1)))
    return out.astype(int)

def cartesian(arrays, out=None):
    """
    Modified from pv. code found at 
    https://stackoverflow.com/questions/1208118/using-numpy-to-build-an-array-of-all-combinations-of-two-arrays
    
    Generate a cartesian product of input arrays.

    Parameters
    ----------
    arrays : list of array-like
        1-D arrays to form the cartesian product of.
    out : ndarray
        Array to place the cartesian product in.

    Returns
    -------
    out : ndarray
        2-D array of shape (M, len(arrays)) containing cartesian products
        formed of input arrays.

    Examples
    --------
    >>> cartesian(([1, 2, 3], [4, 5], [6, 7]))
    array([[1, 4, 6],
           [1, 4, 7],
           [1, 5, 6],
           [1, 5, 7],
           [2, 4, 6],
           [2, 4, 7],
           [2, 5, 6],
           [2, 5, 7],
           [3, 4, 6],
           [3, 4, 7],
           [3, 5, 6],
           [3, 5, 7]])

    """

    arrays = [np.asarray(x) for x in arrays]
    dtype = arrays[0].dtype

    n = np.prod([x.size for x in arrays])
    if out is None:
        out = np.zeros([n, len(arrays)], dtype=dtype)

    m = int(n / arrays[-1].size)
    out[:,-1] = np.repeat(arrays[-1], m)
    if arrays[0:-1]:
        cartesian(arrays[0:-1], out=out[0:m,0:-1])
        for j in range(1, arrays[-1].size):
                        out[j*m:(j+1)*m,0:-1] = out[0:m,0:-1]
    return out 

def edgeNodes(dim,eleNodesArray):
    (nodeVal,nodeCount) = np.unique(eleNodesArray,return_counts=True)
    return nodeVal[nodeCount<=2**(dim-1)]

def sideNodes(dim,dimMem):
    # sNodes generates the list of nodes in each "side" for each member of dimensions
    # dimInt
    #
    # Inputs:
    #   dim- Dimensions of the element
    #   dimInt- Dimension of the member we are interested in, 1 for a line
    #       2 for a surface, 3 for volume, etc.
    # Outputs:
    #   sNodes

    dimInc = 2**(np.arange(dim,0,-1)-1)
    binMat = binaryCounterMatrix(dim)
    const = binMat[np.sum(binMat,1)== dimMem,:].astype(bool)
    uncon = np.logical_not(const)
    
    temp = np.zeros([dim,len(const)])
    for i in range(len(const)):
        temp[:,i] = np.append(dimInc[uncon[i,:]],dimInc[const[i,:]])
    listofNum = np.matmul(binMat,temp)
    listofNum = np.reshape(listofNum,(2**dimMem,int(len(const)*2**(dim-dimMem))), order='F')
    return listofNum.astype(int)
    
    


def meshAssemble(numEle,eleSize):
    # This is the main function
    # Input:
    # dim- dimensions
    # m,n,o number of elements in the x,y,z directions respectively.
    # M,N,O total length in the x,y,z direrctions respectively.
    # Outputs:
    # nodeCoords- the actual location of the nodes in cartesian space. In the form
    #   [[x1,y1,z1,...],[x2,y2,z2,],[x3,y3,z3],...]
    # eleNodesArray-  an array that relates which nodes are in each element.
    #   Each column is a different elements for example col=0 is the element 0.
    # edgeNodesArray- array of nodes on the corner,edge,and outside surfaces.
    
    dim = len(numEle)
#    stepx = M/m
#    stepy = N/n;
#    stepz = O/o;
    
#    width = [M,N,O][:dim]
    numEle = np.array(numEle)
    eleSize = np.array(eleSize)
    coordsArray = []
    for i in range(dim):   
        coordsArray.append((np.arange(numEle[i]+1)*eleSize[i]/numEle[i]).tolist())
#            np.arange(m+1)*stepx,np.arange(n+1)*stepy,np.arange(o+1)*stepz)[:dim] 

    eleNodesArray = eleNodes(numEle).astype(int)   
    nodeCoords = cartesian(coordsArray)
    edgeNodesArray = edgeNodes(dim,eleNodesArray).astype(int)

    return (nodeCoords,eleNodesArray,edgeNodesArray)


    
if __name__ == "__main__":
#    a = meshAssemble([1,2,3],[1,1,1])
#    a = sideNodes(2,1)
    y = lambda x: 1
    n = lambda x: 2
    q = lambda x: x
    b = lambda x: x*2
    a = cartesian(([y,n],[q,b]))