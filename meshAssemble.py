# -*- coding: utf-8 -*-
"""
Created on Fri Jan 19 16:53:15 2018

@author: avila3
"""

import numpy as np
import matplotlib.pyplot as plt 

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
    return out

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

def meshAssemble(dim,m,M,n=1,N=1,o=1,O=1):
    stepx = M/m
    stepy = N/n;
    stepz = O/o;
    
#    width = [M,N,O][:dim]
    numEle = [m,n,o][:dim]    
    coordsArray = (
            np.arange(m+1)*stepx,np.arange(n+1)*stepy,np.arange(o+1)*stepz)[:dim] 

    eleNodesArray = eleNodes(numEle)   
    nodeCoords = cartesian(coordsArray)
    edgeNodesArray = edgeNodes(dim,eleNodesArray)

    return (nodeCoords,eleNodesArray,edgeNodesArray)
    
#if __name__ == "__main__":
#    meshAssemble(2,3,1,n=2,N=1,o=1,O=1)