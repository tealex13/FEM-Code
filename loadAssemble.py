# -*- coding: utf-8 -*-
"""
Created on Sat Jan 20 19:03:56 2018

@author: avila3
"""

import numpy as np

def constraints(dim, numNodes, nodeArray, dimArray, values, pressure = False):
    # Assembles two matracies:
    # returns:
    # constHas indecates which nodes have constraints
    # constVal indecates the constraint values at the node
    #
    # Inputs:
    # dim = int dimensions of the mesh.
    # numNodes = total number of nodes in mesh.
    # nodeArray = array of the nodes with constraints.
    # dimArray = array of the constrained dimension that corresponds with the 
    #            nodes in nodeArray.
    # values = values of the constraints correspoinding with the nodes and 
    #          dimensions in nodeArray and dimArray
    
    ### Test for incorrect dimensions
    exception = False 
        
    if len(nodeArray)!=len(values):
        print("ERROR: nodeArray is not same length as values")
        exception = True
        return (np.zeros(1),np.zeros(1))
    
    if any(np.array(dimArray) > dim-1):
        print("ERROR: dimension in dimArray outside of dimension dim")
        exception = True
        return (np.zeros(1),np.zeros(1))
    
    ### Assemble the array
    if not exception:    
        constHas = np.ones((dim,numNodes))*-1
        constVal = np.zeros((dim,numNodes))
        
        if pressure:
            constHas[:,np.asarray(nodeArray).astype(int)] = 0
            constVal[:,np.asarray(nodeArray).astype(int)] = np.asarray(values)
        else:
            constHas[np.asarray(dimArray).astype(int),np.asarray(nodeArray).astype(int)] = 0
            constVal[np.asarray(dimArray).astype(int),np.asarray(nodeArray).astype(int)] = np.asarray(values)

    #    print(constHas,"\n")
    #    print(constVal,"\n")
        return (constHas,constVal)
#    
if __name__ == "__main__":
    dim = 3
    numNodes = 4
    nodeArray = np.array([0,1,1,2])
    dimArray = np.array([1,2,-1,0])
    values = np.array([10,-3,5,6])
    (constHas,constVal) = constraints(dim, numNodes, nodeArray, dimArray, values)
