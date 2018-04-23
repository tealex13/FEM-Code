# -*- coding: utf-8 -*-
"""
Created on Wed Apr 18 11:49:00 2018

@author: avila3
"""
 
import numpy as np
import matplotlib as plt
import meshAssemble as mas
import gaussAssemble as gas
import basisAssemble as bas
import geometryAssemble as geas
import fextAssemble as fext
import loadAssemble as load

def toVoigt (matrix):
    if len(matrix) == 2:
        matrix = np.array([matrix[0,0],matrix[1,1],matrix[0,1]])
    return(matrix)
def fromVoigt(matrix):
    if len(matrix) == 3:
        matrix = np.array([[matrix[0],matrix[2]],[matrix[2],matrix[1]]])
    return(matrix)
    
def partDeformationGrad(basisdXArray, ui):
    print(np.matmul(ui.transpose(),basisdXArray),'\n')
    return np.matmul(ui.transpose(),basisdXArray)

def deformationGrad(dUdX):
    F = np.identity(len(dUdX))+dUdX
    J = np.linalg.det(F)
    return(F,J)
    
def constitutiveCreater(F,J,C):
    findex = np.array([[1,1,1,1],[1,1,2,2],[1,1,1,2],[2,2,1,1],[2,2,2,2],[2,2,1,2],[1,2,1,1],[1,2,2,2],[1,2,1,2]])-1 
    
    Cref = np.zeros([9,1]) #hardcoded
    for i in range(len(Cref)):
        fTemp = np.prod(F[findex,findex[i,:]],axis = 1)
        Cref[i] = np.matmul(C.reshape((1,9)),fTemp)*1/J
#        print(fTemp,'\n')
    return (Cref.reshape((3,3)))

def greenLagrangeStrain(dUdX):
    # Returns green strain in voigt notation
    
    gStrain = 1/2*(dUdX+dUdX.transpose()+np.matmul(dUdX.transpose(),dUdX))
    if len(gStrain) == 2:
        return (np.array([gStrain[0,0],gStrain[1,1],gStrain[0,1]]))
    else:
        print('error: not 2d')
        return ([])

def pkStress(gStrain,Cref):

    pkS = fromVoigt(np.matmul(Cref,gStrain))
    
    return(pkS)
    
def coachyStress(pkS,F,J):
    cauchyStr = toVoigt(1/J*np.matmul(np.matmul(F,pkS),F.transpose()))
    return(cauchyStr)
if __name__ == "__main__":
    a = 1
