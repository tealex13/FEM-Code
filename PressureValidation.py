# -*- coding: utf-8 -*-
"""
Created on Sat Apr 14 14:31:23 2018

@author: avila3
"""

import numpy as np
import matplotlib.pyplot as plt 
import meshAssemble as mas


E = 1#modulus
v = 0.0 #poisons ratio
Ri = 1 #Inside Radius
Ro = 2 #Outside Radius
P = .063653 #Pressure inside
P = .1
eleNum = 20 #number of elements in the radial direction


R = np.linspace(Ri,Ro,num = eleNum+1)
#Radial displacement
ur =  1/E*(P*Ri**2)/(Ro**2-Ri**2)*((1-v)*R+Ro**2*(1+v)/R)

mas.plotFigure(1,np.array([np.zeros(len(R)),R+ur]).transpose())