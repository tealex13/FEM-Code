# -*- coding: utf-8 -*-
"""
Created on Sat Jan 20 19:51:29 2018

@author: avila3
"""

import numpy as np
import unittest
import loadAssemble as load

class test_constraints(unittest.TestCase):
    
    def test_constraints(self):
        # Test case 1 - 2 Dimensional Array
        a, b = load.constraints(2,3, [0,1,1,2], [1,1,0,0],[10,-3,5,6])
        self.assertEqual(a.tolist(), 
                ([[-1.,  0.,  0.],[0., 0., -1.]]))
        self.assertEqual(b.tolist(), 
                ([[0.,   5.,   6.],[10.,   -3.,   0.]]))
        # Test case 1
        a, b = load.constraints(3,3, [0,1,1,2], [1,2,0,0],[10,-3,5,6])
        self.assertEqual(a.tolist(), 
                ([[-1.,  0.,  0.],[0., -1., -1.],[-1.,  0., -1.]]))
        self.assertEqual(b.tolist(), 
                ([[0.,   5.,   6.],[10.,   0.,   0.],[0.,  -3.,   0.]]))
#    np.array( [[-1.  0.  0.][ 0. -1. -1.][-1.  0. -1.]]))



if __name__ == '__main__':
    unittest.main()
#    
#   dim = 3
#    numNodes = 3
#    nodeArray = np.array([0,1,1,2])
#    dimArray = np.array([1,2,0,0])
#    values = np.array([10,-3,5,6])