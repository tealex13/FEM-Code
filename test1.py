# -*- coding: utf-8 -*-
"""
Created on Sat Jan 20 19:51:29 2018

@author: avila3
"""

import numpy as np
import unittest
import loadAssemble as load
import meshAssemble as mas
import gaussAssemble as gas

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
        
    def test_sideNodes(self):
        a = mas.sideNodes(3,2).tolist()
        b = [[0.,  4.,  0.,  2.,  0.,  1.],[ 1.,  5.,  1.,  3.,  2.,  3.],[ 2.,  6.,  4.,  6.,  4.,  5.],[ 3.,  7.,  5.,  7.,  6.,  7.]]
        self.assertEqual(a,b)

    def test_guassIntegrate(self):
        

        func = lambda x: np.ones(len(x))
        self.assertEqual(gas.gaussIntegrate([1],[1],1,func),1)    
        self.assertEqual(gas.gaussIntegrate([1,1],[1,1],2,func),1)     
        self.assertAlmostEqual(gas.gaussIntegrate([1,1,1],[1,1,1],3,func),1,places=14)
        self.assertAlmostEqual(gas.gaussIntegrate([2,3,4],[1,1,1],3,func),1,places=14)
        self.assertEqual(gas.gaussIntegrate([1,1,1],[2,1,1],3,func),2)
        self.assertEqual(gas.gaussIntegrate([1,1,1],[1,2,1],3,func),2)
        self.assertEqual(gas.gaussIntegrate([1,1,1],[1,1,2],3,func),2)
        self.assertEqual(gas.gaussIntegrate([1,1,1],[2,2,3],3,func),12)
#        self.assertEqual(gas.gaussIntegrate([1,1,1],[2,1,1],2,func),2)
        
        func = lambda x: x[:,0]**3
        self.assertAlmostEqual(gas.gaussIntegrate([1],[1],1,func),1/4,places=14)    
        self.assertEqual(gas.gaussIntegrate([1,1],[1,1],2,func),1/4)     
        self.assertAlmostEqual(gas.gaussIntegrate([1,1,1],[1,1,1],3,func),1/4,places=14)
        self.assertAlmostEqual(gas.gaussIntegrate([2,3,4],[1,1,1],3,func),1/4,places=14)
        self.assertEqual(gas.gaussIntegrate([1,1,1],[2,1,1],3,func),4)
        self.assertEqual(gas.gaussIntegrate([1,1,1],[1,2,1],3,func),1/2)
        self.assertEqual(gas.gaussIntegrate([1,1,1],[1,1,2],3,func),1/2)
        self.assertEqual(gas.gaussIntegrate([1,1,1],[2,2,3],3,func),24)
#        self.assertEqual(gas.gaussIntegrate([1,1,1],[2,1,1],2,func),2)
        
        func = lambda x: np.prod(x,axis=1)
        self.assertAlmostEqual(gas.gaussIntegrate([1],[1],1,func),1/2,places=14)    
        self.assertEqual(gas.gaussIntegrate([1,1],[1,1],2,func),1/4)     
        self.assertAlmostEqual(gas.gaussIntegrate([1,1,1],[1,1,1],3,func),1/8,places=14)
        self.assertAlmostEqual(gas.gaussIntegrate([2,3,4],[1,1,1],3,func),1/8,places=14)
        self.assertAlmostEqual(gas.gaussIntegrate([1,1,1],[2,1,1],3,func),1/2,places=14)
        self.assertAlmostEqual(gas.gaussIntegrate([1,1,1],[1,2,1],3,func),1/2,places=14)
        self.assertAlmostEqual(gas.gaussIntegrate([1,1,1],[1,1,2],3,func),1/2,places=14)
        self.assertAlmostEqual(gas.gaussIntegrate([1,1,1],[2,2,3],3,func),18,places=13)
        
        # Still need to test where dimMem is less than dim

if __name__ == '__main__':
    unittest.main()
#    
#   dim = 3
#    numNodes = 3
#    nodeArray = np.array([0,1,1,2])
#    dimArray = np.array([1,2,0,0])
#    values = np.array([10,-3,5,6])