# -*- coding: utf-8 -*-
"""
Created on Sat May  7 11:46:16 2022

@author: j.reul
"""

import unittest
import numpy as np

import mode_behave_public as mb

class TestCarOwnership(unittest.TestCase):
    
    def test_simulate_hh_cars(self):
        """
        Integration test of the method -simulate_hh_cars()-

        """
        #Define the input data
        regiontype = 1
        hh_size = 2
        adults_working = 1 
        children = 0
        htype = 1
        quali_opnv = 2 
        sharing = 1
        relative_cost_per_car = 2 
        age_adults = 3.5
        
        #simulate
        model = mb.Core(model_type = 'simulation', simulation_type = 'car_ownership')        
        result = model.simulate_hh_cars(regiontype, hh_size,
                             adults_working, children, htype, quali_opnv, sharing,
                             relative_cost_per_car, age_adults)
                 
        #test
        self.assertTrue(
            np.allclose(
            result, 
            [0.500314856405928,
             0.43400343757783405,
             0.06463228081919556,
             0.0010494251970423272],
            atol=0.01)
            )      
        
    def test_simulate_mode_choice(self):
        """
        Integration test of the method -test_simulate_mode_choice()-.

        """
        #simulate
        model = mb.Core(model_type = 'simulation', simulation_type = 'mode_choice')        
        result = model.simulate_mode_choice(
            agegroup=2,
            occupation=1,
            regiontype=1,
            distance=10,
            av=np.array([1]*10)
            )
                 
        #test
        self.assertTrue(
            np.allclose(
            result,
            [0.004245231161460397,
             0.03591730407748586,
             0.37137115355105155,
             0.34324035236575373,
             0.04877611038816796,
             0.04172907422256176,
             0.1287949180524477,
             0.017423477365014716,
             0.008488917395244943,
             1.3461420811290233e-05],
            atol=0.01)
            )    
                         
if __name__ == '__main__':
    unittest.main()