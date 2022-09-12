# -*- coding: utf-8 -*-
"""
Created on Sat May  7 11:46:35 2022

@author: j.reul
"""

import unittest
import pickle
import numpy as np

import mode_behave_public_deploy as mb

class TestEstimation(unittest.TestCase):
    
    def test_estimate_mixed_logit_gpu(self):
        """
        Integration test of the method -estimate_mixed_logit()-
        Utilized GPU hardware.

        """
        
        param_random = 'RELATIVER_KAUFPREIS'
        
        param_ranking = ['RELATIVER_KAUFPREIS', 'REICHWEITE_DURCH100', 'LADE_TANK_ZEIT']
        
        param_ranking.remove(param_random)
        
        param_cars = {'constant': 
                          {
                           'fixed':[],
                           'random':[]
                           },
                      'variable':
                          {
                           'fixed': [],
                           'random':[]
                           }
                      }
        
        param_cars['variable']['fixed'] = param_ranking
        param_cars['variable']['random'] = [param_random]    
            
        #Initialize model
        model = mb.Core(param=param_cars, 
                        data_name = 'example_data', 
                        max_space=81, 
                        alt=4,
                        equal_alt=1
                        )
        
        #estimate mixed logit model
        model.estimate_mixed_logit(
            min_iter=5, 
            max_iter=20,
            opt_search=True,
            space_method = 'std_value',
            blind_exploration = 0,
            scale_space = 1,
            SHARES_max = 1000,
            PROBS_min = 0.95,
            draws_per_iter = 500,
            updated_shares = 15,
            gpu=True,
            bits_64=False
            )

        with open(model.PATH_ModelParam + "initial_point_" + str(param_random) + ".pickle", 'rb') as handle:
            initial_point_compare = pickle.load(handle)  
                
        with open(model.PATH_ModelParam + "shares_" + str(param_random) + ".pickle", 'rb') as handle:
            shares_compare = pickle.load(handle)  
        
        model.shares = model.shares.sort_index().copy()
        shares_compare = shares_compare.sort_index().copy()
        
        #test estimation of initial_point (via the method estimate_logit())
        self.assertTrue(np.allclose(model.initial_point, initial_point_compare, atol=0.1))
        
        #test estimation of shares
        self.assertTrue(np.allclose(model.shares.values, shares_compare.values, atol=0.1))
        
        
    def test_estimate_mixed_logit_cpu(self):
        """
        Integration test of the method -estimate_mixed_logit()-
        Utilized CPU hardware only.

        """
        param_random = 'RELATIVER_KAUFPREIS'
        
        param_ranking = ['RELATIVER_KAUFPREIS', 'REICHWEITE_DURCH100', 'LADE_TANK_ZEIT']
        
        param_ranking.remove(param_random)
        
        param_cars = {'constant': 
                          {
                           'fixed':[],
                           'random':[]
                           },
                      'variable':
                          {
                           'fixed': [],
                           'random':[]
                           }
                      }
        
        param_cars['variable']['fixed'] = param_ranking
        param_cars['variable']['random'] = [param_random]    
            
        #Initialize model
        model = mb.Core(param=param_cars, 
                        data_name = 'example_data', 
                        max_space=81, 
                        alt=4, 
                        equal_alt=1
                        )
        
        #estimate mixed logit model
        model.estimate_mixed_logit(
            min_iter=5, 
            max_iter=20,
            opt_search=True,
            space_method = 'std_value',
            blind_exploration = 0,
            scale_space = 1,
            SHARES_max = 1000,
            PROBS_min = 0.95,
            draws_per_iter = 500,
            updated_shares = 15,
            gpu=False,
            bits_64=True
            )

        with open(model.PATH_ModelParam + "initial_point_" + str(param_random) + ".pickle", 'rb') as handle:
            initial_point_compare = pickle.load(handle)  
                
        with open(model.PATH_ModelParam + "shares_" + str(param_random) + ".pickle", 'rb') as handle:
            shares_compare = pickle.load(handle)  
        
        model.shares = model.shares.sort_index().copy()
        shares_compare = shares_compare.sort_index().copy()
        
        #test estimation of initial_point (via the method estimate_logit())
        self.assertTrue(np.allclose(model.initial_point, initial_point_compare, atol=0.1))
        
        #test estimation of shares
        self.assertTrue(np.allclose(model.shares.values, shares_compare.values, atol=0.1))

        
    def test_estimate_logit(self):
        """
        Integration test of the method -estimate_logit()-

        """
        param_random = 'RELATIVER_KAUFPREIS'
        
        param_ranking = ['RELATIVER_KAUFPREIS', 'REICHWEITE_DURCH100', 'LADE_TANK_ZEIT']
        
        param_ranking.remove(param_random)
        
        param_cars = {'constant': 
                          {
                           'fixed':[],
                           'random':[]
                           },
                      'variable':
                          {
                           'fixed': [],
                           'random':[]
                           }
                      }
        
        param_cars['variable']['fixed'] = param_ranking
        param_cars['variable']['random'] = [param_random]    
            
        #Initialize model
        model = mb.Core(param=param_cars, 
                        data_name = 'example_data', 
                        max_space=81, 
                        alt=4, 
                        equal_alt=1
                        )
                 
        model.initial_point = model.estimate_logit(stats=False)
        
        with open(model.PATH_ModelParam + "initial_point_" + str(param_random) + ".pickle", 'rb') as handle:
            initial_point_compare = pickle.load(handle)  
                
        #test estimation of initial_point (via the method estimate_logit())
        self.assertTrue(np.allclose(model.initial_point, initial_point_compare, atol=0.1))
               
                         
if __name__ == '__main__':
    unittest.main()