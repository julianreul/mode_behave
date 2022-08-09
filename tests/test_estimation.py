# -*- coding: utf-8 -*-
"""
Created on Sat May  7 11:46:35 2022

@author: j.reul
"""

import unittest
import pickle
import numpy as np

import mode_behave_public as mb

class TestEstimation(unittest.TestCase):
    
    def test_estimate_mixed_logit_gpu(self):
        """
        Integration test of the method -estimate_mixed_logit()-
        Utilized GPU hardware.

        """
        param_random = 'RELATIVER_KAUFPREIS'
        
        param_ranking = ['SEGMENT_Kleinwagen', 'SEGMENT_Kompaktklasse', 'SEGMENT_Mittelklasse', 'SEGMENT_Oberklasse', 
                         'SEGMENT_SUV', 'SEGMENT_Van', 'RELATIVER_KAUFPREIS',
                         'RELATIVE_B_KOSTEN_MAL100', 'REICHWEITE_DURCH100', 'LADE_TANK_ZEIT', 'DISTANZ_LADE_TANK', 'CO2_MAL10',
                         'MEAN_HH_AGE', 'HH_SIZE', 'POPULATION', 'CAR_PARK_EV', 'LONG_RANGE_AV', 
                         'OWN_POWER', 'HH_OCC', 'PT_QUALITY']
        
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
                        max_space=1e7, 
                        alt=4, 
                        )
                 
        #estimate mixed logit model
        model.estimate_mixed_logit(
            min_iter=5, 
            max_iter=20,
            opt_search=True,
            space_method = 'mirror',
            blind_exploration = 3,
            scale_space = 2,
            SHARES_max = 1000,
            PROBS_min = 0.95,
            draws_per_iter = 500,
            updated_shares = 15,
            gpu=True,
            bits_64=False
            )
        
        with open(model.PATH_ModelParam + "initial_point_" + str(param_random) + ".pickle", 'rb') as handle:
            initial_point_compare = pickle.load(handle)  
        
        with open(model.PATH_ModelParam + "points_" + str(param_random) + ".pickle", 'rb') as handle:
            points_compare = pickle.load(handle)  
        
        with open(model.PATH_ModelParam + "shares_" + str(param_random) + ".pickle", 'rb') as handle:
            shares_compare = pickle.load(handle)  
        
        #test estimation of initial_point (via the method estimate_logit())
        self.assertTrue(np.allclose(model.initial_point, initial_point_compare, atol=0.1))
        
        #test estimation of points
        self.assertTrue(np.allclose(model.points, points_compare, atol=0.1))

        #test estimation of shares
        self.assertTrue(np.allclose(model.shares, shares_compare, atol=0.1))
        
    def test_estimate_mixed_logit_cpu(self):
        """
        Integration test of the method -estimate_mixed_logit()-
        Utilized CPU hardware only.

        """
        param_random = 'RELATIVER_KAUFPREIS'
        
        param_ranking = ['SEGMENT_Kleinwagen', 'SEGMENT_Kompaktklasse', 'SEGMENT_Mittelklasse', 'SEGMENT_Oberklasse', 
                         'SEGMENT_SUV', 'SEGMENT_Van', 'RELATIVER_KAUFPREIS',
                         'RELATIVE_B_KOSTEN_MAL100', 'REICHWEITE_DURCH100', 'LADE_TANK_ZEIT', 'DISTANZ_LADE_TANK', 'CO2_MAL10',
                         'MEAN_HH_AGE', 'HH_SIZE', 'POPULATION', 'CAR_PARK_EV', 'LONG_RANGE_AV', 
                         'OWN_POWER', 'HH_OCC', 'PT_QUALITY']
        
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
                        max_space=1e7, 
                        alt=4, 
                        )
                 
        #estimate mixed logit model
        model.estimate_mixed_logit(
            min_iter=5, 
            max_iter=20,
            opt_search=True,
            space_method = 'mirror',
            blind_exploration = 3,
            scale_space = 2,
            SHARES_max = 1000,
            PROBS_min = 0.95,
            draws_per_iter = 500,
            updated_shares = 15,
            gpu=False
            )
        
        with open(model.PATH_ModelParam + "initial_point_" + str(param_random) + ".pickle", 'rb') as handle:
            initial_point_compare = pickle.load(handle)  
        
        with open(model.PATH_ModelParam + "points_" + str(param_random) + ".pickle", 'rb') as handle:
            points_compare = pickle.load(handle)  
        
        with open(model.PATH_ModelParam + "shares_" + str(param_random) + ".pickle", 'rb') as handle:
            shares_compare = pickle.load(handle)  
        
        #test estimation of initial_point (via the method estimate_logit())
        self.assertTrue(np.allclose(model.initial_point, initial_point_compare, atol=0.1))
        
        #test estimation of points
        self.assertTrue(np.allclose(model.points, points_compare, atol=0.1))

        #test estimation of shares
        self.assertTrue(np.allclose(model.shares, shares_compare, atol=0.1))
        
    def test_estimate_logit(self):
        """
        Integration test of the method -estimate_logit()-

        """
        param_random = 'RELATIVER_KAUFPREIS'
        
        param_ranking = ['SEGMENT_Kleinwagen', 'SEGMENT_Kompaktklasse', 'SEGMENT_Mittelklasse', 'SEGMENT_Oberklasse', 
                         'SEGMENT_SUV', 'SEGMENT_Van', 'RELATIVER_KAUFPREIS',
                         'RELATIVE_B_KOSTEN_MAL100', 'REICHWEITE_DURCH100', 'LADE_TANK_ZEIT', 'DISTANZ_LADE_TANK', 'CO2_MAL10',
                         'MEAN_HH_AGE', 'HH_SIZE', 'POPULATION', 'CAR_PARK_EV', 'LONG_RANGE_AV', 
                         'OWN_POWER', 'HH_OCC', 'PT_QUALITY']
        
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
                        max_space=1e7, 
                        alt=4, 
                        )
                 
        model.initial_point = model.estimate_logit(stats=False)
        
        with open(model.PATH_ModelParam + "initial_point_" + str(param_random) + ".pickle", 'rb') as handle:
            initial_point_compare = pickle.load(handle)  
                
        #test estimation of initial_point (via the method estimate_logit())
        self.assertTrue(np.allclose(model.initial_point, initial_point_compare, atol=0.1))
               
                         
if __name__ == '__main__':
    unittest.main()