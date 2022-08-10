# -*- coding: utf-8 -*-
"""
Created on Thu Jul 21 09:55:20 2022

@author: j.reul
"""

#import necessary modules
import time
import pickle
import numpy as np

import mode_behave_public as mb

#%% OPTION 1.1: COMPLEX MODEL - 20 ATTRIBUTES

#define model parameters
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

#%%OPTION 1.2: ESTIMATION OF PARAMETERS FOR COMPLEX MODEL

#estimate MXL model
start = time.time()
res = model.estimate_mixed_logit(
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
    gpu=False,
    bits_64=True
    )
end = time.time()
delta = int(end-start)
print('Estimation of logit model took: ', str(delta), ' seconds.')

#store t-statistic
t_stats = model.t_stats

#%%OPTION 1.3 FOR POST-PROCESSING: LOAD PRE-ESTIMATED PARAMETERS

with open(model.PATH_ModelParam + "initial_point_" + str(param_random) + "_complex.pickle", 'rb') as handle:
    model.initial_point = pickle.load(handle)  

with open(model.PATH_ModelParam + "points_" + str(param_random) + "_complex.pickle", 'rb') as handle:
    model.points = pickle.load(handle)  

with open(model.PATH_ModelParam + "shares_" + str(param_random) + "_complex.pickle", 'rb') as handle:
    model.shares = pickle.load(handle)  


#%% OPTION 2.1: SIMPLE MODEL - 3 ATTRIBUTES

#define model parameters
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
                )

#%%OPTION 2.2: ESTIMATION OF PARAMETERS FOR SIMPLE MODEL

#estimate MXL model
start = time.time()
res = model.estimate_mixed_logit(
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
end = time.time()
delta = int(end-start)
print('Estimation of logit model took: ', str(delta), ' seconds.')

#store t-statistic
t_stats = model.t_stats

#%%OPTION 2.3 FOR POST-PROCESSING: LOAD PRE-ESTIMATED PARAMETERS FOR SIMPLE MODEL
with open(model.PATH_ModelParam + "initial_point_" + str(param_random) + "_simple.pickle", 'rb') as handle:
    model.initial_point = pickle.load(handle)  

with open(model.PATH_ModelParam + "points_" + str(param_random) + "_simple.pickle", 'rb') as handle:
    model.points = pickle.load(handle)  

with open(model.PATH_ModelParam + "shares_" + str(param_random) + "_simple.pickle", 'rb') as handle:
    model.shares = pickle.load(handle)  

#%%Post-processing
# Visualization of MXL-results and indication of clustering-results. 
# Play around with bw_adjust to adapt curve-smoothing.
model.visualize_space(
    k=4, 
    scale_individual=True, 
    cluster_method='kmeans', 
    points_group=np.array([model.initial_point]),
    bw_adjust=0.03
    )
#%%
# Forecast of cluster centers on the base data
model.forecast(choice_values = np.array([0,1,2,3]), 
            k_cluster=4,
            cluster_method='kmeans',
            )