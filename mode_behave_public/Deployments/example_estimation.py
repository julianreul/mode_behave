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

#%% SIMPLE MODEL - 3 ATTRIBUTES

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
                data_name='example_data', 
                max_space=10000, 
                alt=4, 
                equal_alt=1
                )

                #max_space=81, 

#%% ESTIMATION OF PARAMETERS. - Takes ca. 5-10 min.

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

#calculate random points from shares.index
model.points = model.get_points(model.shares.index)

#%% FOR POST-PROCESSING: LOAD PRE-ESTIMATED PARAMETERS FOR SIMPLE MODEL
with open(model.PATH_ModelParam + "initial_point_" + str(param_random) + ".pickle", 'rb') as handle:
    model.initial_point = pickle.load(handle)  

with open(model.PATH_ModelParam + "points_" + str(param_random) + ".pickle", 'rb') as handle:
    model.points = pickle.load(handle)  

with open(model.PATH_ModelParam + "shares_" + str(param_random) + ".pickle", 'rb') as handle:
    model.shares = pickle.load(handle)  

#%%Post-processing
# Visualization of MXL-results and indication of clustering-results. 
# Play around with bw_adjust to adapt curve-smoothing.
model.visualize_space(
    k=2, 
    scale_individual=True, 
    cluster_method='kmeans', 
    external_points=np.array([model.initial_point]),
    bw_adjust=0.03,
    names_choice_options={0: "ICEV", 1: "PHEV", 2: "BEV", 3: "FCEV"}
    )
#%%
# Comment on forecasting: The forecasting is performed on training data 
# for simplicity. Please adjust the data (self.data) for further/own 
# forecasting practices by reloading a diverging subset of the base data.

# Forecast of cluster centers on the base data
model.forecast(method='LC', 
            k=2,
            cluster_method='kmeans',
            names_choice_options={0: "ICEV", 1: "PHEV", 2: "BEV", 3: "FCEV"},
            name_scenario='clustering'
            )
#%%
# Forecast of MNL model with parameter variation.
# In this example, the relative purchase price of vehicle technologies is 
# adjusted: ICEV +10%, PHEV +10%, BEV -50%, FCEV -50%
model.forecast(method='MNL', 
            sense_scenarios={"Cheap_EV": {
                "RELATIVER_KAUFPREIS": [[1.1], [1.1], [0.5], [0.5]]}
                },
            names_choice_options={0: "ICEV", 1: "PHEV", 2: "BEV", 3: "FCEV"},
            name_scenario='sensitivity'
            )