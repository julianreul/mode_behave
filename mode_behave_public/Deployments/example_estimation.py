# -*- coding: utf-8 -*-
"""
Created on Thu Jul 21 09:55:20 2022

@author: j.reul
"""

#import necessary modules
import time
from pathlib import Path
import numpy as np
import pandas as pd

import mode_behave_public as mb

#%% LOAD DATA
PATH_InputData = Path(__file__).parents[1] / 'InputData' / 'example_data.csv'
example_data = pd.read_csv(PATH_InputData, sep=";")

#%% SIMPLE MODEL - 3 ATTRIBUTES

#define model parameters
param_fixed = []
param_random = ['PURCHASE_PRICE', 'RANGE', 'CHARGING_FUELING_TIME']

param_temp = {'constant': 
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

param_temp['variable']['fixed'] = param_fixed
param_temp['variable']['random'] = param_random    
    
#Initialize model
model = mb.Core(
    param=param_temp, 
    data_in=example_data, 
    alt=4, 
    equal_alt=1
    )

#%% ESTIMATION OF PARAMETERS.

#estimate MXL model
start = time.time()
#estimate mixed logit model
model.estimate_mixed_logit(
    min_iter=10, 
    max_iter=1000,
    tol=0.01,
    space_method = 'std_value',
    scale_space = 2,
    max_shares = 1000,
    bits_64=True,
    t_stats_out=False
    )
end = time.time()
delta = int(end-start)
print('Estimation of logit model took: ', str(delta), ' seconds.')

#%%Evaluation of MNL- and MXL-model
print("LL-Ratio of MNL-model:", model.loglike_MNL()[0])
print("LL-Ratio of MXL-model:", model.loglike_MXL())

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
# This method generates Figure 2 within paper.md and is stored in the 
# following path: .\mode_behave_public\Visualizations\forecast_clustering.png

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
# This method generates Figure 1 within paper.md and is stored in the 
# following path: .\mode_behave_public\Visualizations\forecast_sensitivity.png

model.forecast(method='MNL', 
            sense_scenarios={"Cheap_EV": {
                "PURCHASE_PRICE": [[1.1], [1.1], [0.5], [0.5]]}
                },
            names_choice_options={0: "ICEV", 1: "PHEV", 2: "BEV", 3: "FCEV"},
            name_scenario='sensitivity'
            )

#%%
# Forecast of MNL model with exogeneous definition of the availability
# of certain choice options.
# In this example, the availability for choice options 0 (ICEV) and 1 (PHEV)
# are set to 0 (never available) and the availability for choice options
# 3 (BEV) and 4 (FCEV) are set to 1 (always available).

model.forecast(method='MNL', 
            av_external = [0, 0, 1, 1],
            names_choice_options={0: "ICEV", 1: "PHEV", 2: "BEV", 3: "FCEV"},
            name_scenario='external_availability',
            )