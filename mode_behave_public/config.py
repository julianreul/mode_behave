# -*- coding: utf-8 -*-
"""
Created on Fri Jan 29 16:39:30 2021

@author: j.reul

This module imports model parameters from pvrevious estimations to 
conduct simulations.
"""

import os
import pickle

PATH_MODULE = os.path.dirname(__file__)
sep = os.path.sep
PATH_ModelParam = PATH_MODULE + sep + 'ModelParam' + sep

# Load parameter-files from the estimation
with open(PATH_ModelParam + "initial_point_" + "car_ownership.pickle", "rb") as handle:
    initial_point_cars = pickle.load(handle)

with open(PATH_ModelParam + "initial_point_mode.pickle", "rb") as handle:
    initial_point_mode = pickle.load(handle)
    
with open(PATH_ModelParam + "asc_offset_hh_cars.pickle", 'rb') as handle:
    asc_offset_hh_cars = pickle.load(handle)  
    
# load auxiliary data to sample distances of activities and to derive the average speed
with open(PATH_ModelParam + "speed_parameters.pickle", "rb") as handle:
    log_param = pickle.load(handle)
    
dict_specific_travel_cost = {1: 0,
    2: 0,
    3: 0.2062,
    4: 0.1473,
    5: 0.0617,
    6: 0.0184,
    7: 0.4653,
    8: 0.0283,
    9: 0.0608,
    10: 0.2546
    }  # 2020: status quo
cc_cost_2020 = 0.33  # €/min carsharing-Kosten