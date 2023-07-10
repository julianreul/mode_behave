# -*- coding: utf-8 -*-
"""
Created on Fri Jan 29 16:39:30 2021

@author: j.reul

This module imports model parameters from pvrevious estimations to 
conduct simulations.
"""

import os
import numpy as np
import pandas as pd

PATH_MODULE = os.path.dirname(__file__)
sep = os.path.sep
PATH_ModelParam = PATH_MODULE + sep + "ModelParam" + sep

# Load pre-estimated parameter-files for simulation
initial_point_car_ownership = np.genfromtxt(
    PATH_ModelParam + "initial_point_car_ownership.csv", delimiter=","
)
initial_point_mode = np.genfromtxt(
    PATH_ModelParam + "initial_point_mode.csv", delimiter=","
)
asc_offset_hh_cars = pd.read_csv(
    PATH_ModelParam + "asc_offset_hh_cars.csv", index_col="Unnamed: 0"
).to_dict()

# load auxiliary data to sample distances of activities and to derive the average speed
log_param = pd.read_csv(PATH_ModelParam + "speed_parameters.csv")

dict_specific_travel_cost = {
    1: 0,
    2: 0,
    3: 0.2062,
    4: 0.1473,
    5: 0.0617,
    6: 0.0184,
    7: 0.4653,
    8: 0.0283,
    9: 0.0608,
    10: 0.2546,
}  # 2020: status quo
cc_cost_2020 = 0.33  # €/min carsharing-Kosten
