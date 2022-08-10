# -*- coding: utf-8 -*-
"""
Created on Mon Jan 25 14:29:51 2021

@author: j.reul

The module incorporates the class "Core", which inherits functionality from the 
classes "Estimation", "Simulation", "PostAnalysis" and "Visualitation".
It defines the core attributes, structure and type of the
discrete choice models to be simulated or estimated. 
Two different types of discrete choice models are differentiated:
    - Multinomial logit models (MNL)
    - Mixed logit models (MXL) with discrete points as parameters

"""

import numpy as np
import pandas as pd
import pickle
import os

from .estimation import Estimation
from .post_analysis import PostAnalysis
from .simulation import Simulation
from . import config

class Core(Estimation, Simulation, PostAnalysis):
    """
    The class "Core" defines the attributes, structure and type of the 
    mixed logit model, being built.
    
    The class inherits functionality from the classes "Estimation"
    and "Simulation"
    """
    
    def __init__(self, **kwargs):
        """
        Object initialization of class "Core".
             
        Parameters
        ----------
        kwargs param : dict
            Holds the names of all attributes in utility function.
            list(param) = ['constant', 'variable'] --> Disctinction between variables
            that are constant over alternatives or vary, respectively.
            list(param['constant']) = ['fixed', 'random'] --> Distinction
            between variables that are not randomly distributed within
            a group of decision-makers ('fixed') and those, that are randomly
            distributed with a discrete distribution ('random')
            
        param_transform : dict
            Same structure as -param-. Is employed to depict a different
            order of parameters. 
                        
        kwargs max_space : int
            Maximum number of data points within parameter space.
            
        kwargs alt: int
            Number of discrete choice alternatives.
            
        kwargs space : dict
            Parameter space for all coefficients (random and fixed).
            Parameter names are keys.
            
        kwargs norm_alt : int
            Defines, which alternative shall be normalized. Defaults to 0.
            
        kwargs sample_data : int
            Defines the size of the dataset, on which the model shall be
            estimated.
                        
        kwargs data_name : str
            Defines the filename of the file, which holds the base data
            to calibrate the MNL or MXL model on.
        
        kwargs data_index : array
            An array, which holds the index values of those datapoints
            of the base data, which shall be used for estimation.
        
        initial_point_name : str
            The filename of the file, which holds the estimated MNL parameters.
                    
        kwargs dc_type : str
            Determines the model type for simulation: MNL or MXL model. 
            Depends on which estimated model parameters are available.
        
        kwargs initial_point_cars_ext : list
            External definition of MNL parameters for the simulation of 
            car ownership on household level.

        kwargs initial_point_car_type_ext : list
            External definition of MNL parameters for the simulation of 
            the choice of a propulsion technology.
        
        kwargs initial_point_mode_ext : list
            External definition of MNL parameters for the simulation of 
            mode choice.
        
        kwargs dict_specific_travel_cost : dictionary
            External specification of transport costs. Relevant for the 
            simulation of mode choice.
        
        Returns
        -------
        None.

        """
        
        self.model_type = kwargs.get('model_type', 'estimation')
        
        if self.model_type == 'simulation':
            #specify the type of discrete choice model (MXL or MNL)
            self.dc_type = kwargs.get('dc_type', 'MNL')
            if self.dc_type == 'MXL':
                self.shares_cars = config.shares_cars
                self.points_cars = config.points_cars
                
            self.initial_point_cars_ext = kwargs.get('initial_point_cars_ext', [])
            self.initial_point_car_type_ext = kwargs.get('initial_point_car_type_ext', [])
            self.initial_point_mode_ext = kwargs.get('initial_point_mode_ext', [])
            self.asc_offset_hh_cars = config.asc_offset_hh_cars
            
            #load previously estimated model parameters, 
            #if model-type is simulation.
            if len(self.initial_point_cars_ext) and self.dc_type == 'MNL':
                #external parameters
                self.initial_point_cars = self.initial_point_cars_ext
            else:
                #default
                self.initial_point_cars = config.initial_point_cars
                                
            if len(self.initial_point_mode_ext) and self.dc_type == 'MNL':
                #external parameters
                self.initial_point_mode = self.initial_point_mode_ext
            else:
                #default
                self.initial_point_mode = config.initial_point_mode
                
            self.log_param = config.log_param
            dict_specific_travel_cost_ext = kwargs.get('dict_specific_travel_cost', {})
            cc_cost_ext = kwargs.get('cc_cost', False)
            if len(dict_specific_travel_cost_ext) > 0:
                self.dict_specific_travel_cost = dict_specific_travel_cost_ext
            else:
                self.dict_specific_travel_cost = config.dict_specific_travel_cost
            if cc_cost_ext:
                self.cc_cost = cc_cost_ext
            else:
                self.cc_cost = config.cc_cost_2020
            
            self.param = kwargs.get('param', {})
            if self.param:
                self.no_constant_fixed = len(self.param['constant']['fixed'])
                self.no_constant_random = len(self.param['constant']['random'])
                self.no_variable_fixed = len(self.param['variable']['fixed'])
                self.no_variable_random = len(self.param['variable']['random'])
                
        else:
            #define path to input data
            PATH_MODULE = os.path.dirname(__file__)
            sep = os.path.sep
            self.data_name = kwargs.get("data_name", False)
            self.initial_point_name = kwargs.get("initial_point_name", False)
            self.param_transform = kwargs.get("param_transform", False)
            self.PATH_InputData = PATH_MODULE + sep + 'InputData' + sep
            self.PATH_ModelParam = PATH_MODULE + sep + 'ModelParam' + sep
            
            #random or fixed within parameter space of Mixed Logit
            self.param = kwargs.get('param', False)
            self.count_c = kwargs.get('alt', False)
            self.max_space = kwargs.get('max_space', False)
            #param_ext defines logit parameters externally.
            #   type : dict
            #   e.g.: {'PARAM_NAME': [param_ext_list, alternative_list},
            #   where -alternative_list- is a list of choice alternatives for
            #   which to set the parameter -PARAM_NAME- to a given value in
            #   -param_ext_list-.
            self.param_ext = kwargs.get('param_ext', {})
            
            #check, if necessary values have been specified.
            if self.param == False:
                raise ValueError('Argument -param- needs to be specified!')                
            if self.count_c == False:
                raise ValueError('Argument -alt- needs to be specified!')                
            if self.max_space == False:
                raise ValueError('Argument -max_space- needs to be specified!')                
            if self.data_name == False:
                raise ValueError('Argument -data_name- needs to be specified!')                
            
            try:
                self.no_constant_fixed = len(self.param_transform['constant']['fixed'])
                self.no_constant_random = len(self.param_transform['constant']['random'])
                self.no_variable_fixed = len(self.param_transform['variable']['fixed'])
                self.no_variable_random = len(self.param_transform['variable']['random'])
            except:
                self.no_constant_fixed = len(self.param['constant']['fixed'])
                self.no_constant_random = len(self.param['constant']['random'])
                self.no_variable_fixed = len(self.param['variable']['fixed'])
                self.no_variable_random = len(self.param['variable']['random'])
                            
            self.space = kwargs.get("space", False)
            self.sample_data = kwargs.get("sample_data", False)
            self.data_index = kwargs.get("data_index", np.array([]))
            #select_data shall be a numpy array of tuples of (attribute, attribute_value, comparison_type)
            #comparison_type can be "equal", "below", "above"
            self.select_data = kwargs.get("select_data", np.array([]))
            
            if self.initial_point_name:
                with open(self.PATH_ModelParam + self.initial_point_name + ".pickle", 'rb') as handle:
                    self.initial_point = pickle.load(handle)  
                    
            if self.initial_point_name and self.param_transform:
                print('Transformation of initial_point.')
                self.initial_point = self.transform_initial_point(
                    self.param, 
                    self.param_transform
                    )
                self.param_init = self.param
                self.param = self.param_transform
            
            print('Data wrangling.')
            
            try:
                self.data = pd.read_csv(self.PATH_InputData + self.data_name + ".csv", sep = ",")
                if len(list(self.data)) == 1:
                    raise ValueError('Check the separator of the imported .csv-files. Should be ","-separated.')
            except:
                with open(self.PATH_InputData + self.data_name + ".pickle", 'rb') as handle:
                    self.data = pickle.load(handle)  
                    
            self.data = self.data.reset_index(drop=True)
                        
            #get data-points from indicated indices
            if self.data_index.size:
                self.data = self.data.iloc[self.data_index]
                                
            #select a subset of the data to estimate socio-economic sub-groups
            if self.select_data.size > 0:
                for i in range(self.select_data.shape[0]):
                    print('Attribute: ', str(self.select_data[i][0]))
                    print('Value: ', str(self.select_data[i][1]))
                    if self.select_data[i][2] == 'equal':
                        print(
                            'Estimate subset of data: Attribute ', 
                            str(self.select_data[i][0]), 
                            ', attribute-value == ', 
                            str(self.select_data[i][1])
                            )
                        self.data = self.data.loc[
                            self.data[
                                self.select_data[i][0]
                                ] == float(self.select_data[i][1])
                            ]
                        self.data_index = self.data.index
                    elif self.select_data[i][2] == 'below':
                        print(
                            'Estimate subset of data: Attribute ', 
                            str(self.select_data[i][0]), 
                            ', attribute-value < ', 
                            str(self.select_data[i][1])
                            )
                        self.data = self.data.loc[
                            self.data[
                                self.select_data[i][0]
                                ] < float(self.select_data[i][1])
                            ]
                        self.data_index = self.data.index
                    elif self.select_data[i][2] == 'above':
                        print(
                            'Estimate subset of data: Attribute ', 
                            str(self.select_data[i][0]), 
                            ', attribute-value > ', 
                            str(self.select_data[i][1])
                            )
                        self.data = self.data.loc[
                            self.data[
                                self.select_data[i][0]
                                ] > float(self.select_data[i][1])
                            ]
                        self.data_index = self.data.index
                    else:
                        raise ValueError('Invalid comparison type!')
                
            #shorten dataset for calculation of numerator of utility-function.
            if self.sample_data:
                if self.sample_data < len(self.data):
                    print('Length of dataset: ', str(self.sample_data))
                    self.data = self.data.sample(self.sample_data)
                    self.data_index = self.data.index
                    self.data = self.data.reset_index(drop=True)
                else:
                    print('Length of dataset: ', str(len(self.data)))
            else:
                print('Length of dataset: ', str(len(self.data)))
                
            self.choice = []
            self.av = []
            
            #define choices and availabilities
            for c in range(self.count_c):
                self.choice.append(self.data["choice_" + str(c)].values)
                self.av.append(self.data["av_" + str(c)].values)
                
            self.choice = np.array(self.choice)
            self.av = np.array(self.av)
                                                            
            #create numpy arrays from self.data and add 1 to range over a to ensure dimensionality.
            self.choice_zero = np.array(self.choice[0])        