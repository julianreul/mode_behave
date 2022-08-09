# -*- coding: utf-8 -*-
"""
Created on Fri Jan 29 16:39:30 2021

@author: j.reul

This module holds the class "Simulation", which incorporates functionality
to simulate the probabilities with stored parameters from previous estimations.

The methods within this class are highly static, since they are close to 
applications within other models. Thus, good documentation is even more important.

"""

import numpy as np

from numba import njit

class Simulation:
    """
    This class incorporates methods to simulate previously estimated
    mixed logit and multinomial logit models.
    """
    
    def simulate_hh_cars(self, urban_region, rural_region, hh_size,
                         adults_working, children, htype, quali_opnv, sharing,
                         relative_cost_per_car, age_adults, regiontype, **kwargs):
        """
        This method calculates the probability of a single household
        to own 0,1,2,3 or more cars.
        Parameters
        ----------
        #   hh_size: Household-size
        #   adults_working: Number of working adults (replace by daily distance to work)
        #   children: Number of children in household
        #   urban_region: RegioStaR2 - Urban 
        #   rural_region: RegioStaR2 - Rural
        #   htype: Haustyp
        #   relative_cost_per_car: Average Price of considered cars / income
        #   sharing: Carsharing-Membership
        #   quali_opnv: Quality of public transport 
        
        param = {'constant': 
                          {
                           'fixed':[],
                           'random':[]
                           },
                      'variable':
                          {
                           'fixed':['urban_region', 'rural_region', 'hh_size', 'children', 
                                    'any_car', 'htype', 'sharing', 'age_adults', 'adults_working', 
                                    'quali_opnv'],
                           'random':[relative_cost']
                           }
                      }

        Returns
        -------
        Probabilities, that a household owns 0,1,2,3 cars.

        """
        
        #get keyword-arguments
        asc_offset_hh_cars = kwargs.get("asc_offset", self.asc_offset_hh_cars)
        asc_offset = np.array(
            [asc_offset_hh_cars['rt_' + str(regiontype)]['offset_' + str(c)] for c in range(4)]
            )
        
        #define model properties below.
        count_c = 4 #number of alternatives: 0-3
        all_alternatives = np.array((0,1,2,3))
        
        if self.param:
            no_constant_fixed = self.no_constant_fixed
            no_constant_random = self.no_constant_random
            no_variable_fixed = self.no_variable_fixed
            no_variable_random = self.no_variable_random
            #specify maximum number of alternatives
            dim_aggr_alt_max = max(
                len(self.param['constant']['fixed']),
                len(self.param['constant']['random']),
                len(self.param['variable']['fixed']),
                len(self.param['variable']['random']),
                )
        else:
            no_constant_fixed = 0
            no_constant_random = 0
            no_variable_fixed = 10
            no_variable_random = 1
            #specify maximum number of alternatives
            dim_aggr_alt_max = max(
                no_constant_fixed,
                no_constant_random,
                no_variable_fixed,
                no_variable_random,
                )
        
        #Define hh_data.
        # IMPORTANT: The order of parameters (see hh_data) must be equal to the order during 
        # estimation (see param), as defined in param = {...} !!!
        hh_data = np.zeros((4, dim_aggr_alt_max, count_c), dtype='float64')
        #fill parameters: variable_fixed
        for i in range(count_c):
            hh_data[2][0][i] = urban_region
            hh_data[2][1][i] = rural_region
            hh_data[2][2][i] = hh_size
            hh_data[2][3][i] = children
            if i == 0:
                hh_data[2][4][i] = 0
            else:
                hh_data[2][4][i] = 1
            hh_data[2][5][i] = htype
            hh_data[2][6][i] = sharing
            hh_data[2][7][i] = quali_opnv
            hh_data[2][8][i] = age_adults
            hh_data[2][9][i] = adults_working
            
        #fill parameters: variable_random
        for i in range(count_c):
            hh_data[3][0][i] = relative_cost_per_car*i
        
        @njit
        def get_utility_fast_MNL_cars(c, data, initial_point, asc_offset):
            if c != 0:
                res_temp = asc_offset[c] + initial_point[c-1]
            else:
                res_temp = asc_offset[c]
            for a in range(no_constant_fixed):
                res_temp = res_temp + initial_point[(count_c-1) + a] * data[0][a][c]
            for a in range(no_constant_random):
                res_temp = res_temp + initial_point[
                    (count_c-1) + 
                    no_constant_fixed + a
                    ] * data[1][a][c]
            for a in range(no_variable_fixed):
                res_temp = res_temp + initial_point[
                    (count_c-1) + 
                    no_constant_fixed + 
                    + (no_variable_fixed + no_variable_random)*c + a
                    ] * data[2][a][c]
            for a in range(no_variable_random):
                res_temp = res_temp + initial_point[
                    (count_c-1) + 
                    no_constant_fixed + 
                    no_constant_random + 
                    (no_variable_fixed + no_variable_random)*c + 
                    no_variable_fixed + a
                    ] * data[3][a][c]
            return res_temp   
                
        @njit
        def calculate_logit_fast_MNL_cars(alternative, data, initial_point, asc_offset):
            """
            This method calculates the multinomial logit probability for a given
            set of coefficients and all choices in the sample of the dataset.
    
            Returns
            -------
            Probability of MNL model at the initial point.
    
            """
            #calculate top
            top = np.exp(get_utility_fast_MNL_cars(alternative, data, initial_point, asc_offset))
            #calculate bottom
            bottom = 0
            for c in range(count_c):   
                bottom += np.exp(get_utility_fast_MNL_cars(c, data, initial_point, asc_offset))
            
            return top/bottom
                            
        #multinomial logit
        probs = []
        for i in all_alternatives:
            prob_temp = calculate_logit_fast_MNL_cars(
                i, 
                hh_data, 
                self.initial_point_cars, 
                asc_offset)
            probs.append(prob_temp)
        
        return probs
        
    def func(self, x, scale):
        return np.log(x + 1) * scale

    # derivation of mode specific speed: UNIT = [km/h]
    def get_speed(self, mode, distance, regiontype):
        if mode == 1:
            return 5
        elif mode in (3, 4, 5):
            scale = self.log_param.loc[
                (self.log_param["regiontype"] == regiontype)
                & (self.log_param["mode"] == mode),
                "scale",
            ].values[0]
        else:
            scale = self.log_param.loc[
                (self.log_param["regiontype"] == 0) & (self.log_param["mode"] == mode),
                "scale",
            ].values[0]

        return self.func(distance, scale)

    def get_travel_duration_single(self, mode, distance, regiontype):
        # self.check_distance = distance
        if distance == 0:
            return 0
        else:
            return distance / (self.get_speed(mode, distance, regiontype) / 60)

    def get_travel_cost(self, distance, mode, regiontype):
        if mode == 10:
            cost_temp = (
                self.get_travel_duration_single(mode, distance, regiontype)
                * self.cc_cost
            )
        else:
            cost_temp = self.dict_specific_travel_cost[mode] * distance
        
        return cost_temp

    def simulate_mode_choice(self, agegroup, occupation, regiontype, distance, av, **kwargs):
        """        
        This method calculates the probability, that a transport mode is chosen.
        Parameters
        ----------
        param = {'constant': 
                          {
                           'fixed':[],
                           'random':[]
                           },
                      'variable':
                          {
                           'fixed':['ag_1', 'ag_2', 'ag_3',
                                    'occ_1', 'occ_2', 'occ_3', 'occ_4',
                                    'urban', 'rural',
                                    'travel_cost'],
                           'random':['travel_time']
                           }
                      }

        Returns
        -------
        Probabilities, that a transport mode is chosen.

        """
                
        #define model properties below.
        count_c = 10 #number of alternatives: 0-9
        all_alternatives = np.arange(10)
        no_constant_fixed = 0
        no_constant_random = 0
        no_variable_fixed = 10
        no_variable_random = 1
        #specify maximum number of alternatives
        dim_aggr_alt_max = 10 #no_variable_fixed
        
        #Define trip_data.
        # IMPORTANT: The order of parameters must be equal to the order during 
        # estimation, as defined in param = {...} !!!
        trip_data = np.zeros((4, dim_aggr_alt_max, count_c))
        #fill parameters: variable_fixed
        if agegroup == 1:
            ag_1 = 1
            ag_2 = 0
            ag_3 = 0
        elif agegroup == 2:
            ag_1 = 0
            ag_2 = 1
            ag_3 = 0
        else:
            ag_1 = 0
            ag_2 = 0
            ag_3 = 1
            
        if occupation == 1:
            occ_1 = 1
            occ_2 = 0
            occ_3 = 0
            occ_4 = 0
        elif occupation == 2:
            occ_1 = 0
            occ_2 = 1
            occ_3 = 0
            occ_4 = 0
        elif occupation == 3:
            occ_1 = 0
            occ_2 = 0
            occ_3 = 1
            occ_4 = 0
        else:
            occ_1 = 0
            occ_2 = 0
            occ_3 = 0
            occ_4 = 1
            
        if regiontype in (1,2,3,4):
            urban = 1
            rural = 0
        else:
            urban = 0
            rural = 1
                    
        for i in range(count_c):
            trip_data[2][0][i] = ag_1
            trip_data[2][1][i] = ag_2
            trip_data[2][2][i] = ag_3
            trip_data[2][3][i] = occ_1
            trip_data[2][4][i] = occ_2
            trip_data[2][5][i] = occ_3
            trip_data[2][6][i] = occ_4
            trip_data[2][7][i] = urban
            trip_data[2][8][i] = rural
            mode = i+1
            trip_data[2][9][i] = self.get_travel_cost(distance, mode, regiontype)
        #fill parameters: variable_random
        for i in range(count_c):
            mode = i+1
            trip_data[3][0][i] = self.get_travel_duration_single(mode, distance, regiontype)
        
        @njit
        def get_utility_fast_MNL_mode(c, data, initial_point):
            if c == 0:
                res_temp = initial_point[c-1]
            else:
                res_temp = 0
            for a in range(no_constant_fixed):
                res_temp = res_temp + initial_point[(count_c-1) + a] * data[0][a][c]
            for a in range(no_constant_random):
                res_temp = res_temp + initial_point[
                    (count_c-1) + 
                    no_constant_fixed + 
                    a
                    ] * data[1][a][c]
            for a in range(no_variable_fixed):
                res_temp = res_temp + initial_point[
                    (count_c-1) + 
                    no_constant_fixed + 
                    no_constant_random + 
                    (no_variable_fixed + no_variable_random)*c + a
                    ] * data[2][a][c]
            for a in range(no_variable_random):
                res_temp = res_temp + initial_point[
                    (count_c-1) + 
                    no_constant_fixed + 
                    no_constant_random + 
                    (no_variable_fixed + no_variable_random)*c + 
                    no_variable_fixed + a
                    ] * data[3][a][c]
            return res_temp   
        
        
        @njit
        def calculate_logit_fast_MNL_mode(alternative, data, initial_point, av):
            """
            This method calculates the multinomial logit probability for a given
            set of coefficients and all choices in the sample of the dataset.
    
            Parameters
            ----------
            A "point" with all coefficients of MLN-attributes.
    
            Returns
            -------
            Probability of MNL model at a specified point.
    
            """
            #calculate top
            top = av[alternative] * np.exp(get_utility_fast_MNL_mode(
                alternative, 
                data, 
                initial_point))
            
            #calculate bottom
            bottom = 0
            for c in range(count_c):   
                bottom += av[c] * np.exp(get_utility_fast_MNL_mode(c, data, initial_point))
                
            return top/bottom
        
        probs = []
        for i in all_alternatives:
            prob_temp = calculate_logit_fast_MNL_mode(i, trip_data, self.initial_point_mode, av)
            probs.append(prob_temp)
        
        return probs