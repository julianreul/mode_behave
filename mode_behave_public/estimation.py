# -*- coding: utf-8 -*-
"""
Created on Mon Jan 25 14:30:06 2021

@author: j.reul

This module holds the class "Estimation", which incorporates functionality
to estimate the coefficients and class-shares of the specified mixed logit
model with a discrete mixing distribution with fixed points as well as 
the coefficients of a multinomial logit model.
"""

import time
import random
from operator import mod
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import pandas as pd
import numpy as np
#import GPUtil
from numba import guvectorize, vectorize, njit, cuda, prange
from scipy.optimize import minimize
from scipy import stats
from math import exp, log, sqrt, pow, floor


class Estimation:
    """
    This class holds all functionality to estimate a mixed logit model 
    with a discrete mixing distribution with fixed points and well as to 
    estimate a multinomial logit model.
    c is short for "choice option", indicating the choice alternative.
    a is short for "attribute", indicating the observed choice attribute.
    
    Methods
    -------
    - estimate_logit: Estimates the coefficients of a multinomial logit model.
    - estimate_mixed_logit: Estimates the shares of all classes for a 
        mixed logit model with an EM algorithm.
    - get_param_space: Derives the parameter space from an initial 
        logit estimation.
            
    """
    
    def __init__(self):
        pass
       
    def estimate_mixed_logit(self, **kwargs):
        """
        This method estimates the mixed logit model for a given set of 
        model attributes. Therefore it first estimates a multinomial logit model
        and, building on that, it estimates the parameters of the mixed logit
        model by iteratively exploring a parameter space around the initial
        parameters of the multinomial logit model.

        Parameters
        ----------
        kwargs tol : float
            Tolerance of the internal EM-algorithm.
        kwargs max_iter : int
            Minimum iterations of the EM-algorithm.
        kwargs min_iter : int
            Maximum iterations of the EM-algorithm.
        kwargs gpu : boolean
            If True, gpu-hardware is utilized for estimation.     
        kwargs space_method : string
            The method which shall be applied to span the parameter space
            around the initially estimated parameter points (from MNL-model).
            Options are "abs_value", "std_value" or "mirror". Defaults to "mirror".
        kwargs scale_space : float
            Sets the size of the parameter space. Defaults to 2.
        kwargs blind_exploration : int
            Sets the number of explorations of the parameter space, which are 
            not anchored in previously estimated parameter shares. A higher 
            values decreases risk of finding local extrema compared to global
            extrema but also adds computation time.
        kwargs bits_64 : Boolean
            If True, numerical precision is increased to 64 bits, instead of 32 bits.
            Defaults to False.
        kwargs PROBS_min : Float
            Indicates the share of the parameter space, which shall be explored.
            Defaults to 0.9.
        kwargs SHARES_max : int
            Specifies the maximum number of points in the parameter space, for which
            a "share" shall be estimated. That does not mean, that only this number
            of points will be explored in the parameter space, but only for this 
            number points a "share" is being stored. This is done to limit the 
            memory of the estimation process. SHARES_max defaults to 1000.
        kwargs draws_per_iter : int
            The exploration of the parameter space is an iterative process.
            "draws_per_iteration" indicates the number of points of the parameter
            space, which are explored within the same iteration step.
            Defaults to 1000.
        kwargs updated_shares : int
            This value specifies the number of points in proximity of 
            actually explored points in the parameter space, which shall 
            be updated according to the assumption of a normal distribution, 
            which flattens out from the actually estimated points as centers.
            A higher value speeds up the estimation process but decreases
            the precision of the estimation, as the actual distribution of 
            shares might not follow a normal distribution in the proximity
            of the actually observed shares. Defaults to 0.           

        Returns
        -------
        self.shares : array
            The central output of this method is the array "self.shares", which
            holds the estimated shares of points within the parameter space.

        """

        #get estimation parameters.
        tol = kwargs.get("tol", 0.01)
        max_iter = kwargs.get("max_iter", 100)
        min_iter = kwargs.get("min_iter", 10)
        gpu = kwargs.get('gpu', False)
        scale_space = kwargs.get('scale_space', 2)
        space_method = kwargs.get('space_method', 'mirror')
        blind_exploration = kwargs.get('blind_exploration', 0)
        t_stats_out = kwargs.get('t_stats_out', True)
        self.bits_64 = kwargs.get('bits_64', False)
        print('_______________________')
        print('Cluster-GPU can only handle float32, NOT float64!')
        print('_______________________')
                    
        #Explore at least X % of the parameter space
        PROBS_min = kwargs.get('PROBS_min', 0.9)
        #treshold for dropping a point: percentage of initial value in 'SHARES'
        SHARES_max = kwargs.get('SHARES_max', 1000)
        #number of draws per iteration
        draws_per_iter = kwargs.get('draws_per_iter', 1000)
        #number of updated, non-drawn shares. Higher values speeds up exploration
        #of parameter space, but may decrease accuracy of calculation.
        updated_shares = kwargs.get('updated_shares', 0)
           
        #reshuffle parameters to fit new definition of param
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
                            
        if self.param_transform:
            print('Transformation of initial_point.')
            self.initial_point = self.transform_initial_point(self.param, self.param_transform)
            self.param = self.param_transform            
        
        #get the maximum number of equally-spaces coefficients per alternative.
        #points per coefficient (ppc)
        no_random = self.no_constant_random + self.no_variable_random*self.count_c
        ppc = int(self.max_space**(1/(no_random)))
        print('Number of estimated points per class: ', str(ppc))
        
        #Define space-boundaries from initial point.
        if space_method == 'abs_value':
            try:
                #define absolute value of parameter as offset
                offset_values = np.array([abs(temp) for temp in self.initial_point])
            except:
                print('Estimate initial coefficients.')
                if t_stats_out:
                    self.initial_point = self.estimate_logit()
                else:
                    self.initial_point = self.estimate_logit(stats=False)
                #define absolute value of parameter as offset
                offset_values = np.array([abs(temp) for temp in self.initial_point])

        elif space_method == 'std_value':
            try:
                #define std of parameter as offset
                offset_values = self.std_cross_val
            except:
                print('Estimate initial coefficients.')
                self.initial_point = self.estimate_logit()
                #define std of parameter as offset
                offset_values = self.std_cross_val
                
        elif space_method == 'mirror':
            try:
                #define std of parameter as offset
                offset_values = self.std_cross_val
            except:
                print('Estimate initial coefficients.')
                
                self.initial_point = self.estimate_logit()
                #define std of parameter as offset
                offset_values = self.std_cross_val
        
        else:
            raise ValueError('Unknown value for keyword-argument -space_method-')
        
        #specify parameter space
        if self.bits_64:
            self.space = np.zeros((no_random, ppc), 'float64')
        else:
            self.space = np.zeros((no_random, ppc), 'float32')
        self.size_space = ppc**no_random
        for a in range(self.no_constant_random):
            mean_coefficient = self.initial_point[self.count_c-1 + self.no_constant_fixed + a]
            offset = offset_values[self.count_c-1 + self.no_constant_fixed + a]
            if space_method == 'mirror':
                if mean_coefficient > 0:
                    upper_bound = mean_coefficient + offset*scale_space
                    lower_bound = min(-mean_coefficient, mean_coefficient - offset*scale_space)
                else:
                    upper_bound = max(-mean_coefficient, mean_coefficient + offset*scale_space)
                    lower_bound = mean_coefficient - offset*scale_space
            else:
                lower_bound = mean_coefficient - offset*scale_space
                upper_bound = mean_coefficient + offset*scale_space
            spread = upper_bound-lower_bound
            range_coefficients = np.linspace(lower_bound, upper_bound, ppc)
            for i in range(ppc):
                self.space[a][i] = range_coefficients[i]
        for c in range(self.count_c):
            for a in range(self.no_variable_random):
                mean_coefficient = self.initial_point[
                    self.count_c-1 + 
                    self.no_constant_fixed + 
                    self.no_constant_random + 
                    (self.no_variable_fixed + self.no_variable_random)*c +
                    self.no_variable_fixed +
                    a
                    ]
                offset = offset_values[
                    self.count_c-1 + 
                    self.no_constant_fixed + 
                    self.no_constant_random + 
                    (self.no_variable_fixed + self.no_variable_random)*c +
                    self.no_variable_fixed +
                    a
                    ]
                
                if space_method == 'mirror':
                    if mean_coefficient > 0:
                        upper_bound = mean_coefficient + offset*scale_space
                        lower_bound = min(-mean_coefficient, mean_coefficient - offset*scale_space)
                    else:
                        upper_bound = max(-mean_coefficient, mean_coefficient + offset*scale_space)
                        lower_bound = mean_coefficient - offset*scale_space
                else:
                    lower_bound = mean_coefficient - offset*scale_space
                    upper_bound = mean_coefficient + offset*scale_space
                    
                spread = upper_bound-lower_bound
                if spread == 0:
                    range_coefficients = [mean_coefficient]*ppc
                else:
                    range_coefficients = np.linspace(lower_bound, upper_bound, ppc)
                    
                if len(range_coefficients) != ppc:
                    raise ValueError("Check specification of ppc")
                    
                for i in range(ppc):
                    self.space[
                        self.no_constant_random + self.no_variable_random*c + a
                        ][i] = range_coefficients[i]
        
        #prepare input for numba
        initial_point = self.initial_point
        count_c = self.count_c
        count_e = self.count_e
        no_constant_fixed = self.no_constant_fixed
        no_constant_random = self.no_constant_random
        no_variable_fixed = self.no_variable_fixed
        no_variable_random = self.no_variable_random
        av = self.av
        choice = self.choice
        
        #maximum number of aggregated alternatives per segment
        dim_aggr_alt_max = max(
            len(self.param['constant']['fixed']),
            len(self.param['constant']['random']),
            len(self.param['variable']['fixed']),
            len(self.param['variable']['random']),
            )
                
        data = np.zeros((4,dim_aggr_alt_max,self.count_c,self.av.shape[1],len(self.data)))
        for c in range(self.count_c):
            for e in range(self.count_e):
                for i, param in enumerate(self.param['constant']['fixed']):
                    data[0][i][c][e] = self.data[param + '_' + str(c) + '_' + str(e)].values
                for i, param in enumerate(self.param['constant']['random']):
                    data[1][i][c][e] = self.data[param + '_' + str(c) + '_' + str(e)].values
                for i, param in enumerate(self.param['variable']['fixed']):
                    data[2][i][c][e] = self.data[param + '_' + str(c) + '_' + str(e)].values
                for i, param in enumerate(self.param['variable']['random']):
                    data[3][i][c][e] = self.data[param + '_' + str(c) + '_' + str(e)].values
                
        if self.bits_64 == False:
            data = data.astype('float32')
            av = av.astype('int32')
            choice = choice.astype('int32')
            
        if gpu:
            @cuda.jit(device=True)
            def get_utility_gpu(c, e, point_, l, data):
                """
                This method calculates the utility of a choice option,
                utilizing GPU-hardware. Therefore, it must be defined internally.

                Parameters
                ----------
                c : int
                    Choice option.
                e : int
                    Number of equal choice options.
                point_ : array
                    Point in the parameter space.
                l : int
                    DESCRIPTION.
                data : multi-dimensional array
                    Base data.

                Returns
                -------
                res_temp : array
                    Utility array.

                """
                if c == 0:
                    res_temp = 0
                else:
                    res_temp = initial_point[c-1]
                for a in range(no_constant_fixed):
                    res_temp += initial_point[(count_c-1) + a] * data[0][a][c][e][l]
                for a in range(no_constant_random):
                    res_temp += point_[a] * data[1][a][c][e][l]
                for a in range(no_variable_fixed):
                    res_temp += initial_point[
                        (count_c-1) + 
                        no_constant_fixed + 
                        no_constant_random + 
                        (no_variable_fixed + no_variable_random)*c + a
                        ] * data[2][a][c][e][l]
                for a in range(no_variable_random):
                    res_temp += point_[
                        no_constant_random + no_variable_random*c + a
                        ] * data[3][a][c][e][l]
                return res_temp   
            
            if self.bits_64:

                @guvectorize(
                    ['float64[:], int64[:, :, :], int64[:, :, :], float64[:, :, :, :, :], float64[:]'], 
                    '(p),(n,e,l),(n,e,l),(i,j,n,e,l)->(l)', 
                    target='cuda'
                    )
                def calculate_logit_gpu(point, av, choice, data, logit_probs_):
                    """
                    This method calculates the multinomial logit probability for a given
                    set of coefficients and all choices in the sample of the dataset.
                
                    Parameters
                    ----------
                    point : array
                        Point in the parameter space.
                    av : array
                        Availability array for all choice options.
                    choice : array
                        Array, indicating the choice of the actor in the base data.
                    data : array
                        Base data.
                
                    Returns
                    -------
                    Array with logit-probabilities.
                
                    """    
                                            
                    #iterate over length of data array (len(av))
                    #What is the shape of the output array?!
                    for l in range(av.shape[2]):
                        #calculate bottom
                        bottom = 0
                        top = 0
                        for c in range(count_c):
                            for e in range(count_e):
                                bottom += av[c][e][l] * exp(get_utility_gpu(c, e, point, l, data))                
                                top += av[c][e][l] * choice[c][e][l] * exp(
                                    get_utility_gpu(c, e, point, l, data)
                                    )
                        if bottom>0:
                            logit_probs_[l] = top/bottom      
                        else:
                            logit_probs_[l] = 0     
                                
                        
            else:
                @guvectorize(
                    ['float32[:], int32[:, :, :], int32[:, :, :], float32[:, :, :, :, :], float32[:]'], 
                    '(p),(n,e,l),(n,e,l),(i,j,n,e,l)->(l)', 
                    target='cuda'
                    )
                def calculate_logit_gpu(point, av, choice, data, logit_probs_):
                    """
                    This method calculates the multinomial logit probability for a given
                    set of coefficients and all choices in the sample of the dataset.
                
                    Parameters
                    ----------
                    point : array
                        Point in the parameter space.
                    av : array
                        Availability array for all choice options.
                    choice : array
                        Array, indicating the choice of the actor in the base data.
                    data : array
                        Base data.
                
                    Returns
                    -------
                    Array with logit-probabilities.
                
                    """    
                    #iterate over length of data array (len(av))
                    #What is the shape of the output array?!
                    for l in range(av.shape[2]):
                        #calculate bottom
                        bottom = 0
                        top = 0
                        for c in range(count_c):
                            for e in range(count_e):
                                bottom += av[c][e][l] * exp(get_utility_gpu(c, e, point, l, data,))                
                                top += av[c][e][l] * choice[c][e][l] * exp(
                                    get_utility_gpu(c, e, point, l, data)
                                    )
                        if bottom>0:
                            logit_probs_[l] = top/bottom      
                        else:
                            logit_probs_[l] = 0     
                    
        else:
            @njit
            def get_utility_vector(c, e, point, l, data):
                """
                Calculates the utility of a choice option.

                Parameters
                ----------
                c : int
                    Choice option.
                point : array
                    Multi-dimensional point in the parameter space.
                l : array
                    DESCRIPTION.
                data : array
                    Base data.

                Returns
                -------
                res_temp : float
                    Utility of a choice option.

                """               
                if c == 0:
                    res_temp = 0
                else:
                    res_temp = initial_point[c-1]
                
                for a in range(no_constant_fixed):
                    res_temp += initial_point[(count_c-1) + a] * data[0][a][c][e][l]
                for a in range(no_constant_random):
                    res_temp += point[a] * data[1][a][c][e][l]
                for a in range(no_variable_fixed):
                    res_temp += initial_point[
                        (count_c-1) + 
                        no_constant_fixed + 
                        no_constant_random + 
                        (no_variable_fixed + no_variable_random)*c + a
                        ] * data[2][a][c][e][l]
                for a in range(no_variable_random):
                    res_temp += point[no_constant_random + no_variable_random*c + a] * data[3][a][c][e][l]
                                
                return res_temp   
            
            if self.bits_64:
                @guvectorize(
                    ['float64[:, :], int64[:, :, :], float64[:, :, :, :, :], float64[:, :]'], 
                    '(m,p),(n,e,l),(i,j,n,e,l)->(m,l)', 
                    nopython=True, 
                    target="parallel"
                    )
                def calculate_logit_vector(points, av, data, logit_probs_):
                    """
                    This method calculates the multinomial logit probability for a given
                    set of coefficients and all choices in the sample of the dataset.
                
                    Parameters
                    ----------
                    point : array
                        Point in the parameter space.
                    av : array
                        Availability array for all choice options.
                    data : array
                        Base data.
                
                    Returns
                    -------
                    Array with logit-probabilities.
                
                    """    
                    
                    for m in prange(points.shape[0]):  
                        point = points[m]
                        
                        #iterate over length of data array (len(av))
                        #What is the shape of the output array?!
                        for l in prange(av.shape[2]):
                            #calculate bottom
                            bottom = 0
                            top = 0
                            for c in range(count_c):
                                for e in range(count_e):
                                    exp_temp = exp(get_utility_vector(c, e, point, l, data))   
                                    bottom += av[c][e][l] * exp_temp        
                                    top += av[c][e][l] * choice[c][e][l] * exp_temp
                                    res_temp = top/bottom
                            if np.isfinite(res_temp):
                                logit_probs_[m][l] = res_temp  
                            else:
                                logit_probs_[m][l] = 0
                                
            else:
                @guvectorize(
                    ['float32[:, :], int32[:, :, :], float32[:, :, :, :, :], float32[:, :]'], 
                    '(m,p),(n,e,l),(i,j,n,e,l)->(m,l)', 
                    nopython=True, 
                    target="parallel"
                    )
                def calculate_logit_vector(points, av, data, logit_probs_):
                    """
                    This method calculates the multinomial logit probability for a given
                    set of coefficients and all choices in the sample of the dataset.
                
                    Parameters
                    ----------
                    point : array
                        Point in the parameter space.
                    av : array
                        Availability array for all choice options.
                    data : array
                        Base data.
                
                    Returns
                    -------
                    Array with logit-probabilities.
                
                    """    
                    
                    for m in prange(points.shape[0]):  
                        point = points[m]
                        
                        #iterate over length of data array (len(av))
                        #What is the shape of the output array?!
                        for l in prange(av.shape[2]):
                            #calculate bottom
                            bottom = 0
                            top = 0
                            for c in range(count_c):
                                for e in range(count_e):
                                    exp_temp = exp(get_utility_vector(c, e, point, l, data))
                                    bottom += av[c][e][l] * exp_temp               
                                    top += av[c][e][l] * choice[c][e][l] * exp_temp
                                    res_temp = top/bottom
                            if np.isfinite(res_temp):
                                logit_probs_[m][l] = res_temp
                            else:
                                logit_probs_[m][l] = 0
                                        
        if gpu:
            if self.bits_64:
                @guvectorize(['float64[:], float64[:], float64[:]'], '(),(l)->(l)', target='cuda')
                def get_pp_top_gpu(shares, logit_probs, pp_top): 
                    """
                    This methods calculates the weighted probability for
                    each point, utilizing GPU-hardware.

                    Parameters
                    ----------
                    shares : array
                        The share (weight) of a point within the parameter space.
                    logit_probs : array
                        The logit probability of a point..
                    pp_top : array
                        The weighted probability of a point.

                    Returns
                    -------
                    None.

                    """
                    #initialize points_prob_top
                    for l in range(logit_probs.shape[0]):
                        pp_top[l] = shares[0]*logit_probs[l]    
                                  
                @njit#(fastmath=True)
                def calc_expect(share_single, pp_single):
                    """
                    This method calculates the EM-expectation for a SINGLE 
                    point, derived from the weighted probability (pp), 
                    utilizing GPU-hardware.

                    Parameters
                    ----------
                    share_single : float
                        The share (weight) of a point in the parameter space.
                    pp_single : float
                        The logit probability of a point in the parameter space.

                    Returns
                    -------
                    float
                        EM-expectation of a point.

                    """
                    if share_single == 0:
                        return 0
                    else:
                        return log(share_single) * pp_single
                        
                @guvectorize(['float64[:], float64[:], float64[:]'], '(),(l)->()', target='cuda')
                def get_expectation_gpu(shares_update, pp, expect): 
                    """
                    This method calculates the EM-expectation for EACH point,
                    utilizing GPU-hardware.

                    Parameters
                    ----------
                    shares_update : array
                        Shares or weights of each point..
                    pp : array
                        Weighted probabilities.

                    Returns
                    -------
                    Array
                        EM-expectations for each point.

                    """
                    #calculate expectation
                    expect_tmp = 0
                    for l in range(pp.shape[0]):
                        share_single = shares_update[0]
                        pp_single = pp[l]
                        expect_tmp += calc_expect(share_single, pp_single)
                            
                    expect[0] = expect_tmp   
                    
            else:
                @guvectorize(['float32[:], float32[:], float32[:]'], '(),(l)->(l)', target='cuda')
                def get_pp_top_gpu(shares, logit_probs, pp_top):
                    """
                    This methods calculates the weighted probability for
                    each point, utilizing GPU-hardware.

                    Parameters
                    ----------
                    shares : array
                        The share (weight) of a point within the parameter space.
                    logit_probs : array
                        The logit probability of a point..
                    pp_top : array
                        The weighted probability of a point.

                    Returns
                    -------
                    None.

                    """
                    #initialize points_prob_top
                    for l in range(logit_probs.shape[0]):
                        pp_top[l] = shares[0]*logit_probs[l]    
                                  
                @njit#(fastmath=True)
                def calc_expect(share_single, pp_single):
                    """
                    This method calculates the EM-expectation for a SINGLE 
                    point, derived from the weighted probability (pp), 
                    utilizing GPU-hardware.

                    Parameters
                    ----------
                    share_single : float
                        The share (weight) of a point in the parameter space.
                    pp_single : float
                        The logit probability of a point in the parameter space.

                    Returns
                    -------
                    float
                        EM-expectation of a point.

                    """
                    if share_single == 0:
                        return 0
                    else:
                        return log(share_single) * pp_single
                        
                @guvectorize(['float32[:], float32[:], float32[:]'], '(),(l)->()', target='cuda')
                def get_expectation_gpu(shares_update, pp, expect): 
                    """
                    This method calculates the EM-expectation for EACH point,
                    utilizing GPU-hardware.

                    Parameters
                    ----------
                    shares_update : array
                        Shares or weights of each point..
                    pp : array
                        Weighted probabilities.

                    Returns
                    -------
                    Array
                        EM-expectations for each point.

                    """
                    #calculate expectation
                    expect_tmp = 0
                    for l in range(pp.shape[0]):
                        share_single = shares_update[0]
                        pp_single = pp[l]
                        expect_tmp += calc_expect(share_single, pp_single)
                            
                    expect[0] = expect_tmp   
                                
        else:
            if self.bits_64:
                @guvectorize(
                    ['float64[:], float64[:, :], float64[:], float64[:]'], 
                    '(m),(m,l)->(),(m)', 
                    nopython=True, 
                    target="parallel"
                    )
                def get_expectation(shares, logit_probs, expect, shares_update): 
                    """
                    This method calculates the EM-expectation for EACH point.

                    Parameters
                    ----------
                    shares : array
                        Shares or weights of each point..
                    logit_probs : array
                        Logit probabilities.

                    Returns
                    -------
                    Array
                        EM-expectations for each point.

                    """
                    #get point_probs_top
                    sum_point_probs_top = 0
                    #initialize points_prob_top
                    point_probs_top = logit_probs
                    for m in prange(logit_probs.shape[0]):
                        for l in prange(logit_probs.shape[1]):
                            temp = shares[m]*logit_probs[m][l]
                            point_probs_top[m][l] = temp
                            sum_point_probs_top += temp
                        
                    #get point_probs    
                    sum_point_probs = 0
                    #initialize point_probs
                    point_probs = point_probs_top
                    #initialize sum along axis of point_probs
                    sum_point_probs_axis = shares
                    for m in prange(logit_probs.shape[0]):
                        sum_point_probs_axis_single = 0
                        for l in prange(logit_probs.shape[1]):
                            point_probs_temp = point_probs_top[m][l] / sum_point_probs_top
                            point_probs[m][l] = point_probs_temp
                            sum_point_probs += point_probs_temp
                            sum_point_probs_axis_single += point_probs_temp
                        sum_point_probs_axis[m] = sum_point_probs_axis_single
                        
                    #update shares
                    #initialize shares_update
                    for m in prange(logit_probs.shape[0]):
                        shares_update[m] = sum_point_probs_axis[m] / sum_point_probs
                        
                    #calculate expectation
                    expect_tmp = 0
                    for m in prange(logit_probs.shape[0]):
                        if shares_update[m] == 0:
                            continue
                        else:
                            for l in prange(logit_probs.shape[1]):
                                expect_tmp += log(shares_update[m]) * point_probs[m][l]
                    expect[0] = expect_tmp 
            else:
                @guvectorize(
                    ['float32[:], float32[:, :], float32[:], float32[:]'], 
                    '(m),(m,l)->(),(m)', 
                    nopython=True, 
                    target="parallel"
                    )
                def get_expectation(shares, logit_probs, expect, shares_update):
                    """
                    This method calculates the EM-expectation for EACH point.

                    Parameters
                    ----------
                    shares : array
                        Shares or weights of each point..
                    logit_probs : array
                        Logit probabilities.

                    Returns
                    -------
                    Array
                        EM-expectations for each point.

                    """
                    #get point_probs_top
                    sum_point_probs_top = 0
                    #initialize points_prob_top
                    point_probs_top = logit_probs
                    for m in prange(logit_probs.shape[0]):
                        for l in prange(logit_probs.shape[1]):
                            temp = shares[m]*logit_probs[m][l]
                            point_probs_top[m][l] = temp
                            sum_point_probs_top += temp
                        
                    #get point_probs    
                    sum_point_probs = 0
                    #initialize point_probs
                    point_probs = point_probs_top
                    #initialize sum along axis of point_probs
                    sum_point_probs_axis = shares
                    for m in prange(logit_probs.shape[0]):
                        sum_point_probs_axis_single = 0
                        for l in prange(logit_probs.shape[1]):
                            point_probs_temp = point_probs_top[m][l] / sum_point_probs_top
                            point_probs[m][l] = point_probs_temp
                            sum_point_probs += point_probs_temp
                            sum_point_probs_axis_single += point_probs_temp
                        sum_point_probs_axis[m] = sum_point_probs_axis_single
                        
                    #update shares
                    #initialize shares_update
                    for m in prange(logit_probs.shape[0]):
                        shares_update[m] = sum_point_probs_axis[m] / sum_point_probs
                        
                    #calculate expectation
                    expect_tmp = 0
                    for m in prange(logit_probs.shape[0]):
                        if shares_update[m] == 0:
                            continue
                        else:
                            for l in prange(logit_probs.shape[1]):
                                expect_tmp += log(shares_update[m]) * point_probs[m][l]
                    expect[0] = expect_tmp 
        
        print('Estimate shares.')      
        
        if self.bits_64:
            @guvectorize(
                ['float64[:, :], int32[:], float64[:, :]'], 
                '(r,m),(n)->(n,r)', 
                nopython=True, 
                target="parallel"
                )
            def get_points_from_draws_vector(space, draws, drawn_points): 
                """
                This methods defines the points in the parameter space,
                indicated by the indices in the vector "draws".
                This method in necessary, since the points are "created" upon
                request, as storing all points of the parameter space
                would lead to excessive memory usage.

                Parameters
                ----------
                space : array
                    2D-Matrix of all potential values, that constitute the
                    parameter space.
                draws : array
                    Index values of the points that shall be drawn from space.

                Returns
                -------
                drawn_points : array
                    Array with drawn points from the parameter space.
                """
                no_random = space.shape[0]
                ppc = space.shape[1]
                #convention: Count indices, starting from last column.
                for d in prange(len(draws)):
                    rest = draws[d]
                    for i in prange(no_random):
                        exp_temp = no_random-(i+1)
                        index_temp = floor(rest/pow(ppc, exp_temp))
                        drawn_points[d][i] = space[i][index_temp]
                        rest = mod(rest, pow(ppc, exp_temp))
        else:
            @guvectorize(
                ['float32[:, :], int32[:], float32[:, :]'], 
                '(r,m),(n)->(n,r)', 
                nopython=True, 
                target="parallel"
                )
            def get_points_from_draws_vector(space, draws, drawn_points): 
                """
                This methods defines the points in the parameter space,
                indicated by the indices in the vector "draws".
                This method in necessary, since the points are "created" upon
                request, as storing all points of the parameter space
                would lead to excessive memory usage.

                Parameters
                ----------
                space : array
                    2D-Matrix of all potential values, that constitute the
                    parameter space.
                draws : array
                    Index values of the points that shall be drawn from space.

                Returns
                -------
                drawn_points : array
                    Array with drawn points from the parameter space.

                """
                no_random = space.shape[0]
                ppc = space.shape[1]
                #convention: Count indices, starting from last column.
                for d in prange(len(draws)):
                    rest = draws[d]
                    for i in prange(no_random):
                        exp_temp = no_random-(i+1)
                        index_temp = floor(rest/pow(ppc, exp_temp))
                        drawn_points[d][i] = space[i][index_temp]
                        rest = mod(rest, pow(ppc, exp_temp))
                                    
        def get_draws(draws_per_iter, PROBS_values, PROBS_index, PROB_UNKNOWN):
            """
            This methods gets the index values for the points that shall be
            drawn from space.

            Parameters
            ----------
            draws_per_iter : int
                Number of points that shall be drawn (per iteration).
            PROBS_values : array
                Array with the choice prabability of points in PROBS_index.
            PROBS_index : array
                Array with index-values of already explored points.
            PROB_UNKNOWN : float
                Ratio of the unkown space in relation to overall space.

            Returns
            -------
            draws_unkown : array
                Draws from unkown space.
            draws_explored : array
                Draws from known space.

            """
            #get number of draws from explored and unknown space
            no_unknown = floor(draws_per_iter*PROB_UNKNOWN)
            no_explored = draws_per_iter-no_unknown
            #get draws from already explored space
            if no_explored > 0:
                draws_explored = np.random.choice(
                    PROBS_index, size=no_explored, p=PROBS_values, replace=False
                    )
            else:
                draws_explored = np.array([])
            
            #get draws from unknown space
            if PROBS_index.size > 0:
                self.start_c = True
                draws_unknown = np.array([], dtype='int32')
                delete_temp = np.append(PROBS_index, EXCLUDED)
                start_c_1 = time.time()
                #select "no_unknown" values from self.size_space, excluding "delete_temp"
                
                prob_random = np.ones(self.size_space)
                prob_random[delete_temp] = 0
                prob_random_scaled = prob_random / np.sum(prob_random)
                   
                draws_unknown = np.random.choice(
                    np.arange(self.size_space),
                    size=no_unknown,
                    p=prob_random_scaled
                    )
                    
                end_c_1 = time.time()
                self.delta_c_1 = end_c_1 - start_c_1
            else:     
                self.start_c = False
                draws_unknown = np.array([random.randrange(0,self.size_space) for _ in range(no_unknown)])
                            
            return draws_unknown.astype('int32'), draws_explored.astype('int32')
                
        #OPT SEARCH
        #SG: 'Saturated Grid' (--> Avoid full initialization to safe memory), CG: 'Coarse Grid'
        #POINTS: Numpy array, that holds the indices of all explored points.
        #SHARES: Pandas series, that holds all explored shares.
        # SHARES.index correspond to the global indices,
        # as if all points were initialized from the start.
                    
        #step 1.0: initialization
        if self.bits_64:
            SHARES = pd.Series(dtype='float32')
            PROBS_values = np.array([], dtype='float32')
        else:
            SHARES = pd.Series(dtype='float32')
            PROBS_values = np.array([], dtype='float32')
            
        EXCLUDED = np.array([], dtype='int64')
        PROBS_index = np.array([], dtype='int64')
        draws_store = np.array([], dtype='int64')
                   
        #Marginal probabilities of explored and unknown paramter spaces.
        #No parameters explored yet.
        PROB_EXPLORED = 0
        PROB_EXPLORED_ref = 0
        PROB_UNKNOWN = 1
        
        if self.bits_64:
            @vectorize(["float64(float64, float64)"], nopython=True, target="parallel")
            def get_norm(x, sigma):
                """
                This method returns the probability of the normal distribution.

                Parameters
                ----------
                x : float
                    Euclidean distance to mean.
                sigma : float
                    Standard deviation.

                Returns
                -------
                probability : float
                    Probability of the normal distribution.

                """
                                
                a = np.sqrt(2 * np.pi) * sigma
                b = np.exp(-x ** 2 / (2 * sigma ** 2))
            
                probability = 1 / a * b
            
                return probability
            
            @guvectorize(
                ['float64[:], float64[:], float64[:], float64[:]'], 
                '(),(m),(m)->()', 
                nopython=True, 
                target="parallel"
                )
            def get_euclidean_vector(acc, point_A, point_B, euclidean): 
                """
                This methods calculates the euclidean distance between two points.

                Parameters
                ----------
                acc : Float
                    Distance between two points..
                point_A : array
                    Point in multi-dimensional space.
                point_B : array
                    Point in multi-dimensional space.
                euclidean : float
                    Euclidean distance.

                Returns
                -------
                euclidean : float
                    Euclidean distance.

                """
                #convention: Count indices, starting from last column.
                for m in prange(point_A.shape[0]):
                    acc[0] += (point_A[m] - point_B[m])**2
                euclidean[0] = sqrt(acc[0])
        else:
            @vectorize(["float32(float32, float32)"], nopython=True, target="parallel")
            def get_norm(x, sigma):
                """
                This method returns the probability of the normal distribution.

                Parameters
                ----------
                x : float
                    Euclidean distance to mean.
                sigma : float
                    Standard deviation.

                Returns
                -------
                probability : float
                    Probability of the normal distribution.

                """
                                
                a = np.sqrt(2 * np.pi) * sigma
                b = np.exp(-x ** 2 / (2 * sigma ** 2))
            
                probability = 1 / a * b
            
                return probability
            
            @guvectorize(
                ['float32[:], float32[:], float32[:], float32[:]'], 
                '(),(m),(m)->()', 
                nopython=True, 
                target="parallel"
                )
            def get_euclidean_vector(acc, point_A, point_B, euclidean): 
                """
                This methods calculates the euclidean distance between two points.

                Parameters
                ----------
                acc : Float
                    Distance between two points..
                point_A : array
                    Point in multi-dimensional space.
                point_B : array
                    Point in multi-dimensional space.
                euclidean : float
                    Euclidean distance.

                Returns
                -------
                euclidean : float
                    Euclidean distance.

                """
                #convention: Count indices, starting from last column.
                for m in prange(point_A.shape[0]):
                    acc[0] += (point_A[m] - point_B[m])**2
                euclidean[0] = sqrt(acc[0])
        
        def update_proximity(SHARES, CG, draws, size_space, space):
            """
            This method is incorporated to speed up estimation, i.e. the exploration
            of the parameter space. It updates the shares of the points, which
            are within an externally defined proximity to an estimated point.
            The method assumes a normal distribution, centered in the estimated point,
            wich flattens out with increasing distance of the points in proximity
            to the estimated point.
            

            Parameters
            ----------
            SHARES : array
                Array, which holds the shares of each estimated point.
            CG : array
                Short for "Coarse Grid". This array holds the points in the parameter space.
                It is partially extended to the values, which have been observed, to 
                save on memory.
            draws : array
                Array, which holds every point of the parameter space for which
                shares have been estimated.
            size_space : int
                The size of the parameter space.
            space : 2D-array
                2D-array, which indicates the dimensions of the parameter space.

            Returns
            -------
            SHARES : array
                Array, which holds the shares of each estimated point..

            """
                            
            for i in range(len(draws)):
                #print('Update draw: ', str(i))
                #get drawn index d and drawn point point_temp
                draw_temp = draws[i]
                point_temp = CG[i]
                #get minimum index
                if (draw_temp-updated_shares) < 0:
                    d_min = 0
                else:
                    d_min = draw_temp-updated_shares
                #get maximum index
                max_index = size_space
                if (draw_temp+updated_shares) > max_index:
                    d_max = max_index
                else:
                    d_max = draw_temp+updated_shares
                
                #get points in proximity of drawn point
                draws_proximity = np.arange(d_min, d_max, dtype='int32')
                
                points_proximity = get_points_from_draws_vector(space, draws_proximity)
                
                #get scale factors for unknown and known shares in proximity.
                #Aim: Sum of shares in updated shares (shares+proximity) = 1
                draws_in_SHARES = np.in1d(draws_proximity, SHARES.index)
                draws_not_in_SHARES = ~draws_in_SHARES
                count_unknown = np.sum(~draws_in_SHARES)
                
                size_shares = len(SHARES)
                size_shares_plus_proximity = size_shares + count_unknown
                
                scale_shares = size_shares / size_shares_plus_proximity
            
                #get maximum euclidean within proximity of point_temp
                if self.bits_64:
                    acc = np.array([0], dtype='float64')
                else:
                    acc = np.array([0], dtype='float32')

                euclidean_max = get_euclidean_vector(acc, points_proximity[-1], points_proximity[0])[0]
                
                sigma_ = euclidean_max/2
                if self.bits_64:
                    zero = np.array([0], dtype='float64')
                else:
                    zero = np.array([0], dtype='float32')
                prob_d_zero = get_norm(zero, sigma_)
                scale_norm = 1 / prob_d_zero
                
                #get_euclidean distances for all points in proximity
                if self.bits_64:
                    acc = np.array([0]*len(points_proximity), dtype='float64')
                else:
                    acc = np.array([0]*len(points_proximity), dtype='float32')
                    
                x_ = get_euclidean_vector(acc, point_temp, points_proximity)
                
                #get probability of normal distribution
                #Def: sigma=euclidean_max/2 --> 68% within proximity.
                prob_d_temp = get_norm(x_, sigma_)
                prob_d_temp_scale = prob_d_temp*scale_norm
                
                #scale SHARES to new size
                SHARES = SHARES*scale_shares
                #Append SHARES in proximity, which are out of SHARES
                SHARE_prox_unknown_temp = 1/size_shares_plus_proximity
                if self.bits_64:
                    SHARES_prox_unknown = pd.Series(
                        [SHARE_prox_unknown_temp]*count_unknown, 
                        index=draws_proximity[draws_not_in_SHARES], 
                        dtype='float64'
                        )
                else:
                    SHARES_prox_unknown = pd.Series(
                        [SHARE_prox_unknown_temp]*count_unknown, 
                        index=draws_proximity[draws_not_in_SHARES], 
                        dtype='float32'
                        )
                SHARES = SHARES.append(SHARES_prox_unknown)
                SHARES = SHARES[~SHARES.index.duplicated(keep='last')]
                                    
                #Update SHARES
                SHARES.loc[draws_proximity] = (
                    SHARES.loc[draw_temp] - SHARES.loc[draws_proximity]
                    ) * prob_d_temp_scale + SHARES.loc[draws_proximity]
                
                SHARES = SHARES / np.sum(SHARES)
                                                            
            return SHARES
        
        iter_outer = 0
        
        #if gpu == True:
        #    GPUtil.showUtilization()
        
        start = time.time()
        
        print('____EM algorithm starts.')
        
        count_progress = 1
        delta_outer_iteration = 0
        
        while PROB_EXPLORED_ref < PROBS_min:
            
            start_outer = time.time()
            
            
            #step 2.1: draw indices
            draws_unknown, draws_explored = get_draws(
                draws_per_iter, PROBS_values, PROBS_index, PROB_UNKNOWN
                )
            
            draws = np.append(draws_unknown, draws_explored)
            
            if count_progress*0.01 < PROB_EXPLORED_ref:
                print('________Iteration: ', str(iter_outer))
                print('________Last iteration took: ', delta_outer_iteration, 'seconds.')
                print('________Last delta_c_1: ', self.delta_c_1, 'seconds.')
                print('________Bool delta_c: ', self.start_c, 'seconds.')
                print('____________Explored parameter space: ', str(PROB_EXPLORED_ref*100), '%' )
                print('____________SHARES summed: ', np.sum(SHARES))
                print('________________ratio UNKNOWN: ', len(draws_unknown) / len(draws))
                print('________________ratio EXPLORED: ', len(draws_explored) / len(draws))
                
                count_progress += 1
                
            #Step 3: Get the shares and logit_probs for the drawn indices
            #   Step 3.1: Draw points (CG = Coarse Grid)
            CG = get_points_from_draws_vector(self.space, draws)
            #   Step 3.2: Recalculate SHARES & POINTS
            scale_shares_explored = len(SHARES) / len(draws)
            drawn_shares_explored = SHARES.loc[draws_explored].values*scale_shares_explored
            if self.bits_64:
                drawn_shares_unknown = np.array([1/len(draws)]*len(draws_unknown), dtype='float64')
            else:
                drawn_shares_unknown = np.array([1/len(draws)]*len(draws_unknown), dtype='float32')
            
            drawn_shares = np.append(drawn_shares_unknown, drawn_shares_explored)
            drawn_shares = drawn_shares / np.sum(drawn_shares)
            if self.bits_64:
                drawn_shares = drawn_shares.astype('float64')
            else:
                drawn_shares = drawn_shares.astype('float32')
                                            
            if gpu:
                #data management: move data to device
                d_points = cuda.to_device(CG)
                d_av = cuda.to_device(av)
                d_data = cuda.to_device(data)
                d_choice = cuda.to_device(choice)
                #calculation
                d_drawn_logit_probs = calculate_logit_gpu(
                    d_points, d_av, d_choice, d_data
                    )
                cuda.synchronize()
                #memory management: move data to host
                drawn_logit_probs = d_drawn_logit_probs.copy_to_host()
            else:
                drawn_logit_probs = calculate_logit_vector(CG, av, data)
                               
            #Step 4: Apply EM-algorithm on drawn indices
            convergence = 0
            iter_inner = 0
            expect_before = 0
                                                
            while convergence == 0:                    
                #calculate probability, that a person has the coefficients of a 
                #specific point, given his/her choice: point_probs = h_nc

                if gpu:                        
                    #memory management: Move data to device (pp)
                    d_shares = cuda.to_device(drawn_shares)
                    d_logit_probs = cuda.to_device(drawn_logit_probs)
                    
                    #GPU (EXP) calculation
                    d_pp_top = get_pp_top_gpu(d_shares, d_logit_probs)
                    cuda.synchronize()
                    
                    #copy to host (pp)
                    pp_top = d_pp_top.copy_to_host()
                    
                    #HOST calculations
                    sum_pp_top = np.sum(pp_top)
                    pp = pp_top / sum_pp_top
                    sum_pp = np.sum(pp)
                    sum_pp_axis = np.sum(pp, axis=1)
                    drawn_shares = sum_pp_axis / sum_pp
                    
                    #copy to device (EXP)
                    d_shares_update = cuda.to_device(drawn_shares)
                    d_pp = cuda.to_device(pp)
                    
                    #GPU calculations (EXP)
                    d_expect = get_expectation_gpu(d_shares_update, d_pp)
                    cuda.synchronize()
                    
                    #copy to host (EXP)
                    expect = d_expect.copy_to_host()
                    expect = np.sum(expect)
                    
                else:
                    expect, drawn_shares = get_expectation(drawn_shares, drawn_logit_probs)
                                        
                diff = abs(expect-expect_before)
                expect_before = expect
                iter_inner += 1
                if diff < tol and iter_inner > min_iter:
                    convergence = 1
                    break
                if iter_inner == max_iter:
                    break
                                
            #step 5: Update SHARES.
            #   step 5.1: Identify scale-factors for draws and previous shares.
            #   SHARES & drawn_shares, both sum to one. --> What is the joined scale?
            size_shares = len(SHARES)
            size_shares_updated = size_shares + len(draws_unknown)
            size_draws = len(draws)
            
            scale_draws = size_draws / size_shares_updated
            scale_shares = size_shares / size_shares_updated
            
            #print('scale_draws:',scale_draws)
            #print('scale_shares:',scale_shares)
            
            #   step 5.2: scale known shares and drawn shares
            SHARES = SHARES * scale_shares
            drawn_shares = drawn_shares * scale_draws
                                            
            #   step 5.3: Merge SHARES and drawn_shares
            if self.bits_64:
                SHARES_TO_MERGE = pd.Series(drawn_shares, index = draws, dtype='float64')
            else:
                SHARES_TO_MERGE = pd.Series(drawn_shares, index = draws, dtype='float32')
            index_intersection = np.intersect1d(SHARES_TO_MERGE.index, SHARES.index)
            #drop old values
            SHARES = SHARES.drop(index_intersection)
            #append updated values
            SHARES = SHARES.append(SHARES_TO_MERGE)
            
            
            #delete duplicates in SHARES and keep the last one added.
            
            #   step 5.4: Normalize SHARES, although they already should
            #   sum to one. --> Normalize due to numerical reasons.
            SHARES = SHARES / np.sum(SHARES)
            
            #   step 5.5: Update SHARES in SG, which have not been drawn, 
            #   based on distance to these values divided by average distance.    
            #   Assumption: The SHARES of all random parameters are distributed with a normal distribution.
            #   Method: The SHARE of point S_i equals the SHARE of point S_0, multiplied
            #   by the normal probability of the euclidean distance between i and 0, divided 
            #   by the maximum euclidean distance between two points within the chosen proximity.
                            
            if updated_shares > 0:
                SHARES = update_proximity(SHARES, CG, draws, self.size_space, self.space)
                #   step 5.6: Normalize SHARES
                SHARES = SHARES / np.sum(SHARES)
                                 
            #   step 5.7: Drop values below treshold and add those to EXCLUDE
            if len(SHARES) > SHARES_max:
                surplus = len(SHARES) - SHARES_max
                EXCLUDED_temp = SHARES.nsmallest(n=surplus, keep='all').index
                EXCLUDED = np.append(EXCLUDED, EXCLUDED_temp)
                SHARES = SHARES.drop(labels = EXCLUDED_temp)
                #   step 5.8: Normalize again.
                SHARES = SHARES / np.sum(SHARES)
            else:
                EXCLUDED_temp = SHARES.loc[SHARES==0].index
                EXCLUDED = np.append(EXCLUDED, EXCLUDED_temp)
                SHARES = SHARES.drop(labels = EXCLUDED_temp)
                            
            #print(SHARES)
            if iter_outer < blind_exploration:
                #Store the drawn indices and do NOT update PROBS & PROBS_iter.
                draws_store = np.append(draws_store, draws)     
                                    
            else:
                #Update PROBS 'regularly', after number of iteration 
                #exceeds specified value for 'blind_exploration'.
                
                #use PROB_EXPLORED_ref for printing, to display exploration since start.
                PROB_EXPLORED_ref = (len(SHARES)+len(EXCLUDED)) / (self.size_space)
                #use PROB_EXPLORED for calculation, since this 
                #step 6.0: Update PROBS. - Adjust PROBS from SHARES.
                if self.bits_64:
                    PROBS_values = np.array(SHARES.values, dtype='float64')
                else:
                    PROBS_values = np.array(SHARES.values, dtype='float32')
                PROBS_index = np.array(SHARES.index, dtype='int64')
                #step 6.1: Update PROBS. - Exclude temporary draws from future draws.
                index_to_zero = np.in1d(PROBS_index, draws_store)
                PROBS_values[index_to_zero] = 0
                self.check_PROBS_values = PROBS_values
                #acknowledges additional information. I.e. excluded points.
                PROB_EXPLORED = len(SHARES)*np.sum(PROBS_values) / (self.size_space - len(EXCLUDED))
                PROB_UNKNOWN = 1 - PROB_EXPLORED
                #step 6.2: scale PROBS_values after calculating PROB_EXPLORED!
                if np.sum(PROBS_values) > 0:
                    PROBS_values = PROBS_values / np.sum(PROBS_values)
                #step 6.3: Store drawn indices
                draws_store = np.append(draws_store, draws)  
                                    
            #count up one iteration.
            iter_outer += 1
            end_outer = time.time()
            delta_outer_iteration = end_outer - start_outer
            
        end = time.time()
        delta = end-start
        print('Estimation of shares took: ', str(delta), 'seconds.')
                  
        if np.sum(np.isnan(SHARES)):
            raise ValueError(
                'NaN-values detected in -shares-. Debug hint: You may adjust the parameter space (smaller).'
                )
        else:  
            self.shares = SHARES
            
                
    def estimate_logit(self, **kwargs):
        """
        This method estimates the coefficients of a standard MNL model.
        
        Parameters
        ----------
        
        kwarg stats : Boolean
            If True, summary statistics are returned as well. Defaults to True.
                    
        Returns
        -------
        list
            List of estimated coefficients of standard MNL model.

        """
        
        stats_sum = kwargs.get('stats', True)
                
        def loglike(x):
            
            #logged numerator of MNL model
            utility_single = (
                
                sum([                
                self.av[0][e] * self.choice[0][e] * (
                    sum(
                        [
                            (x[(self.count_c-1) + a] * 
                             self.data[self.param['constant']['fixed'][a] + '_' + str(0) + '_' + str(e)]) 
                            for a in range(self.no_constant_fixed)
                            ]
                        ) +
                    sum(
                        [
                            (x[(self.count_c-1) + self.no_constant_fixed + a] * 
                             self.data[self.param['constant']['random'][a] + '_' + str(0) + '_' + str(e)]) 
                            for a in range(self.no_constant_random)
                            ]
                        ) + 
                    sum(
                        [
                            (x[(self.count_c-1) + self.no_constant_fixed + self.no_constant_random + a] * 
                             self.data[self.param['variable']['fixed'][a] + '_' + str(0) + '_' + str(e)]) 
                            for a in range(self.no_variable_fixed)
                            ]
                        ) + 
                    sum(
                        [
                            (x[(self.count_c-1) + self.no_constant_fixed + self.no_constant_random + self.no_variable_fixed + a] * 
                             self.data[self.param['variable']['random'][a] + '_' + str(0) + '_' + str(e)]) 
                            for a in range(self.no_variable_random)
                            ]
                        ) 
                    )
                for e in range(self.av.shape[1])])
                
                +
                
                sum([
                    sum([
                        self.av[c][e] * self.choice[c][e] * (
                        x[c-1] +
                        sum(
                            [
                                (x[(self.count_c-1) + a] * 
                                 self.data[self.param['constant']['fixed'][a] + '_' + str(c) + '_' + str(e)]) 
                                for a in range(self.no_constant_fixed)
                                ]
                            ) +
                        sum(
                            [
                                (x[(self.count_c-1) + self.no_constant_fixed + a] * 
                                 self.data[self.param['constant']['random'][a] + '_' + str(c) + '_' + str(e)]) 
                                for a in range(self.no_constant_random)
                                ]
                            ) + 
                        sum(
                            [
                                (x[
                                    (self.count_c-1) + 
                                    self.no_constant_fixed + 
                                    self.no_constant_random + 
                                    (self.no_variable_fixed+self.no_variable_random)*c + a
                                    ] * 
                                 self.data[self.param['variable']['fixed'][a] + '_' + str(c) + '_' + str(e)]) 
                                for a in range(self.no_variable_fixed)
                                ]
                            ) + 
                        sum(
                            [
                                (x[
                                    (self.count_c-1) + 
                                    self.no_constant_fixed + 
                                    self.no_constant_random + 
                                    (self.no_variable_fixed+self.no_variable_random)*c + 
                                    self.no_variable_fixed + a
                                    ] * 
                                 self.data[self.param['variable']['random'][a] + '_' + str(c) + '_' + str(e)]) 
                                for a in range(self.no_variable_random)
                                ]
                            ) 
                        ) for e in range(self.av.shape[1])])
                    for c in range(1,self.count_c)
                    ]
                )
            )
                                    
            #logged denominator of MNL model
            utility_all = (
                    sum([
                        self.av[0][e] * (
                            np.exp(
                                sum([
                                        (x[(self.count_c-1) + a] * 
                                         self.data[self.param['constant']['fixed'][a] + '_' + str(0) + '_' + str(e)]) 
                                        for a in range(self.no_constant_fixed)
                                        ]
                                    ) +
                                sum([
                                        (x[(self.count_c-1) + self.no_constant_fixed + a] * 
                                         self.data[self.param['constant']['random'][a] + '_' + str(0) + '_' + str(e)]) 
                                        for a in range(self.no_constant_random)
                                        ]
                                    ) + 
                                sum([
                                        (x[(self.count_c-1) + self.no_constant_fixed + self.no_constant_random + a] * 
                                         self.data[self.param['variable']['fixed'][a] + '_' + str(0) + '_' + str(e)]) 
                                        for a in range(self.no_variable_fixed)
                                        ]
                                    ) + 
                                sum([
                                        (x[(self.count_c-1) + self.no_constant_fixed + self.no_constant_random + self.no_variable_fixed + a] * 
                                         self.data[self.param['variable']['random'][a] + '_' + str(0) + '_' + str(e)]) 
                                        for a in range(self.no_variable_random)
                                        ]
                                    ) 
                                )
                            ) 
                    for e in range(self.av.shape[1])])
                
                +
                
                sum([
                    sum([
                        self.av[c][e] * (
                            np.exp(
                                x[c-1] +
                                sum([
                                        (x[(self.count_c-1) + a] * 
                                         self.data[self.param['constant']['fixed'][a] + '_' + str(c) + '_' + str(e)]) 
                                        for a in range(self.no_constant_fixed)
                                        ]
                                    ) +
                                sum([
                                        (x[(self.count_c-1) + self.no_constant_fixed + a] * 
                                         self.data[self.param['constant']['random'][a] + '_' + str(c) + '_' + str(e)]) 
                                        for a in range(self.no_constant_random)
                                        ]
                                    ) + 
                                sum([
                                        (x[
                                            (self.count_c-1) + 
                                            self.no_constant_fixed + 
                                            self.no_constant_random + 
                                            (self.no_variable_fixed+self.no_variable_random)*c + a
                                            ] * 
                                         self.data[self.param['variable']['fixed'][a] + '_' + str(c) + '_' + str(e)]) 
                                        for a in range(self.no_variable_fixed)
                                        ]
                                    ) + 
                                sum([
                                        (x[
                                            (self.count_c-1) + 
                                            self.no_constant_fixed + 
                                            self.no_constant_random + 
                                            (self.no_variable_fixed+self.no_variable_random)*c + 
                                            self.no_variable_fixed + a
                                            ] * 
                                         self.data[self.param['variable']['random'][a] + '_' + str(c) + '_' + str(e)]) 
                                        for a in range(self.no_variable_random)
                                        ]
                                    ) 
                                )
                            )
                        for e in range(self.av.shape[1])])
                    for c in range(1,self.count_c)])
            )
               
            self.check_utility_all = utility_all
                             
            #logged probability of MNL model
            log_prob = utility_single - np.log(utility_all)
            
            res = -np.sum(log_prob)
                                    
            return res
        
        #initialize optimization of MNL coefficients
        no_param = (
            self.no_constant_fixed + 
            self.no_constant_random + 
            self.count_c * (self.no_variable_fixed + self.no_variable_random)
            )
        no_coeff = int(self.count_c-1 + no_param)
        x0 = np.zeros((no_coeff,), dtype=float)

   	    #definition of estimation bounds. Alternative norm_alt is normalized.
        bounds_def = np.empty((no_coeff,), dtype=object)
        for c in range(self.count_c-1):
            bounds_def[c] = (None, None)
        
        for a in range(no_param):
            bounds_def[a+self.count_c-1] = (None, None)
            
        #derive bounds from external parameter specification.
        for p in list(self.param_ext):
            for a in range(self.no_constant_fixed):
                if p == self.param['constant']['fixed'][a]:  
                    c_list = self.param_ext[p][1]
                    param_ext_list = self.param_ext[p][0]
                    for i, c in enumerate(c_list):
                        bounds_def[
                            (self.count_c-1) + a
                            ] = (param_ext_list[i], param_ext_list[i])
            for a in range(self.no_constant_random):
                if p == self.param['constant']['random'][a]:   
                    c_list = self.param_ext[p][1]
                    param_ext_list = self.param_ext[p][0]
                    for i, c in enumerate(c_list):
                        bounds_def[
                            (self.count_c-1) + 
                            self.no_constant_fixed + a
                            ] = (param_ext_list[i],param_ext_list[i])
            for a in range(self.no_variable_fixed):
                if p == self.param['variable']['fixed'][a]:   
                    c_list = self.param_ext[p][1]
                    param_ext_list = self.param_ext[p][0]
                    for i, c in enumerate(c_list):
                        bounds_def[
                            (self.count_c-1) + self.no_constant_fixed + 
                            self.no_constant_random + 
                            (self.no_variable_fixed+self.no_variable_random)*c + a
                            ] = (param_ext_list[i], param_ext_list[i])
            for a in range(self.no_variable_random):
                if p == self.param['variable']['random'][a]:   
                    c_list = self.param_ext[p][1]
                    param_ext_list = self.param_ext[p][0]
                    for i, c in enumerate(c_list):
                        bounds_def[
                            (self.count_c-1) + self.no_constant_fixed + 
                            self.no_constant_random + 
                            (self.no_variable_fixed+self.no_variable_random)*c + 
                            self.no_variable_fixed + a
                            ] = (param_ext_list[i], param_ext_list[i])
        
        # optimization of objective function: Nelder-Mead, L-BFGS-B
        res = minimize(
            loglike, 
            x0, 
            method="L-BFGS-B", 
            tol=1e-6, 
            bounds=bounds_def,
            jac='cs'
            )
        res_param = res.x
        
        self.check_res = res
        
        print(res_param)
 
        if stats_sum:   
            print('Calculation of summary statistics starts.')
            
            data_safe = self.data
            size_subset = int(len(self.data) / 10)
            param_cross_val = {j: [] for j in range(no_coeff)}
            for i in range(10):
                print('Cross-validation run: ', str(i))
                self.data = data_safe[size_subset*i:size_subset*(i+1)]
                
                self.choice = np.zeros((self.count_c,self.count_e,len(self.data)), dtype=np.int64)
                self.av = np.zeros((self.count_c,self.count_e,len(self.data)), dtype=np.int64)
                for c in range(self.count_c):
                    for e in range(self.count_e):
                        self.choice[c][e] = self.data["choice_" + str(c) + "_" + str(e)].values
                        self.av[c][e] = self.data["av_" + str(c) + "_" + str(e)].values

                res = minimize(
                    loglike, 
                    x0, 
                    method="L-BFGS-B", 
                    tol=1e-6, 
                    bounds=bounds_def,
                    jac='cs')
                #iterate over estimated coefficients
                for j, param in enumerate(res.x):
                    param_cross_val[j].append(param)
            
            self.check_param_cross_val = param_cross_val
            
            #calculate statistics
            self.t_stats = [stats.ttest_1samp(param_cross_val[j], 0) for j in range(no_coeff)]
            self.std_cross_val = [np.std(param_cross_val[j]) for j in range(no_coeff)]
     
            #reset self.data, self.av, self.choice
            self.data = data_safe
            #define choices and availabilities
            self.choice = np.zeros((self.count_c,self.count_e,len(self.data)), dtype=np.int64)
            self.av = np.zeros((self.count_c,self.count_e,len(self.data)), dtype=np.int64)
            for c in range(self.count_c):
                for e in range(self.count_e):
                    self.choice[c][e] = self.data["choice_" + str(c) + "_" + str(e)].values
                    self.av[c][e] = self.data["av_" + str(c) + "_" + str(e)].values
                             
        print('LL_0: ', loglike(x0))
        print('LL_final: ', loglike(res_param))
                
        return res_param