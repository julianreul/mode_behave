# -*- coding: utf-8 -*-
"""
Created on Sat May  7 11:46:35 2022

@author: j.reul
"""

import unittest
import numpy as np
import pandas as pd
from scipy import stats

import mode_behave_public as mb

class TestEstimation(unittest.TestCase):       
        
    def get_artificial_data(self):
        """
        This method was used to create the artificial dataset, upon which
        the tested estimation routines below are running.

        Raises
        ------
        AttributeError
            DESCRIPTION.

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
                
        size_dataset = 1000
        
        #DEFINE COPULA
        #mean values
        mu_c = np.array([0,0,0,0])
        #Covariance between attributes and choice
        psi_choice = 0.5
        psi_between = 0.8
        dist_attr = "weibull"
        #risks: Choice, attr_1, attr_2, attr_3
        cov_c = np.array(
            [
                [1, psi_choice, psi_choice, psi_choice],
                [psi_choice, 1, psi_between, psi_between],
                [psi_choice, psi_between, 1, psi_between],
                [psi_choice, psi_between, psi_between, 1],
                ]
            )
        dist_c = stats.multivariate_normal(mean=mu_c, cov=cov_c)
        #____obtain random sample from copula distribution
        sample_c = dist_c.rvs(size=size_dataset, random_state=42)
        #____obtain marginals from copula distribution
        choice = sample_c[:,0]
        attr_x = sample_c[:,1]
        attr_y = sample_c[:,2]
        attr_z = sample_c[:,3]
        
        #TRANSFORM TO UNIFORM
        u_choice = stats.norm.cdf(choice)
        u_attr_x = stats.norm.cdf(attr_x)
        u_attr_y = stats.norm.cdf(attr_y)
        u_attr_z = stats.norm.cdf(attr_z)
        
        #DEFINE MARGINAL DISTRIBUTIONS
        
        dist_choice = stats.uniform()
        if dist_attr == "uniform":
            dist_attr_x = stats.uniform()
            dist_attr_y = stats.uniform()
            dist_attr_z = stats.uniform()
        elif dist_attr == "weibull":
            dist_attr_x = stats.weibull_min(c=1.5)
            dist_attr_y = stats.weibull_min(c=1.1)
            dist_attr_z = stats.weibull_min(c=0.5)
        else:
            raise AttributeError("unknown dist_attr.")
            
        #DRAW VECTOR FROM COPULA
        beta = np.array(list(zip(
            dist_choice.ppf(u_choice), 
            dist_attr_x.ppf(u_attr_x), 
            dist_attr_y.ppf(u_attr_y),
            dist_attr_z.ppf(u_attr_z), 
            )))
        
        #Create pandas dataframe out of beta
        dataset = pd.DataFrame(data=beta, columns=["choice_float", "attr_x", "attr_y", "attr_z"])
        
        #def get choice int
        def get_choice_int(choice_float):
            if choice_float < 0.33:
                return 0
            elif choice_float < 0.66:
                return 1
            else:
                return 2
            
        dataset["choice_int"] = dataset["choice_float"].apply(get_choice_int)
        dataset["choice_0"] = (dataset["choice_int"] == 0)*1
        dataset["choice_1"] = (dataset["choice_int"] == 1)*1
        dataset["choice_2"] = (dataset["choice_int"] == 2)*1
        
        #create columns for attributes
        dataset["attr_x_0"] = dataset["attr_x"].copy()
        dataset["attr_x_1"] = dataset["attr_x"].copy()
        dataset["attr_x_2"] = dataset["attr_x"].copy()
        
        dataset["attr_y_0"] = dataset["attr_y"].copy()
        dataset["attr_y_1"] = dataset["attr_y"].copy()
        dataset["attr_y_2"] = dataset["attr_y"].copy()
        
        dataset["attr_z_0"] = dataset["attr_z"].copy()
        dataset["attr_z_1"] = dataset["attr_z"].copy()
        dataset["attr_z_2"] = dataset["attr_z"].copy()
        
        #create columns for availability
        dataset["av_0"] = 1
        dataset["av_1"] = 1
        dataset["av_2"] = 1
        
        #delete unnecessary columns
        dataset = dataset.drop(columns=[
            "choice_float",
            "attr_x", 
            "attr_y",
            "attr_z",
            "choice_int"
            ])
        
        #append "_0" to each column to fit required data format.
        for col in dataset.columns:
            dataset = dataset.rename(columns={col : col + "_0"})
            
        return dataset
           
    
    def test_estimate_logit(self):
        """
        Integration test of the method -estimate_logit()-

        """
        
        param_fixed = []
        param_random = [
            "attr_x",
            "attr_y",
            "attr_z"
            ]
        
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
            
        artificial_data = self.get_artificial_data()
        
        #Initialize model
        model = mb.Core(
            param=param_temp, 
            data_in=artificial_data, 
            alt=3,
            equal_alt=1,
            include_weights=False,
            )
                 
        model.initial_point = model.estimate_logit(stats=False)
        
        initial_point_compare = np.genfromtxt(model.PATH_ModelParam + "initial_point_artificial_data.csv", delimiter=",")
        
        print("Initial_point GitHub:", model.initial_point)
        print("Initial_point Local:", initial_point_compare)
        
        #test estimation of initial_point (via the method estimate_logit())
        self.assertTrue(np.allclose(model.initial_point, initial_point_compare, atol=0.1))
    
    
    def test_estimate_mixed_logit(self):
        """
        Integration test of the method -estimate_mixed_logit()-
        Utilized CPU hardware only.

        """
        param_fixed = []
        param_random = [
            "attr_x",
            "attr_y",
            "attr_z"
            ]
        
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
            
        artificial_data = self.get_artificial_data()
        
        #Initialize model
        model = mb.Core(
            param=param_temp, 
            data_in=artificial_data, 
            alt=3,
            equal_alt=1,
            include_weights=False,
            )
        
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
        
        initial_point_compare = np.genfromtxt(model.PATH_ModelParam + "initial_point_artificial_data.csv", delimiter=",")
        shares_compare = np.genfromtxt(model.PATH_ModelParam + "shares_artificial_data.csv", delimiter=",")
        points_compare = np.genfromtxt(model.PATH_ModelParam + "points_artificial_data.csv", delimiter=",")

        #test estimation of initial_point (via the method estimate_logit())
        self.assertTrue(np.allclose(model.initial_point, initial_point_compare, atol=0.1))
        
        #test estimation of shares
        self.assertTrue(np.allclose(model.shares, shares_compare, atol=0.1))
        
        #test definition of parameter space (points)
        self.assertTrue(np.allclose(np.array(model.points), points_compare, atol=0.1))



if __name__ == '__main__':
    unittest.main()