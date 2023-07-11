"""
The module incorporates the class "Core", which inherits functionality from the 
classes "Estimation", "Simulation" and "PostAnalysis".
It defines the core attributes, structure and type of the
discrete choice models to be simulated or estimated. 
Two different types of discrete choice models are differentiated - 
Multinomial logit (MNL) and nonparametrics mixed logit models (MXL).

"""

import numpy as np
import pandas as pd
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
        model_type : string, optional
            Indicates, whether to initiate a simulation- or estimation-model.
        param : dict, optional
            Holds the names of all attributes in utility function.
            list(param) = ['constant', 'variable'] --> Disctinction between variables
            that are constant over alternatives or vary, respectively.
            list(param['constant']) = ['fixed', 'random'] --> Distinction
            between variables that are not randomly distributed within
            a group of decision-makers ('fixed') and those, that are randomly
            distributed with a discrete distribution ('random')
        max_space : int, optional
            Maximum number of data points within parameter space.
        alt: int, optional
            Number of discrete choice alternatives.
        equal_alt: int, optional
            Maximum number of equal alternatives (with different attributes)
            in a single observation/choice set.
            E.g. mode choice: Available are two buses and one car. This would
            lead to a maximum of two equal alternatives (two buses).
        norm_alt : int, optional
            Defines, which alternative shall be normalized. Defaults to 0.
        include_weights : boolean, optional
            If True, the model searches for a column in the input data, which
            is called "weight". This column indicates the weight of each
            observation in the input data. Defaults to True.
        data_name : str, optional
            Defines the filename of the file, which holds the base data
            to calibrate the MNL or MXL model on.
        data_in : DataFrame, optional
            Alternative to -data_name-, if survey data is already loaded
            to the workspace as a pandas dataframe.
        data_index : array, optional
            An array, which holds the index values of those datapoints
            of the base data, which shall be used for estimation.
        initial_point_name : str, optional
            The filename of the file, which holds the estimated MNL parameters.
        dc_type : str, optional
            Determines the model type for simulation: MNL or MXL model.
            Depends on which estimated model parameters are available.
        dict_specific_travel_cost : dict, optional
            External specification of transport costs. Relevant for the
            simulation of mode choice.

        Returns
        -------

        """

        self.model_type = kwargs.get("model_type", "estimation")

        # define path to input data
        PATH_MODULE = os.path.dirname(__file__)
        sep = os.path.sep

        self.PATH_InputData = PATH_MODULE + sep + "InputData" + sep
        self.PATH_ModelParam = PATH_MODULE + sep + "ModelParam" + sep
        self.PATH_Visualize = PATH_MODULE + sep + "Visualizations" + sep

        if self.model_type == "simulation":

            self.asc_offset_hh_cars = config.asc_offset_hh_cars

            # load previously estimated model parameters,
            # if model-type is simulation.
            self.simulation_type = kwargs.get("simulation_type", [])
            self.initial_point_in = kwargs.get("initial_point_in", None)
            if self.initial_point_in is not None:
                print("Prepare self-configured simulation")
                self.initial_point = self.initial_point_in.copy()
                self.param = kwargs.get("param", {})
                if self.param:
                    self.no_constant_fixed = len(self.param["constant"]["fixed"])
                    self.no_constant_random = len(self.param["constant"]["random"])
                    self.no_variable_fixed = len(self.param["variable"]["fixed"])
                    self.no_variable_random = len(self.param["variable"]["random"])

            else:
                if self.simulation_type == "car_ownership":
                    print("Prepare car ownership simulation")
                    self.initial_point = config.initial_point_car_ownership

                elif self.simulation_type == "mode_choice":
                    print("Prepare mode choice simulation")

                    self.initial_point = config.initial_point_mode
                    self.log_param = config.log_param
                    dict_specific_travel_cost_ext = kwargs.get(
                        "dict_specific_travel_cost", {}
                    )
                    cc_cost_ext = kwargs.get("cc_cost", False)
                    if len(dict_specific_travel_cost_ext) > 0:
                        self.dict_specific_travel_cost = dict_specific_travel_cost_ext
                    else:
                        self.dict_specific_travel_cost = (
                            config.dict_specific_travel_cost
                        )
                    if cc_cost_ext:
                        self.cc_cost = cc_cost_ext
                    else:
                        self.cc_cost = config.cc_cost_2020

                else:
                    raise AttributeError(
                        """
                        No pre-estimated parameters for an MNL-model are provided.
                        If data is available, please indicate the -simulation_type- argument
                        to find the data in the package-folder ./InputData
                        If no data is available, please estimate MNL-data first.
                        """
                    )

        else:
            # define path to input data
            self.data_name = kwargs.get("data_name", None)
            self.data_in = kwargs.get("data_in", None)

            # random or fixed within parameter space of Mixed Logit
            self.param = kwargs.get("param", False)
            self.count_c = kwargs.get("alt", False)
            self.count_e = kwargs.get("equal_alt", False)

            # check, if necessary values have been specified.
            if self.param == False:
                raise ValueError("Argument -param- needs to be specified!")
            if self.count_c == False:
                raise ValueError("Argument -alt- needs to be specified!")
            if self.count_e == False:
                raise ValueError("Argument -equal_alt- needs to be specified!")

            self.no_constant_fixed = len(self.param["constant"]["fixed"])
            self.no_constant_random = len(self.param["constant"]["random"])
            self.no_variable_fixed = len(self.param["variable"]["fixed"])
            self.no_variable_random = len(self.param["variable"]["random"])

            self.data_index = kwargs.get("data_index", np.array([]))

            self.include_weights = kwargs.get("include_weights", True)

            print("Data wrangling.")

            try:
                self.data = self.data_in.copy()
            except:
                try:
                    self.data = pd.read_csv(
                        self.PATH_InputData + self.data_name + ".csv", sep=","
                    )
                    if len(list(self.data)) == 1:
                        raise ValueError(
                            'Check the separator of the imported .csv-files. Should be ","-separated.'
                        )
                except:
                    raise AttributeError(
                        "No input data specified. Define -data_in- or -data_name-."
                    )

            self.data = self.data.reset_index(drop=True)

            # get data-points from indicated indices
            if self.data_index.size:
                self.data = self.data.iloc[self.data_index]

            print("Length of dataset:", len(self.data))

            # define choices and availabilities
            # scale availabilities, if weights are provided in input-data
            if "weight" in self.data.columns and self.include_weights == True:
                self.weight_vector = self.data["weight"].values.copy()
            else:
                self.weight_vector = np.ones(shape=len(self.data), dtype=np.float64)
            self.choice = np.zeros(
                (self.count_c, self.count_e, len(self.data)), dtype=np.int64
            )
            self.av = np.zeros(
                (self.count_c, self.count_e, len(self.data)), dtype=np.float64
            )
            for c in range(self.count_c):
                for e in range(self.count_e):
                    self.choice[c][e] = self.data[
                        "choice_" + str(c) + "_" + str(e)
                    ].values
                    self.av[c][e] = self.data["av_" + str(c) + "_" + str(e)].values
            self.av_backup = self.av.copy()
