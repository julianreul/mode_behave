Model Purpose and General Information
=====================================
MO|DE.behave is a Python-based software package for the estimation and 
simulation of discrete choice models. The purpose of this software is to enable 
the rapid quantitative analysis of survey data on choice behavior, 
utilizing advanced discrete choice methods. 
Therefore, MO|DE.behave incorporates estimation routines for conventional 
multinomial logit models, as well as for mixed logit models with nonparametric 
distributions.
Furthermore, MO|DE.behave contains a set of post-processing tools for visualizing 
estimation and simulation results. Additionally, pre-estimated 
discrete choice simulation methods for transportation research are included to 
enrich the software package for this specific community.

On mixed logit models:
In recent years, a new modeling approach in the field of discrete choice theory 
became popular – the mixed logit model (see Train, K. (2009): "Mixed logit", 
in Discrete choice methods with simulation (pp. 76–93), Cambridge University Press). 
Conventional discrete choice models only have a limited capability to describe 
the heterogeneity of choice preferences within a base population, i.e., 
the divergent choice behavior of different individuals or consumer groups can 
only be studied to a limited degree. Mixed logit models overcome this deficiency and 
allow for the analysis of preference distributions across base populations.

Use MO|DE.behave via web app: https://julianreul-streamlit-mode-behave-home-nmwcdr.streamlit.app/

Installation
============
1. Download or clone the repository to a local folder.
#. Open (Anaconda) Prompt.
#. Create a new environment from reference_environment.yml file::

      conda env create -f reference_environment.yml
      
#. Activate the environment with::

      conda activate env_mode_behave
      
#. cd to the directory, where you stored the repository and where the setup.py file is located.

#. In this folder run::
    
      pip install -e .
      
#. Alternatively, run::
      
      pip install mode-behave


Workflow
========
1. Import model with::

      import mode_behave_public as mb
      
2. Load data with (exemplary import requires pandas-module)::
    
      example_data = pd.read_csv(PATH_TO_DATA + "example_data.csv")

2. Initialize a model with::
    
      model = mb.Core(
          param = param_temp, 
          data_in=example_data, 
          alt=4,
          equal_alt=1,
          include_weights=False
          )
      
   The structure of the input data and the parameter-input are given below.

3. Estimate the model with::

      model.estimate_mixed_logit(**kwargs)  
      
The estimation of the mixed logit model can be modified by definition of keyword-arguments
during instantiation and within the estimation-method itself.

| **Arguments for instantiation** (ov.Core(...)):
| dict param: Indicated the names of the model attributes. The attribute-names 
|       shall be derived from the column names of the input data.
| str data_name: Indicates the name of the input data-file. 
| int alt: Indicates the number of considered choice alternatives.
| int equal_alt: Indicates the maximum number of equal choice alternatives per choice set.
|
| **Keyword-arguments for instantiation** (ov.Core(...)): 
| boolean include_weights: If this is set to True, the model will search for a
|     column in the input-data, called "weight", which indicates the weight
|     for each observation. Defaults to True.
|
| **Keyword-arguments for estimation-method (model.estimate_mixed_logit(...))**:
| int min_inter: Min. iterations for EM-algorithm.
| int max_iter: Max. iterations for EM-algorithm.
| float tol: Numerical tolerance of EM-algorithm.
| bool bit_64: Defaults to False. If set to True, all numbers are calculated
|     in 64-bit format, which increases precision, but also runtime.
| str space_method: Defines the chosen method to span the parameter space for the mixed logit estimation.
| int scale_space: Defines the size of the space, relative to the chosen space_method.
| int max_shares: Defines the maximum number of points to be observed in the parameter space.
|
      
4. The package can also be used to estimate multinomial logit models::

      model.estimate_logit(**kwargs)  
      
| **Keyword-arguments for estimation-method (model.estimate_logit(...))**:
| bool stats: If set to True, t-statistics from the estimation process are evaluated.
|

5. An exemplary model workflow is provided with the package and can be accessed via the following path::

    PATH_TO_PACKAGE/mode_behave_public/Deployments/example_estimation.py


Structure of Parameters and Input Data
======================================

1. Input data

   The input dataset contains the observations with which the model is 
   calibrated. The input data is called with the specified string of the
   keyword-argument *data_name*. The input data shall be placed in .csv- or 
   .pickle-format within the subfolder *InputData* of the package *mode_behave*.
   The data shall follow the structure below:
   
   * Rows: Observations.
   * Columns:
         - One column per parameter of the utility function AND per alternative AND per equal alternative.
           Specified as: 'Attribute_name_' + str(no_alternative) + str(no_equal_alternative)
         - One column for the choice-indication of EACH alternative AND per equal alternative.
           Specified as: 'choice_' + str(no_alternative) + str(no_equal_alternative)
         - One column per alternative AND per equal alternative, indicating the availability.
           Specified as: 'av_' + str(no_alternative) + str(no_equal_alternative)
         - If a parameter is constant across alternatives or equal alternatives, then let the columns be equal.
         - Furthermore, the observations can be given a weight. Therefore, an additional column needs to be provided, named 'weight'. - Without any further suffix.
   * Index: The index shall start from '0'.
          
2. Initialization argument 'param':
    
   'param' is specified as a dictionary containing the attribute names of the 
   utility function, sorted by type.
   
   * param['constant']['fixed']: Attributes, which are constant over choice 
     options and fixed within the parameter space. 
   * param['constant']['random']: Attributes, which are constant over choice 
     options and randomly distributed over the parameter space. 
   * param['variable']['fixed']: Attributes, which vary over choice 
     options and are fixed within the parameter space. 
   * param['variable']['random']: Attributes, which vary over choice 
     options and are randomly distributed over the parameter space. 
     
3. The vector x, containing the initial estimates for the logit coefficients.

   The coefficients in vector x (solution vector of maximum likelihood optimization)
   follow a certain structure (alternatives=alt):
   
   * x[:(alt-1)]: ASC-constants for the alternatives 1-#of alternatives. ASC for choice option 0 defaults to 0.
   * x[(alt-1):(alt-1)+no_constant_fixed]: Coefficients of constant and fixed attributes.
   * x[(alt-1)+no_constant_fixed:(alt-1)+(no_constant_fixed+no_constant_random)]: 
     Coefficients of constant and fixed attributes.   
   * x[(alt-1)+(no_constant_fixed+no_constant_random):(alt-1)+(no_constant_fixed+no_constant_random)+no_variable_fixed*alt]: 
     Coefficients of variable (thus multiplication with alternatives) 
     and fixed attributes.
   * x[(alt-1)+(no_constant_fixed+no_constant_random)+no_variable_fixed*alt:(alt-1)+(no_constant_fixed+no_constant_random)+(no_variable_fixed+no_variable_random)*alt]: 
     Coefficients of variable and random attributes.
      
Theoretical Background
======================
A mixed logit model is a multinomial logit model (MNL), in which the coefficients 
do not take a single value, but are distributed over a parameter space. 
Within this package, the mixed logit models 
are estimated on a discrete parameter space, which is specified by the researcher (nonparametric design).
The discrete subsets of the parameter space are called classes, 
analogously to latent class models (LCM). The goal of the estimation procedure
is to estimate the optimal share, i.e. weight, of each class within the discrete parameter space.
The algorithm roughly follows the procedure below:

1. Estimate initial coefficients of a standard multinomial logit model.
2. Specify a continuous parameter space for the random coefficients with
   the mean and the standard deviation of each initially calculated random coefficient. 
   (The standard deviation can be calculated from a k-fold cross-validation.)
   Alternatively, the parameter space can be defined via the absolute values
   of the parameters.
3. Draw points (maximum number of point = -max_shares-) from the parameter space via latin hypercube sampling.
3. Estimate the optimal share for each drawn point with an expectation-maximization (EM) algorithm. (see Train, 2009)

      
Further reading:

* Train, K. (2009): "Mixed logit", in Discrete choice methods with simulation (pp. 76–93), Cambridge University Press
* Train, K. (2008): "EM algorithms for nonparametric estimation of mixing distributions", in Journal of Choice Modelling, 1(1), 40–69, https://doi.org/10.1016/S1755-5345(13)70022-8
* Train, K. (2016): "Mixed logit with a flexible mixing distribution", in Journal of Choice Modelling, 19, 40–53, https://doi.org/10.1016/j.jocm.2016.07.004
* McFadden, D. and Train, K. (2000): "Mixed MNL models for discrete response", in Journal of Applied Econometrics, 15(5), 447-470, https://www.jstor.org/stable/2678603 

Post-Analysis
=============

1. Access of estimated coefficients and summary statistics

   * **model.shares**: Estimated shares of discrete classes within parameter space.
   * **model.points**: Parameter space of random coefficients.
   * **model.initial_point**: Coefficients of initially estimated logit model.
     
2. Visualization of parameter space::

      model.visualize_space(**kwargs)
      
      Most important keyword-argument is "k". - "k" incidates the number of cluster
      centers, to which the estimated random parameters of the mixed logit model
      shall be attributed. The cluster centers indicate different potential
      choice or consumer groups. This method clusters the estimated random preferences
      and shows the position of the cluster centers as well as the overall distribution
      of estimated random parameters across the whole parameter space.
      
3. Forecast with cluster centers::

    model.forecast(method, **kwargs)
                
    "method" indicates the type of the discrete choice model ("MNL", "MXL", or "LC" for latent class).
    In **kwargs, also "k" can be given to indicate the number of cluster
    centers which shall be analyzed. This method forecasts the mean choice, based
    on the estimated parameters of each cluster center and the attribute values
    of the base data. It is a good reference point to study the diverging choice
    behavior of each cluster center. Furthermore, the keyword-argument
    "sense_scenarios" can be given to study model sensitivities by 
    indicating a relative change in the value of certain model attributes.

4. Cluster the drawn points from the parameter space to similar preference groups (e.g. consumer groups)::

    model.cluster_space(method, k, **kwargs)
    
    "method" indicates the clustering algorithm, e.g. kmeans. 
    "k" indicates the number of cluster centers.
    The output of this method is the classification of the drawn points
    from the parameter space into clusters. The second output are
    the calculated cluster centers. 
    The clusters can be interpreted as consumer groups.

5. Assignment of observations to cluster centers::
    
    model.assign_to_cluster(**kwargs)
    
    This method calculates probabilities for each observation in the base data,
    which indicate the likelihood with which an observation belongs to a 
    cluster center (the method internally calls self.cluster_space to
    determine the cluster centers). 
    This method is useful to characterize the consumer groups.
          
Simulation
==========

The model incorporates a class **Simulation**, which contains customized
methods to simulate previously estimated choice models.
In order to simulate choice probabilities, the model must be instantiated as follows::

   model = ov.Core(model_type = 'simulation', simulation_type = 'mode_choice')
   
The keyword-argument *simulation_type* specifies which kind of simulation
shall be conducted.
Currently only MNL-simulations are implemented.

The following MNL-simulations are currently available:

**MNL-Model for Mode-Choice (simulation_type = 'mode_choice')**::

    model.simulate_mode_choice(agegroup, occupation, regiontype, distance, av)
    
The method simulates the probability of mode choice for ten different modes
(Walking, Biking, MIV-self, MIV-co, bus_near, train_near, train_city, bus_far, train_far, carsharing).
Input parameters are the agegroup of the simulated agent (1: <18, 2: 18-65, 3: >65),
the occupation (1: full-time work, 2: part-time, 3: education, 4: no occupation),
the regiontype of residence (according to RegioStaR7 - BMVI classification),
distance (travel cost and time are derived from this variable, based on 
cost-assumptions for the year 2020. Also, the regiontype for the calculation
of average speeds is assumed to be identical with the specified regiontype
of the home location of the agent),
as well as the availability of each mode in numpy-array format.
Filename of pre-estimated model parameters: 'initial_point_mode'

**MNL-model for the probability of the number of cars per households (simulation_type = 'car_ownership')**::

   model.simulate_hh_cars(urban_region, rural_region, hh_size,
                         adults_working, children, htype, quali_opnv, sharing,
                         relative_cost_per_car, age_adults_scaled)
                         
The method simulates the probability, that a household owns 0-3+ cars (4 discrete alternatives).
Input paramters are the regiontype of residence in I/O-format according to 
RegioStaR2 BMVI classification (e.g.: urban_region = 1, rural_region = 0),
the household size (hh_size), the number of working adults (adults_working),
the number of children in the household (children), the housing type (htype)
in I/O-format (e.g.: 1, if individual house, 0, if multi-apartment house),
the quality of public transport in the residence area (1: Very Bad, 2: Bad, 3: Good, 4: Very Good),
whether the household holds a carsharing-membership (sharing), the
ratio of the average car price divided by household income (relative_cost_per_car).
Average market prices can be derived from Kraus' vehicle cost model.
Last input parameter is the average age of the adults, living in the household,
scaled by *0.1!