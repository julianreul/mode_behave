Model Purpose and General Information
=====================================
This package is a framework for the estimation of multinomial logit and 
mixed logit models with a discrete mixing distribution with fixed points 
(see K. Train (2009): Discrete Choice Methods and Simulation).
A mixed logit model is a multinomial logit model, in which the coefficients 
are distributed over a (discrete) parameter space. The discrete subsets
of the parameter space are called classes (compare latent class model).
The goal of the estimation procedure for mixed logit models is to estimate 
the optimal shares of the classes.
The model can be calibrated on any dataset, following a certain structure.

The benefit of this package is, that it enables access to estimation routines
for mixed logit models with discrete mixing distributions, leveraging 
computation of GPU-hardware.

The benefit of mixed logit models is, that they enable capturing
preference heterogeneities within the oberved choice data.

Installation
============
1. Download or clone the repository to a local folder.
#. Open (Anaconda) Prompt
#. Create a new environment from reference_environment.yml file (same location as setup.py file)::

      conda env create -f reference_environment.yml

#. cd to the directory, where you stored the repository and where the setup.py file is located
#. In this folder run::
    
      pip install -e .
      
#. If a NVIDIA-GPU is accessible, gpu-based estimation methods can be used. Therefore, please also install the python-package "cuda" and an appropriate NVIDIA-driver.


Workflow
========
1. Import model with::

      import mode_behave as mb

2. Initialize a model with::
    
      model = mb.Core(param = param_temp, max_space = 1000, data_name = 'data_in', alt=3)
      
   The structure of the input data and the parameter-input are given below.

3. Estimate the model with::

      model.estimate_mixed_logit(**kwargs)  
      
The estimation of the mixed logit model can be modified by definition of keyword-arguments
during instantiation and within the estimation-method itself.

| **Arguments for instantion** (ov.Core(...)):
| dict param: Indicated the names of the model attributes. The attribute-names 
|       shall be derived from the column names of the input data.
| int max_space: Defines the maximum number of parameter points within the 
|       considered parameter space.
| str data_name: Indicates the name of the input data-file. 
| int alt: Indicates the number of considered choice alternatives.
|
| **Keyword-arguments for Instantion** (ov.Core(...)): 
| int sample_data: Define subset of data for estimation of base MNL-model.
| tuple select_data: Defines a tuple of (str: attribute name, int/float: attribute value)
|     to estimate mixed logit model upon specific subset of data.
| str initial_point_name: Specify name of pickle-file within subfolder *ModelParam*,
|     if MNL-model was previously estimated (saves CPU-time).
| dict param_transform: Specify alternative definition of parameter structure,
|     if a previously estimated initial point (MNL-model) is loaded, but 
|     further estimation is carried out upon alternative param-structure. 
|     (e.g.: Parameters are set to random, which were fixed before.)
|
| **Keyword-arguments for estimation-method (model.estimate_mixed_logit(...))**:
| int min_inter: Min. iterations for EM-algorithm.
| int max_iter: Max. iterations for EM-algorithm
| bool gpu: If set to True, then the estimation process is conducted using GPU-hardware.
|     A necessary pre-condition for that is the existence of some NVIDIA-GPU
|     on your hardware ans the installation of the python module numba.
| bool bit_64: Defaults to False. If set to True, all numbers are calculated
|     in 64-bit format, which increases precision, but also runtime.
| str space_method: 'abs_value' or 'std_value', defines type of space-creation
| int blind_exploration: Number of iteration in which shares are not updated. 
|     Reduces initialization bias of random draws.
| int scale_space: Defines the size of the space, relative to the chosen space_method
| float PROBS_single_percent: Treshold between 0-1, when to neglect certain points. Default = 0.9
| float PROBS_min: Min. percentage to which the parameter space shall be analyzed.
| int draws_per_iter: Number of random draws per iteration of EM-algorithm.
| int updated_shares: Number of stochastically updated shares around the analyzed point.
|     If large, calculation speeds up exponentially, but results might diverge from optimum.
|
      
4. The package can also be used to estimate multinomial logit models::

      model.estimate_logit(**kwargs)  
      
| **Keyword-arguments for estimation-method (model.estimate_logit(...))**:
| bool stats: If set to True, t-statistics from the estimation process are evaluated.
|
   
Structure of Parameters and Input Data
======================================

1. Input data

   The input dataset contains the observations with which the model is 
   calibrated. The input data is called with the specified string of the
   keyword-argument *data_name*. The input data shall be placed within 
   the subfolder *InputData* within the package *mode_behave*.
   The data shall follow the structure below:
   
   * Rows: Observations.
   * Columns:
         - One column per parameter of the utility function AND per alternative.
           Specified as: 'Attribute_name_' + str(no_alternative)
         - One column for the choice-indication of each  alternative.
           Specified as: 'choice_' + str(no_alternative)
         - One column per alternative, indicating the availability.
           Specified as: 'av_' + str(no_alternative)
         - If a parameter is constant across alternatives, then let the columns be equal.
          
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
   
   * x[:(alt-1)] : ASC-constants for the alternatives 1-#of alternatives. ASC for choice option 0 defaults to 0.
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
are estimated on a discrete parameter space, which is specified by the researcher.
The discrete subsets of the parameter space are called classes, 
analogously to latent class models (LCM). The goal of the estimation procedure
is to estimate the optimal share of each class within the discrete parameter space.
The algorithm roughly follows the procedure below:

1. Estimate initial coefficients of a standard multinomial logit model.
2. Specify the discrete parameter space of the random coefficients with
   the mean and the standard deviation of each initially calculated random coefficient. 
   (The standard deviation can be calculated from a k-fold cross-validation.)
   Alternatively, the parameter space can be defined via the absolute values
   of the parameters. Let the number of classes, i.e. the granularity of the discrete parameter space,
   be determined by the maximum number of classes, specified during initialization.
3. Estimate the optimal share for each class in the discrete parameter space
   with an expectation-maximization (EM) algorithm. (see Train, 2009)
4. In order to speed up the estimation procedure and to handle memory-issues,
   three adaptations can be/are applied:
   
|   4.1 Batch-Estimation (Default):
|       The estimation procedure can be conducted in batches, not optimizing
|       the whole parameter-space at once, but exploring it incrementally
|       in batches. This method reduced memory allocation.
|   4.2 Assuming a proximity (Optional): 
|       If we additionally assume, that the estimated shares
|       for a single batch are itself normally distributed over classes in 
|       proximity to the estimated ones within the batch, estimation time
|       is reduced.
|   4.3 GPU-utilization (Optional): The optimization during the estimation process
|       can be performed on GPU-hardware, if available.
|
      

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

    model.forecast(
        choice_values = np.array([0,1,2,3]), **kwargs
                )
                
    "Choice values" indicated the numerical value of each choice option.
    In **kwargs, also "k_cluster" can be given to indicate the number of cluster
    centers which shall be analyzed. This method forecasts the mean choice, based
    on the estimated parameters of each cluster center and the attribute values
    of the base data. It is a good reference point to study the diverging choice
    behavior of each cluster center.      

          
Simulation
==========

The model incorporates a class **Simulation**, which contains customized
methods to simulate previously estimated choice models.
In order to simulate choice probabilities, the model must be instantiated as follows::

   model = ov.Core(model_type = 'simulation', dc_type = 'MNL')
   
The keyword-argument *dc_type* specifies the type of choice model. 
Currently only MNL-simulations are implemented.

The following MNL-simulations are currently available:

**MNL-Model for Mode-Choice**::

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

**MNL-model for the probability of the number of cars per households.**::

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
   



















