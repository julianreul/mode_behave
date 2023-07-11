"""
This module holds the class "Estimation", which incorporates functionality
to estimate the coefficients and class-shares of the specified mixed logit
model with a discrete mixing distribution with fixed points as well as 
the coefficients of a multinomial logit model.
"""

import time
import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)

import numpy as np
from numba import guvectorize, njit, prange
from scipy.optimize import minimize
from scipy import stats
from skopt.sampler import Lhs
from skopt.sampler import Halton
from skopt.space import Space
from math import exp, pow


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
        tol : float, optional
            Tolerance of the internal EM-algorithm.
        max_iter : int, optional
            Maximum iterations of the EM-algorithm.
        min_iter : int, optional
            Minimum iterations of the EM-algorithm.
        space_method : string, optional
            The method which shall be applied to span the parameter space
            around the initially estimated parameter points (from MNL-model).
            Options are "abs_value", "std_value" or "mirror". Defaults to "mirror".
        scale_space : float, optional
            Sets the size of the parameter space. Defaults to 2.
        bits_64 : Boolean, optional
            If True, numerical precision is increased to 64 bits, instead of 32 bits.
            Defaults to False.
        max_shares : int, optional
            Specifies the maximum number of points in the parameter space, for which
            a "share" shall be estimated. That does not mean, that only this number
            of points will be explored in the parameter space, but only for this
            number points a "share" is being stored. This is done to limit the
            memory of the estimation process. max_shares defaults to 1000.

        Returns
        -------
        points : array
            Numpy array, which holds all points of the discrete parameter space.
        shares : array
            The central output of this method is the array "self.shares", which
            holds the estimated shares of points within the parameter space.
        """

        # get estimation parameters.
        tol = kwargs.get("tol", 0.01)
        max_iter = kwargs.get("max_iter", 1000)
        min_iter = kwargs.get("min_iter", 10)
        scale_space = kwargs.get("scale_space", 2)
        space_method = kwargs.get("space_method", "mirror")
        t_stats_out = kwargs.get("t_stats_out", True)
        self.bits_64 = kwargs.get("bits_64", False)
        quiet = kwargs.get("quiet", True)

        # treshold for dropping a point: percentage of initial value in 'SHARES'
        max_shares = kwargs.get("max_shares", 1000)

        self.no_constant_fixed = len(self.param["constant"]["fixed"])
        self.no_constant_random = len(self.param["constant"]["random"])
        self.no_variable_fixed = len(self.param["variable"]["fixed"])
        self.no_variable_random = len(self.param["variable"]["random"])

        # get the maximum number of equally-spaces coefficients per alternative.
        # points per coefficient (ppc)
        no_random = self.no_constant_random + self.no_variable_random * self.count_c

        # Define space-boundaries from initial point.
        if space_method == "abs_value":
            try:
                # define absolute value of parameter as offset
                offset_values = np.array([abs(temp) for temp in self.initial_point])
            except AttributeError:
                print("Estimate initial coefficients.")
                if t_stats_out:
                    self.initial_point = self.estimate_logit()
                else:
                    self.initial_point = self.estimate_logit(stats=False)
                # define absolute value of parameter as offset
                offset_values = np.array([abs(temp) for temp in self.initial_point])
        elif space_method == "std_value":
            try:
                # define std of parameter as offset
                offset_values = self.std_cross_val
            except AttributeError:
                print("Estimate initial coefficients.")
                self.initial_point = self.estimate_logit()
                # define std of parameter as offset
                offset_values = self.std_cross_val

        elif space_method == "mirror":
            try:
                # define std of parameter as offset
                offset_values = self.std_cross_val
            except AttributeError:
                print("Estimate initial coefficients.")
                self.initial_point = self.estimate_logit()
                # define std of parameter as offset
                offset_values = self.std_cross_val
        elif space_method == "uniform":
            print("Estimate initial coefficients.")
            self.initial_point = self.estimate_logit()
        else:
            raise ValueError("Unknown value for keyword-argument -space_method-")

        # specify parameter space
        if self.bits_64:
            self.space_bounds = np.zeros((no_random, 2), "float64")
        else:
            self.space_bounds = np.zeros((no_random, 2), "float32")
        for a in range(self.no_constant_random):
            if space_method == "uniform":
                upper_bound = 1
                lower_bound = -1
            else:
                mean_coefficient = self.initial_point[
                    self.count_c - 1 + self.no_constant_fixed + a
                ]
                offset = offset_values[self.count_c - 1 + self.no_constant_fixed + a]
                if space_method == "mirror":
                    if mean_coefficient > 0:
                        upper_bound = mean_coefficient + offset * scale_space
                        lower_bound = min(
                            -mean_coefficient, mean_coefficient - offset * scale_space
                        )
                    else:
                        upper_bound = max(
                            -mean_coefficient, mean_coefficient + offset * scale_space
                        )
                        lower_bound = mean_coefficient - offset * scale_space
                else:
                    lower_bound = mean_coefficient - offset * scale_space
                    upper_bound = mean_coefficient + offset * scale_space

            if lower_bound == upper_bound:
                lower_bound = lower_bound - 0.01
                upper_bound = upper_bound + 0.01

            self.space_bounds[a][0] = lower_bound
            self.space_bounds[a][1] = upper_bound

        for c in range(self.count_c):
            for a in range(self.no_variable_random):
                if space_method == "uniform":
                    upper_bound = 1
                    lower_bound = -1
                else:
                    mean_coefficient = self.initial_point[
                        self.count_c
                        - 1
                        + self.no_constant_fixed
                        + self.no_constant_random
                        + (self.no_variable_fixed + self.no_variable_random) * c
                        + self.no_variable_fixed
                        + a
                    ]
                    offset = offset_values[
                        self.count_c
                        - 1
                        + self.no_constant_fixed
                        + self.no_constant_random
                        + (self.no_variable_fixed + self.no_variable_random) * c
                        + self.no_variable_fixed
                        + a
                    ]

                    if space_method == "mirror":
                        if mean_coefficient > 0:
                            upper_bound = mean_coefficient + offset * scale_space
                            lower_bound = min(
                                -mean_coefficient,
                                mean_coefficient - offset * scale_space,
                            )
                        else:
                            upper_bound = max(
                                -mean_coefficient,
                                mean_coefficient + offset * scale_space,
                            )
                            lower_bound = mean_coefficient - offset * scale_space
                    else:
                        lower_bound = mean_coefficient - offset * scale_space
                        upper_bound = mean_coefficient + offset * scale_space

                if lower_bound == upper_bound:
                    lower_bound = lower_bound - 0.01
                    upper_bound = upper_bound + 0.01

                self.space_bounds[
                    self.no_constant_random + self.no_variable_random * c + a
                ][0] = lower_bound
                self.space_bounds[
                    self.no_constant_random + self.no_variable_random * c + a
                ][1] = upper_bound

        self.space_lhs = Space(self.space_bounds)
        # lhs = Halton()
        lhs = Lhs(lhs_type="classic", criterion="correlation", iterations=10)

        # prepare input for numba
        initial_point = self.initial_point
        count_c = self.count_c
        count_e = self.count_e
        no_constant_fixed = self.no_constant_fixed
        no_constant_random = self.no_constant_random
        no_variable_fixed = self.no_variable_fixed
        no_variable_random = self.no_variable_random
        av = self.av
        weight = self.weight_vector
        choice = self.choice

        # maximum number of aggregated alternatives per segment
        dim_aggr_alt_max = max(
            len(self.param["constant"]["fixed"]),
            len(self.param["constant"]["random"]),
            len(self.param["variable"]["fixed"]),
            len(self.param["variable"]["random"]),
        )

        data = np.zeros(
            (4, dim_aggr_alt_max, self.count_c, self.av.shape[1], len(self.data))
        )
        for c in range(self.count_c):
            for e in range(self.count_e):
                for i, param in enumerate(self.param["constant"]["fixed"]):
                    data[0][i][c][e] = self.data[
                        param + "_" + str(c) + "_" + str(e)
                    ].values
                for i, param in enumerate(self.param["constant"]["random"]):
                    data[1][i][c][e] = self.data[
                        param + "_" + str(c) + "_" + str(e)
                    ].values
                for i, param in enumerate(self.param["variable"]["fixed"]):
                    data[2][i][c][e] = self.data[
                        param + "_" + str(c) + "_" + str(e)
                    ].values
                for i, param in enumerate(self.param["variable"]["random"]):
                    data[3][i][c][e] = self.data[
                        param + "_" + str(c) + "_" + str(e)
                    ].values

        if self.bits_64 == False:
            data = data.astype("float32")
            av = av.astype("float32")
            weight = weight.astype("float32")
            choice = choice.astype("int32")

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
                Point in base data.
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
                res_temp = initial_point[c - 1]

            for a in range(no_constant_fixed):
                res_temp += initial_point[(count_c - 1) + a] * data[0][a][c][e][l]
            for a in range(no_constant_random):
                res_temp += point[a] * data[1][a][c][e][l]
            for a in range(no_variable_fixed):
                res_temp += (
                    initial_point[
                        (count_c - 1)
                        + no_constant_fixed
                        + no_constant_random
                        + (no_variable_fixed + no_variable_random) * c
                        + a
                    ]
                    * data[2][a][c][e][l]
                )
            for a in range(no_variable_random):
                res_temp += (
                    point[no_constant_random + no_variable_random * c + a]
                    * data[3][a][c][e][l]
                )

            return res_temp

        if self.bits_64:

            @guvectorize(
                [
                    "float64[:, :], float64[:, :, :], float64[:], float64[:, :, :, :, :], float64[:, :]"
                ],
                "(m,p),(n,e,l),(l),(i,j,n,e,l)->(m,l)",
                nopython=True,
                target="parallel",
            )
            def calculate_logit_vector(points, av, weight, data, logit_probs_):
                """
                This method calculates the multinomial logit probability for a given
                set of coefficients and all choices in the sample of the dataset.

                Parameters
                ----------
                point : array
                    Point in the parameter space.
                av : array
                    Availability array for all choice options.
                weight : array
                    Weights of data points in base data.
                data : array
                    Base data.

                Returns
                -------
                Array with logit-probabilities.

                """

                for m in prange(points.shape[0]):
                    point = points[m]

                    # iterate over length of data array (len(av))
                    # What is the shape of the output array?!
                    for l in prange(av.shape[2]):
                        # calculate bottom
                        bottom = 0
                        top = 0
                        for c in prange(count_c):
                            for e in prange(count_e):
                                exp_temp = exp(get_utility_vector(c, e, point, l, data))
                                bottom += av[c][e][l] * exp_temp
                                top += av[c][e][l] * choice[c][e][l] * exp_temp
                        res_temp = top / bottom
                        if np.isfinite(res_temp):
                            logit_probs_[m][l] = pow(res_temp, weight[l])
                        else:
                            logit_probs_[m][l] = 0

        else:

            @guvectorize(
                [
                    "float32[:, :], float32[:, :, :], float32[:], float32[:, :, :, :, :], float32[:, :]"
                ],
                "(m,p),(n,e,l),(l),(i,j,n,e,l)->(m,l)",
                nopython=True,
                target="parallel",
            )
            def calculate_logit_vector(points, av, weight, data, logit_probs_):
                """
                This method calculates the multinomial logit probability for a given
                set of coefficients and all choices in the sample of the dataset.

                Parameters
                ----------
                point : array
                    Point in the parameter space.
                av : array
                    Availability array for all choice options.
                weight : array
                    Weights of data points in base data.
                data : array
                    Base data.

                Returns
                -------
                Array with logit-probabilities.

                """

                for m in prange(points.shape[0]):
                    point = points[m]

                    # iterate over length of data array (len(av))
                    # What is the shape of the output array?!
                    for l in prange(av.shape[2]):
                        # calculate bottom
                        bottom = 0
                        top = 0
                        for c in prange(count_c):
                            for e in prange(count_e):
                                exp_temp = exp(get_utility_vector(c, e, point, l, data))
                                bottom += av[c][e][l] * exp_temp
                                top += av[c][e][l] * choice[c][e][l] * exp_temp
                        res_temp = top / bottom
                        if np.isfinite(res_temp):
                            logit_probs_[m][l] = pow(res_temp, weight[l])
                        else:
                            logit_probs_[m][l] = 0

        def get_expectation(shares, logit_probs):
            # get point_probs
            point_probs_T = logit_probs.T * shares
            point_probs = point_probs_T.T

            # get h
            h = point_probs / point_probs.sum(axis=0)

            # update shares
            shares_update = h.sum(axis=1) / h.sum()

            # get expectation
            expect = (h.T * np.log(shares_update)).sum()

            return expect, shares_update

        print("Estimate shares.")

        start = time.time()

        print("____EM algorithm starts.")

        # Step 2: Draw points for EM-optimization via latin hypercube sampling.
        GRID = lhs.generate(self.space_lhs.dimensions, max_shares, random_state=6)

        if self.bits_64:
            GRID_np = np.array(GRID, dtype="float64")
        else:
            GRID_np = np.array(GRID, dtype="float32")

        # Step 3: Initialize the shares, which are associated with GRID
        drawn_shares = np.array([1 / max_shares] * max_shares)

        # normalize, due to numerical reasons.
        drawn_shares = drawn_shares / np.sum(drawn_shares)

        if self.bits_64:
            drawn_shares = drawn_shares.astype("float64")
        else:
            drawn_shares = drawn_shares.astype("float32")

        # Step 4: calculate logit probabilities for each point.
        drawn_logit_probs = calculate_logit_vector(GRID_np, av, weight, data)

        self.check_drawn_logit_probs = drawn_logit_probs.copy()

        # Step 5: Apply EM-algorithm to drawn indices
        convergence = 0
        iter_inner = 0
        expect_before = 0

        while convergence == 0:
            # calculate probability, that a person has the coefficients of a
            # specific point, given his/her choice: point_probs = h_nc
            expect, drawn_shares = get_expectation(drawn_shares, drawn_logit_probs)

            self.shares_after_update = drawn_shares.copy()

            diff = abs(expect - expect_before)
            if quiet == False:
                print("____ITER INNER:", iter_inner)
                print("________DIFF:", diff)
                print("________EXPECT:", expect)
            expect_before = expect
            iter_inner += 1
            if diff < tol and iter_inner > min_iter:
                convergence = 1
                break
            if iter_inner == max_iter:
                break

        # Step 6: Assign result of EM-optimization (drawn_shares) to SHARES.
        SHARES = drawn_shares.copy()
        if self.bits_64:
            SHARES = SHARES.astype("float64")
        else:
            SHARES = SHARES.astype("float32")
        # normalize, although SHARES should sum to one already.
        SHARES = SHARES / np.sum(SHARES)

        end = time.time()
        delta = end - start
        print("____EM algorithm took: ", str(delta), "seconds.")

        if np.sum(np.isnan(SHARES)):
            raise ValueError(
                "NaN-values detected in -shares-. Debug hint: You may adjust the parameter space (smaller)."
            )
        else:
            self.shares = SHARES
            self.points = GRID

    def estimate_logit(self, **kwargs):
        """
        This method estimates the coefficients of a standard MNL model.

        Parameters
        ----------
        stats : Boolean, optional
            If True, summary statistics are returned as well. Defaults to True.

        Returns
        -------
        list
            List of estimated coefficients of standard MNL model.

        """

        stats_sum = kwargs.get("stats", True)

        def loglike(x):

            # logged numerator of MNL model
            utility_single = sum(
                [
                    self.av[0][e]
                    * self.choice[0][e]
                    * (
                        sum(
                            [
                                (
                                    x[(self.count_c - 1) + a]
                                    * self.data[
                                        self.param["constant"]["fixed"][a]
                                        + "_"
                                        + str(0)
                                        + "_"
                                        + str(e)
                                    ]
                                )
                                for a in range(self.no_constant_fixed)
                            ]
                        )
                        + sum(
                            [
                                (
                                    x[(self.count_c - 1) + self.no_constant_fixed + a]
                                    * self.data[
                                        self.param["constant"]["random"][a]
                                        + "_"
                                        + str(0)
                                        + "_"
                                        + str(e)
                                    ]
                                )
                                for a in range(self.no_constant_random)
                            ]
                        )
                        + sum(
                            [
                                (
                                    x[
                                        (self.count_c - 1)
                                        + self.no_constant_fixed
                                        + self.no_constant_random
                                        + a
                                    ]
                                    * self.data[
                                        self.param["variable"]["fixed"][a]
                                        + "_"
                                        + str(0)
                                        + "_"
                                        + str(e)
                                    ]
                                )
                                for a in range(self.no_variable_fixed)
                            ]
                        )
                        + sum(
                            [
                                (
                                    x[
                                        (self.count_c - 1)
                                        + self.no_constant_fixed
                                        + self.no_constant_random
                                        + self.no_variable_fixed
                                        + a
                                    ]
                                    * self.data[
                                        self.param["variable"]["random"][a]
                                        + "_"
                                        + str(0)
                                        + "_"
                                        + str(e)
                                    ]
                                )
                                for a in range(self.no_variable_random)
                            ]
                        )
                    )
                    for e in range(self.av.shape[1])
                ]
            ) + sum(
                [
                    sum(
                        [
                            self.av[c][e]
                            * self.choice[c][e]
                            * (
                                x[c - 1]
                                + sum(
                                    [
                                        (
                                            x[(self.count_c - 1) + a]
                                            * self.data[
                                                self.param["constant"]["fixed"][a]
                                                + "_"
                                                + str(c)
                                                + "_"
                                                + str(e)
                                            ]
                                        )
                                        for a in range(self.no_constant_fixed)
                                    ]
                                )
                                + sum(
                                    [
                                        (
                                            x[
                                                (self.count_c - 1)
                                                + self.no_constant_fixed
                                                + a
                                            ]
                                            * self.data[
                                                self.param["constant"]["random"][a]
                                                + "_"
                                                + str(c)
                                                + "_"
                                                + str(e)
                                            ]
                                        )
                                        for a in range(self.no_constant_random)
                                    ]
                                )
                                + sum(
                                    [
                                        (
                                            x[
                                                (self.count_c - 1)
                                                + self.no_constant_fixed
                                                + self.no_constant_random
                                                + (
                                                    self.no_variable_fixed
                                                    + self.no_variable_random
                                                )
                                                * c
                                                + a
                                            ]
                                            * self.data[
                                                self.param["variable"]["fixed"][a]
                                                + "_"
                                                + str(c)
                                                + "_"
                                                + str(e)
                                            ]
                                        )
                                        for a in range(self.no_variable_fixed)
                                    ]
                                )
                                + sum(
                                    [
                                        (
                                            x[
                                                (self.count_c - 1)
                                                + self.no_constant_fixed
                                                + self.no_constant_random
                                                + (
                                                    self.no_variable_fixed
                                                    + self.no_variable_random
                                                )
                                                * c
                                                + self.no_variable_fixed
                                                + a
                                            ]
                                            * self.data[
                                                self.param["variable"]["random"][a]
                                                + "_"
                                                + str(c)
                                                + "_"
                                                + str(e)
                                            ]
                                        )
                                        for a in range(self.no_variable_random)
                                    ]
                                )
                            )
                            for e in range(self.av.shape[1])
                        ]
                    )
                    for c in range(1, self.count_c)
                ]
            )

            # logged denominator of MNL model
            utility_all = sum(
                [
                    self.av[0][e]
                    * (
                        np.exp(
                            sum(
                                [
                                    (
                                        x[(self.count_c - 1) + a]
                                        * self.data[
                                            self.param["constant"]["fixed"][a]
                                            + "_"
                                            + str(0)
                                            + "_"
                                            + str(e)
                                        ]
                                    )
                                    for a in range(self.no_constant_fixed)
                                ]
                            )
                            + sum(
                                [
                                    (
                                        x[
                                            (self.count_c - 1)
                                            + self.no_constant_fixed
                                            + a
                                        ]
                                        * self.data[
                                            self.param["constant"]["random"][a]
                                            + "_"
                                            + str(0)
                                            + "_"
                                            + str(e)
                                        ]
                                    )
                                    for a in range(self.no_constant_random)
                                ]
                            )
                            + sum(
                                [
                                    (
                                        x[
                                            (self.count_c - 1)
                                            + self.no_constant_fixed
                                            + self.no_constant_random
                                            + a
                                        ]
                                        * self.data[
                                            self.param["variable"]["fixed"][a]
                                            + "_"
                                            + str(0)
                                            + "_"
                                            + str(e)
                                        ]
                                    )
                                    for a in range(self.no_variable_fixed)
                                ]
                            )
                            + sum(
                                [
                                    (
                                        x[
                                            (self.count_c - 1)
                                            + self.no_constant_fixed
                                            + self.no_constant_random
                                            + self.no_variable_fixed
                                            + a
                                        ]
                                        * self.data[
                                            self.param["variable"]["random"][a]
                                            + "_"
                                            + str(0)
                                            + "_"
                                            + str(e)
                                        ]
                                    )
                                    for a in range(self.no_variable_random)
                                ]
                            )
                        )
                    )
                    for e in range(self.av.shape[1])
                ]
            ) + sum(
                [
                    sum(
                        [
                            self.av[c][e]
                            * (
                                np.exp(
                                    x[c - 1]
                                    + sum(
                                        [
                                            (
                                                x[(self.count_c - 1) + a]
                                                * self.data[
                                                    self.param["constant"]["fixed"][a]
                                                    + "_"
                                                    + str(c)
                                                    + "_"
                                                    + str(e)
                                                ]
                                            )
                                            for a in range(self.no_constant_fixed)
                                        ]
                                    )
                                    + sum(
                                        [
                                            (
                                                x[
                                                    (self.count_c - 1)
                                                    + self.no_constant_fixed
                                                    + a
                                                ]
                                                * self.data[
                                                    self.param["constant"]["random"][a]
                                                    + "_"
                                                    + str(c)
                                                    + "_"
                                                    + str(e)
                                                ]
                                            )
                                            for a in range(self.no_constant_random)
                                        ]
                                    )
                                    + sum(
                                        [
                                            (
                                                x[
                                                    (self.count_c - 1)
                                                    + self.no_constant_fixed
                                                    + self.no_constant_random
                                                    + (
                                                        self.no_variable_fixed
                                                        + self.no_variable_random
                                                    )
                                                    * c
                                                    + a
                                                ]
                                                * self.data[
                                                    self.param["variable"]["fixed"][a]
                                                    + "_"
                                                    + str(c)
                                                    + "_"
                                                    + str(e)
                                                ]
                                            )
                                            for a in range(self.no_variable_fixed)
                                        ]
                                    )
                                    + sum(
                                        [
                                            (
                                                x[
                                                    (self.count_c - 1)
                                                    + self.no_constant_fixed
                                                    + self.no_constant_random
                                                    + (
                                                        self.no_variable_fixed
                                                        + self.no_variable_random
                                                    )
                                                    * c
                                                    + self.no_variable_fixed
                                                    + a
                                                ]
                                                * self.data[
                                                    self.param["variable"]["random"][a]
                                                    + "_"
                                                    + str(c)
                                                    + "_"
                                                    + str(e)
                                                ]
                                            )
                                            for a in range(self.no_variable_random)
                                        ]
                                    )
                                )
                            )
                            for e in range(self.av.shape[1])
                        ]
                    )
                    for c in range(1, self.count_c)
                ]
            )

            self.check_utility_all = utility_all

            # logged probability of MNL model
            log_prob = self.weight_vector * (utility_single - np.log(utility_all))

            res = -np.sum(log_prob)

            return res

        # initialize optimization of MNL coefficients
        no_param = (
            self.no_constant_fixed
            + self.no_constant_random
            + self.count_c * (self.no_variable_fixed + self.no_variable_random)
        )
        no_coeff = int(self.count_c - 1 + no_param)
        x0 = np.zeros((no_coeff,), dtype=float)

        # optimization of objective function: Nelder-Mead, L-BFGS-B
        start_logit = time.time()
        res = minimize(loglike, x0, method="L-BFGS-B", tol=1e-11, jac="cs")
        end_logit = time.time()
        delta_logit = end_logit - start_logit
        print("Estimation of standard logit took [sec.]:", int(delta_logit))

        res_param = res.x

        print(res_param)

        if stats_sum:
            print("Calculation of summary statistics starts.")
            start_stats = time.time()
            data_safe = self.data
            size_subset = int(len(self.data) / 10)
            param_cross_val = {j: [] for j in range(no_coeff)}
            for i in range(10):
                print("Cross-validation run: ", str(i))
                self.data = data_safe[size_subset * i : size_subset * (i + 1)]
                if "weight" in self.data.columns and self.include_weights == True:
                    self.weight_vector = self.data["weight"].values.copy()
                else:
                    self.weight_vector = np.ones(shape=len(self.data))
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

                res = minimize(loglike, x0, method="L-BFGS-B", tol=1e-11, jac="cs")
                # iterate over estimated coefficients
                for j, param in enumerate(res.x):
                    param_cross_val[j].append(param)

            end_stats = time.time()
            delta_stats = end_stats - start_stats
            print("Estimation of summary statistics took [sec.]:", int(delta_stats))

            self.check_param_cross_val = param_cross_val

            # calculate statistics
            self.t_stats = [
                stats.ttest_1samp(param_cross_val[j], 0) for j in range(no_coeff)
            ]
            self.std_cross_val = [np.std(param_cross_val[j]) for j in range(no_coeff)]

            # reset self.data, self.av, self.choice
            self.data = data_safe
            # define choices and availabilities
            if "weight" in self.data.columns and self.include_weights == True:
                self.weight_vector = self.data["weight"].values.copy()
            else:
                self.weight_vector = np.ones(shape=len(self.data))
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

        print("LL_0: ", loglike(x0))
        print("LL_final: ", loglike(res_param))

        return res_param
