"""
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

    def simulate_hh_cars(
        self,
        regiontype,
        hh_size,
        adults_working,
        children,
        htype,
        quali_opnv,
        sharing,
        relative_cost_per_car,
        age_adults,
        **kwargs
    ):
        """
        This method calculates the probability of a single household
        to own 0,1,2,3 or more cars.
        Parameters
        ----------
        regiontype: int
            RegioStaR7-classification of regions (1-7), where 1-4 are urban 
            classifications and 5-7 rural classifications. 
        hh_size: int
            Household size
        adults_working: int 
            Number of working adults
        children: int
            Number of children in household
        htype: int
            1, if individual house, 0, if multi-apartment house
        quali_opnv: int
            Quality of public transport on a scale from 1-4 
            (1: Very Bad, 2: Bad, 3: Good, 4: Very Good)
        sharing: float
            Carsharing membership (1: Yes, 0: no)
        relative_cost_per_car: 
            Average Price of considered cars / net household income
        age_adults: float
            Mean age of adults in the household scaled by 0.1!

        Returns
        -------
        res_temp : array
            Probabilities, that a household owns 0,1,2, or 3 cars.

        """

        # get keyword-arguments
        asc_offset_hh_cars = kwargs.get("asc_offset", self.asc_offset_hh_cars)
        asc_offset = np.array(
            [
                asc_offset_hh_cars["rt_" + str(regiontype)]["offset_" + str(c)]
                for c in range(4)
            ]
        )

        # define model properties below.
        count_c = 4  # number of alternatives: 0-3
        all_alternatives = np.array((0, 1, 2, 3))
        no_constant_fixed = 0
        no_constant_random = 0
        no_variable_fixed = 10
        no_variable_random = 1
        # specify maximum number of alternatives
        dim_aggr_alt_max = max(
            no_constant_fixed,
            no_constant_random,
            no_variable_fixed,
            no_variable_random,
        )

        # Derive urban / rural regiontype according to RegioStaR2-scale from regiontype
        if regiontype in [1, 2, 3, 4]:
            urban_region = 1
            rural_region = 0
        elif regiontype in [5, 6, 7]:
            urban_region = 0
            rural_region = 1
        else:
            raise ValueError("Regiontype must be a value between 1-7.")

        # Define hh_data.
        # IMPORTANT: The order of parameters (see hh_data) must be equal to the order during
        # estimation (see param), as defined in param = {...} !!!
        hh_data = np.zeros((4, dim_aggr_alt_max, count_c), dtype="float64")
        # fill parameters: variable_fixed
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

        # fill parameters: variable_random
        for i in range(count_c):
            hh_data[3][0][i] = relative_cost_per_car * i

        @njit
        def get_utility_fast_MNL_cars(c, data, initial_point, asc_offset):
            if c != 0:
                res_temp = asc_offset[c] + initial_point[c - 1]
            else:
                res_temp = asc_offset[c]
            for a in range(no_constant_fixed):
                res_temp = res_temp + initial_point[(count_c - 1) + a] * data[0][a][c]
            for a in range(no_constant_random):
                res_temp = (
                    res_temp
                    + initial_point[(count_c - 1) + no_constant_fixed + a]
                    * data[1][a][c]
                )
            for a in range(no_variable_fixed):
                res_temp = (
                    res_temp
                    + initial_point[
                        (count_c - 1)
                        + no_constant_fixed
                        + +(no_variable_fixed + no_variable_random) * c
                        + a
                    ]
                    * data[2][a][c]
                )
            for a in range(no_variable_random):
                res_temp = (
                    res_temp
                    + initial_point[
                        (count_c - 1)
                        + no_constant_fixed
                        + no_constant_random
                        + (no_variable_fixed + no_variable_random) * c
                        + no_variable_fixed
                        + a
                    ]
                    * data[3][a][c]
                )
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
            # calculate top
            top = np.exp(
                get_utility_fast_MNL_cars(alternative, data, initial_point, asc_offset)
            )
            # calculate bottom
            bottom = 0
            for c in range(count_c):
                bottom += np.exp(
                    get_utility_fast_MNL_cars(c, data, initial_point, asc_offset)
                )

            return top / bottom

        # multinomial logit
        probs = []
        for i in all_alternatives:
            prob_temp = calculate_logit_fast_MNL_cars(
                i, hh_data, self.initial_point, asc_offset
            )
            probs.append(prob_temp)

        return probs

    def func(self, x, scale):
        """
        This method is a supportive function for the method get_speed().

        Parameters
        ----------
        x : float
            Input value to be scaled.
        scale : float
            scaling factor.

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        return np.log(x + 1) * scale

    # derivation of mode specific speed: UNIT = [km/h]
    def get_speed(self, mode, distance, regiontype):
        """
        This method calculates the average speed for a given transport mode,
        travel distance and travel regiontype.

        Parameters
        ----------
        mode : int
            Transport mode.
        distance : int
            Travel distance.
        regiontype : int
            RegioStaR7-classification of regions (1-7), where 1-4 are urban 
            classifications and 5-7 rural classifications. 

        Returns
        -------
        speed : float
            Average speed.

        """
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

        speed = self.func(distance, scale)

        return speed

    def get_travel_duration_single(self, mode, distance, regiontype):
        """
        This method calculates the duration of travel.

        Parameters
        ----------
        mode : int
            Transport mode.
        distance : int
            Travel distance.
        regiontype : int
            RegioStaR7-classification of regions (1-7), where 1-4 are urban 
            classifications and 5-7 rural classifications. 

        Returns
        -------
        float
            Travel duration.

        """
        # self.check_distance = distance
        if distance == 0:
            return 0
        else:
            return distance / (self.get_speed(mode, distance, regiontype) / 60)

    def get_travel_cost(self, distance, mode, regiontype):
        """
        This method calculated the travel costs.

        Parameters
        ----------
        mode : int
            Transport mode.
        distance : int
            Travel distance.
        regiontype : int
            RegioStaR7-classification of regions (1-7), where 1-4 are urban 
            classifications and 5-7 rural classifications. 

        Returns
        -------
        cost_temp : float
            Travel costs in €.

        """
        if mode == 10:
            cost_temp = (
                self.get_travel_duration_single(mode, distance, regiontype)
                * self.cc_cost
            )
        else:
            cost_temp = self.dict_specific_travel_cost[mode] * distance

        return cost_temp

    def simulate_mode_choice(
        self, agegroup, occupation, regiontype, distance, av, **kwargs
    ):
        """
        This method calculates the probability, that a transport mode is chosen.
        
        Parameters
        ----------
        agegroup : int
            Agegroup of the agent (1: <18, 2: 18-65, 3: >65)
        occupation : int
            Occupation type of the agent (1: full-time work, 2: part-time, 3: education, 4: no occupation)
        regiontype : int
            RegioStaR7-classification of regions (1-7), where 1-4 are urban 
            classifications and 5-7 rural classifications. 
        distance : int
            Distance traveled in km.
        av : array
            The availability of each mode in numpy array format, indicated
            by 1 (available) or 0 (not available).
        
        Returns
        -------
        probs : array
            Probabilities, that a transport mode is chosen.

        """

        # define model properties below.
        count_c = 10  # number of alternatives: 0-9
        all_alternatives = np.arange(10)
        no_constant_fixed = 0
        no_constant_random = 0
        no_variable_fixed = 10
        no_variable_random = 1
        # specify maximum number of alternatives
        dim_aggr_alt_max = 10  # no_variable_fixed

        # Define trip_data.
        # IMPORTANT: The order of parameters must be equal to the order during
        # estimation, as defined in param = {...} !!!
        trip_data = np.zeros((4, dim_aggr_alt_max, count_c))
        # fill parameters: variable_fixed
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

        if regiontype in (1, 2, 3, 4):
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
            mode = i + 1
            trip_data[2][9][i] = self.get_travel_cost(distance, mode, regiontype)
        # fill parameters: variable_random
        for i in range(count_c):
            mode = i + 1
            trip_data[3][0][i] = self.get_travel_duration_single(
                mode, distance, regiontype
            )

        @njit
        def get_utility_fast_MNL_mode(c, data, initial_point):
            """
            This method calculates the utility of a choice option.

            Parameters
            ----------
            c : int
                choice option.
            data : array
                Array, containing the base data.
            initial_point : array
                Array, containing the estimated model parameters.

            Returns
            -------
            res_temp : float
                Utility of choice option.

            """
            if c == 0:
                res_temp = initial_point[c - 1]
            else:
                res_temp = 0
            for a in range(no_constant_fixed):
                res_temp = res_temp + initial_point[(count_c - 1) + a] * data[0][a][c]
            for a in range(no_constant_random):
                res_temp = (
                    res_temp
                    + initial_point[(count_c - 1) + no_constant_fixed + a]
                    * data[1][a][c]
                )
            for a in range(no_variable_fixed):
                res_temp = (
                    res_temp
                    + initial_point[
                        (count_c - 1)
                        + no_constant_fixed
                        + no_constant_random
                        + (no_variable_fixed + no_variable_random) * c
                        + a
                    ]
                    * data[2][a][c]
                )
            for a in range(no_variable_random):
                res_temp = (
                    res_temp
                    + initial_point[
                        (count_c - 1)
                        + no_constant_fixed
                        + no_constant_random
                        + (no_variable_fixed + no_variable_random) * c
                        + no_variable_fixed
                        + a
                    ]
                    * data[3][a][c]
                )
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
            # calculate top
            top = av[alternative] * np.exp(
                get_utility_fast_MNL_mode(alternative, data, initial_point)
            )

            # calculate bottom
            bottom = 0
            for c in range(count_c):
                bottom += av[c] * np.exp(
                    get_utility_fast_MNL_mode(c, data, initial_point)
                )

            return top / bottom

        probs = []
        for i in all_alternatives:
            prob_temp = calculate_logit_fast_MNL_mode(
                i, trip_data, self.initial_point, av
            )
            probs.append(prob_temp)

        return probs

    def simulate_hh_car_type_germany(
        self,
        AV,
        SEGMENT,
        RELATIVE_B_KOSTEN_MAL100,
        REICHWEITE_DURCH100,
        LADE_TANK_ZEIT,
        DISTANZ_LADE_TANK,
        CO2_MAL10,
        MEAN_HH_AGE,
        HH_SIZE,
        POPULATION,
        CAR_PARK_EV,
        CARSHARING,
        LONG_RANGE_AV,
        EV_EXPERIENCE,
        OWN_POWER,
        HH_OCC,
        PT_QUALITY,
        RELATIVER_KAUFPREIS,
        **kwargs
    ):
        """
        Please contact the package maintainers via GitHub to receive the 
        required choice estimates of the MNL-model and further simulation instructions.

        """

        ASC_OFFSET = kwargs.get("ASC_OFFSET", [0, 0, 0, 0])

        count_c = 4  # number of alternatives: 0-3
        all_alternatives = np.array((0, 1, 2, 3))

        no_constant_fixed = 0
        no_constant_random = 0
        no_variable_fixed = 21
        no_variable_random = 1
        # specify maximum number of alternatives
        dim_aggr_alt_max = max(
            no_constant_fixed,
            no_constant_random,
            no_variable_fixed,
            no_variable_random,
        )

        # Define hh_data.
        # IMPORTANT: The order of parameters (see hh_data) must be equal to the order during
        # estimation (see param), as defined in param = {...} !!!
        hh_data = np.zeros((4, dim_aggr_alt_max, count_c), dtype="float64")

        map_segment = {
            "Kleinwagen": 0,
            "Kompaktklasse": 1,
            "Mittelklasse": 2,
            "Oberklasse": 3,
            "SUV": 4,
            "Van": 5,
        }

        # fill parameters: variable_fixed
        for i in range(count_c):
            # SEGMENT
            segment_temp = map_segment[SEGMENT[i]]
            hh_data[2][segment_temp][i] = 1
            all_seg = [0, 1, 2, 3, 4, 5]
            all_seg.remove(segment_temp)
            for seg in all_seg:
                hh_data[2][seg][i] = 0

            # OTHER ATTRIBUTES
            hh_data[2][6][i] = RELATIVER_KAUFPREIS[i]
            hh_data[2][7][i] = RELATIVE_B_KOSTEN_MAL100[i]
            hh_data[2][8][i] = LADE_TANK_ZEIT[i]
            hh_data[2][9][i] = DISTANZ_LADE_TANK[i]
            hh_data[2][10][i] = CO2_MAL10[i]
            hh_data[2][11][i] = MEAN_HH_AGE[i]
            hh_data[2][12][i] = HH_SIZE[i]
            hh_data[2][13][i] = POPULATION[i]
            hh_data[2][14][i] = CAR_PARK_EV[i]
            hh_data[2][15][i] = CARSHARING[i]
            hh_data[2][16][i] = LONG_RANGE_AV[i]
            hh_data[2][17][i] = EV_EXPERIENCE[i]
            hh_data[2][18][i] = OWN_POWER[i]
            hh_data[2][19][i] = HH_OCC[i]
            hh_data[2][20][i] = PT_QUALITY[i]
            hh_data[3][0][i] = REICHWEITE_DURCH100[i]

        # @njit
        def get_utility_fast_MNL_cars(c, data, initial_point, asc_offset):
            if c != 0:
                res_temp = initial_point[c - 1] + asc_offset[c]
            else:
                res_temp = asc_offset[c]
            for a in range(no_constant_fixed):
                res_temp = res_temp + initial_point[(count_c - 1) + a] * data[0][a][c]
            for a in range(no_constant_random):
                res_temp = (
                    res_temp
                    + initial_point[(count_c - 1) + no_constant_fixed + a]
                    * data[1][a][c]
                )
            for a in range(no_variable_fixed):
                res_temp = (
                    res_temp
                    + initial_point[
                        (count_c - 1)
                        + no_constant_fixed
                        + no_constant_random
                        + (no_variable_fixed + no_variable_random) * c
                        + a
                    ]
                    * data[2][a][c]
                )
            for a in range(no_variable_random):
                res_temp = (
                    res_temp
                    + initial_point[
                        (count_c - 1)
                        + no_constant_fixed
                        + no_constant_random
                        + (no_variable_fixed + no_variable_random) * c
                        + no_variable_fixed
                        + a
                    ]
                    * data[3][a][c]
                )
            return res_temp

        # @njit
        def calculate_logit_fast_MNL_cars(
            alternative, av, data, initial_point, asc_offset
        ):
            """
            This method calculates the multinomial logit probability for a given
            set of coefficients and all choices in the sample of the dataset.

            Returns
            -------
            Probability of MNL model at the initial point.

            """
            # calculate top
            top = av[alternative] * np.exp(
                get_utility_fast_MNL_cars(alternative, data, initial_point, asc_offset)
            )
            # calculate bottom
            bottom = 0
            for c in range(count_c):
                bottom += av[c] * np.exp(
                    get_utility_fast_MNL_cars(c, data, initial_point, asc_offset)
                )

            return top / bottom

        # multinomial logit
        probs = []
        for i in all_alternatives:
            prob_temp = calculate_logit_fast_MNL_cars(
                i, AV, hh_data, self.initial_point, np.array(ASC_OFFSET)
            )
            probs.append(prob_temp)

        return probs

    def simulate_hh_car_type_japan(
        self,
        AV,
        SEGMENT,
        OPERATING_COSTS_SCALED,
        RANGE_SCALED,
        CHARGING_TIME,
        CHARGING_DISTANCE,
        CO2,
        HH_SIZE,
        MEAN_HH_AGE,
        CAR_PARK_EV,
        CARSHARING,
        LONG_RANGE_AV,
        EV_EXPERIENCE,
        OWN_POWER,
        PT_QUALITY,
        RELATIVE_COSTS,
        **kwargs
    ):

        """
        Please contact the package maintainers via GitHub to receive the 
        required choice estimates of the MNL-model and further simulation instructions.
        """

        ASC_OFFSET = kwargs.get("ASC_OFFSET", [0, 0, 0, 0])

        count_c = 4  # number of alternatives: 0-3
        all_alternatives = np.array((0, 1, 2, 3))

        no_constant_fixed = 0
        no_constant_random = 0
        no_variable_fixed = 18
        no_variable_random = 1
        # specify maximum number of alternatives
        dim_aggr_alt_max = max(
            no_constant_fixed,
            no_constant_random,
            no_variable_fixed,
            no_variable_random,
        )

        # Define hh_data.
        # IMPORTANT: The order of parameters (see hh_data) must be equal to the order during
        # estimation (see param), as defined in param = {...} !!!
        hh_data = np.zeros((4, dim_aggr_alt_max, count_c), dtype="float64")

        map_segment = {
            "Kei": 0,
            "Small": 1,
            "Sedan": 2,
            "Mini-Van": 3,
            "Full-Size-Van": 4,
        }

        # fill parameters: variable_fixed
        for i in range(count_c):
            # SEGMENT
            segment_temp = map_segment[SEGMENT[i]]
            hh_data[2][segment_temp][i] = 1
            all_seg = [0, 1, 2, 3, 4]
            all_seg.remove(segment_temp)
            for seg in all_seg:
                hh_data[2][seg][i] = 0

            # OTHER ATTRIBUTES
            hh_data[2][6][i] = OPERATING_COSTS_SCALED[i]
            hh_data[2][7][i] = RANGE_SCALED[i]
            hh_data[2][8][i] = CHARGING_DISTANCE[i]
            hh_data[2][9][i] = CO2[i]
            hh_data[2][10][i] = HH_SIZE[i]
            hh_data[2][11][i] = MEAN_HH_AGE[i]
            hh_data[2][12][i] = CAR_PARK_EV[i]
            hh_data[2][13][i] = CARSHARING[i]
            hh_data[2][14][i] = LONG_RANGE_AV[i]
            hh_data[2][15][i] = EV_EXPERIENCE[i]
            hh_data[2][16][i] = OWN_POWER[i]
            hh_data[2][17][i] = PT_QUALITY[i]
            hh_data[3][0][i] = RELATIVE_COSTS[i]

        # @njit
        def get_utility_fast_MNL_cars(c, data, initial_point, asc_offset):
            """
            This method calculates the utility of choice alternative c.

            Parameters
            ----------
            c : int
                Choice alternative.
            data : numpy-array
                Choice attributes.
            initial_point : numpy array
                parameters
            asc_offset : numpy array
                Offset values for ASC constants.

            Returns
            -------
            res_temp : float
                Utility of choice option c.
            res_parts : list
                Utility parts of choice option c.

            """
            res_parts = []

            if c != 0:
                res_part_temp = initial_point[c - 1] + asc_offset[c]
                res_parts.append(res_part_temp)
                res_temp = res_part_temp
            else:
                res_part_temp = asc_offset[c]
                res_parts.append(res_part_temp)
                res_temp = res_part_temp
            for a in range(no_constant_fixed):
                res_part_temp = initial_point[(count_c - 1) + a] * data[0][a][c]
                res_parts.append(res_part_temp)
                res_temp = res_temp + res_part_temp
            for a in range(no_constant_random):
                res_part_temp = (
                    initial_point[(count_c - 1) + no_constant_fixed + a] * data[1][a][c]
                )
                res_parts.append(res_part_temp)
                res_temp = res_temp + res_part_temp

            for a in range(no_variable_fixed):
                res_part_temp = (
                    initial_point[
                        (count_c - 1)
                        + no_constant_fixed
                        + no_constant_random
                        + (no_variable_fixed + no_variable_random) * c
                        + a
                    ]
                    * data[2][a][c]
                )
                res_parts.append(res_part_temp)
                res_temp = res_temp + res_part_temp
            for a in range(no_variable_random):
                res_part_temp = (
                    initial_point[
                        (count_c - 1)
                        + no_constant_fixed
                        + no_constant_random
                        + (no_variable_fixed + no_variable_random) * c
                        + no_variable_fixed
                        + a
                    ]
                    * data[3][a][c]
                )
                res_parts.append(res_part_temp)
                res_temp = res_temp + res_part_temp
            return res_temp, res_parts

        # @njit
        def calculate_logit_fast_MNL_cars(
            alternative, av, data, initial_point, asc_offset
        ):
            """
            This method calculates the multinomial logit probability for a given
            set of coefficients and all choices in the sample of the dataset.

            Returns
            -------
            Probability of MNL model at the initial point.

            """
            # calculate top
            utility_temp = get_utility_fast_MNL_cars(
                alternative, data, initial_point, asc_offset
            )
            top = av[alternative] * np.exp(utility_temp[0])
            utility_parts = utility_temp[1]
            # calculate bottom
            bottom = 0
            for c in range(count_c):
                utility_temp = get_utility_fast_MNL_cars(
                    c, data, initial_point, asc_offset
                )
                bottom += av[c] * np.exp(utility_temp[0])

            logit_prob = top / bottom

            return logit_prob, utility_parts

        # multinomial logit
        probs = []
        utility_parts = {}
        for i in all_alternatives:
            logit_res = calculate_logit_fast_MNL_cars(
                i, AV, hh_data, self.initial_point, np.array(ASC_OFFSET)
            )
            prob_temp = logit_res[0]
            probs.append(prob_temp)
            utility_parts[i] = logit_res[1]

        return probs, utility_parts
