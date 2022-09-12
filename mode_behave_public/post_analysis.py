# -*- coding: utf-8 -*-
"""
Created on Fri Jan 29 16:39:30 2021

@author: j.reul

This module holds the class "PostAnalysis", which incorporates functionality
to analyze the estimated mixed logit model.
"""

import pandas as pd
import numpy as np
from math import sqrt, pow, floor, exp
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.cluster import KMeans, AgglomerativeClustering, MeanShift, DBSCAN
from operator import mod
from numba import guvectorize, prange, njit
import seaborn as sns

class PostAnalysis:
    """
    This class incorporates methods to analyze the estimated mixed logit
    model as well as methods to visualize the estimation & simulation results.
    """
    
    def _init_(self):
        pass          

    def loglike_MNL(self):
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
        
        top = np.zeros(shape=self.av.shape[2], dtype="float64")
        bottom = np.zeros(shape=self.av.shape[2], dtype="float64")
        for c in range(self.count_c):
            for e in range(self.count_e):
                top += self.av[c][e] * self.choice[c][e] * np.exp(self.get_utility(c,e))
                bottom += self.av[c][e] * np.exp(self.get_utility(c,e))
        
        self.check_top = top
        self.check_bottom = bottom
        
        log_res = np.log(top/bottom)
        res = np.nansum(log_res)
        number_nan = np.sum(np.isnan(log_res))
        
        return res, number_nan
    
    def get_utility(self, c, e):
        """
        Calculation of the utility-function for all observations within 
        a given data-sample with respect to the persons choice c.

        Parameters
        ----------
        c : int
            c is the choice.
        point : list
            point specifies the random parameters within the parameter space.

        Returns
        -------
        list
            Returns the utility for each observation.

        """
        
        if c == 0:
            ASC = 0
        else:
            ASC = self.initial_point[c-1]
        
        utility = (
            ASC +
            np.sum(
                [self.initial_point[(self.count_c-1) + a] * 
                 self.data[self.param['constant']['fixed'][a] + '_' + str(c) + '_' + str(e)] 
                 for a in range(self.no_constant_fixed)], axis=0
                ) +
            np.sum(
                [
                    self.initial_point[
                        (self.count_c-1) + 
                        self.no_constant_fixed + a
                        ] * 
                    self.data[self.param['constant']['random'][a] + '_' + str(c) + '_' + str(e)] 
                    for a in range(self.no_constant_random)
                    ], axis=0
                ) +
            np.sum(
                [
                    self.initial_point[
                        (self.count_c-1) + 
                        self.no_constant_fixed + 
                        self.no_constant_random + 
                        (self.no_variable_fixed + 
                         self.no_variable_random)*c + a
                                       ] * 
                    self.data[
                        self.param['variable']['fixed'][a] + '_' + str(c) + '_' + str(e)
                        ] for a in range(self.no_variable_fixed)
                 ], axis=0
                ) +
            np.sum(
                [
                    self.initial_point[
                        (self.count_c-1) + 
                        self.no_constant_fixed + 
                        self.no_constant_random + 
                        (self.no_variable_fixed + self.no_variable_random)*c + 
                        self.no_variable_fixed + a
                        ] * self.data[
                            self.param['variable']['random'][a] + '_' + str(c) + '_' + str(e)
                            ] for a in range(self.no_variable_random)
                 ], axis=0
                )
            )
        
        return utility
    
    def loglike_MXL(self, **kwargs):
        """
        For reference on log-likelihood calculation see:
            Ch. 5.5 (pp. 118) in "Discrete Choice Analysis", by Ben-Akiva (1985)
        
        Returns
        -------
        Float64
            Returns the log-likelihood (LL) of the estimated mixed logit model.
            The following should hold for model validity:
                LL_mixed_logit > LL_initial_point (multinomial logit)
                
        """
        points_in = kwargs.get('points_in', self.points)
        
        # calculates the logit probabilities 
        # for each data point and each choice option
        logit_vector = self.simulate_mixed_logit(
            points_in=points_in, 
            vector_output=True
            )
        
        logit_vector_s = np.swapaxes(logit_vector, 0, 1)
        
        # calculates the logit probability for the chosen choice option
        logit_vector_choice = np.sum(np.sum(logit_vector_s*self.choice, axis=0), axis=0)
        
        # get the log of each logit probability
        logit_vector_choice_log = np.log(logit_vector_choice)
        
        # return the sum of the log-probabilities. Ignore nan-values
        
        return np.nansum(logit_vector_choice_log)
    
    def visualize_space(self, **kwargs):
        """
        This method visualizes the distribution of preferences across the 
        base population for the randomized model attributes, which have been
        analyzed within the estimation of the mixed logit model. 
        Furthermore, the estimated (mean) preferences from the multinomial
        logit model are displayed as reference points.

        Parameters
        ----------
        kwargs return_res : Boolean
            If True, the clustering results are returned. Defaults to False.
            
        kwargs cluster_method : string
            Specification of the clustering method, which shall be used to cluster
            the preferences estimated by the mixed logit model.
            Defaults to "kmeans". Other options: "agglo", "meanshift", "dbscan".
            
        kwargs scale_individual : Boolean
            If True, the x- and y-axis of the visualizations are scaled to one.
            This eases the comparison of the different attributes. The scale
            is indicates on the respective axes and is very important for 
            quantitative and qualitative interpretations. Defaults to False.
            
        kwargs external_points : array
            This array holds further parameter points in the parameter space,
            which should be visualized as reference points. E.g.: The initial
            point, as being calculated by the multinomial logit model.

        kwargs k : int
            Number of cluster centers to be calculated. Defaults to 3.

        kwargs save_fig_path : string
            Path, which indicated the place where to store the visualization
            as a .png-file.
            
        kwargs name_scenario : string
            The scenario name can be added additionally, to distinguish 
            several scenarios.
            
        kwargs bw_adjust : float
            This value adjusts the smoothing of the visualized preference
            distribution. A higher value increases the smoothing of the 
            displayed curve, but may conceal certain findings in the distribution.
            Defaults to 0.03.

        kwargs names_choice_options : dict
            If given, this shall be a dictionary, which holds the 
            names of the choice options as values and the numerical
            indication of the choice option (0,1,2,...) as keys.

        Returns
        -------
        res_clustering : List
            Clustering results are returned, if keyword return_res == True.

        """
                   
        sns.set_theme(style="white", rc={"axes.facecolor": (0, 0, 0, 0)})
             
        return_res = kwargs.get('return_res', False)
        method_temp = kwargs.get('cluster_method', 'kmeans')
        names_choice_options = kwargs.get("names_choice_options", {})
        
        #PREPARE DATA
        #   step 0: Get row names (random variables)
        names_constant_fixed = self.param['constant']['fixed'] 
        names_constant_random = self.param['constant']['random'] 
        names_variable_fixed = self.param['variable']['fixed']        
        names_variable_random = self.param['variable']['random']
        number_random = len(self.param['constant']['random']) + len(self.param['variable']['random'])*self.count_c
        number_variable_random = len(self.param['variable']['random'])
        
        #   step 1: Scale parameters
        try:
            points = np.nan_to_num(self.points)
        except:
            points = np.nan_to_num(self.get_points(np.array(self.shares.index, dtype='int64')))
            
        #   get only points, above share-treshold.
        shares = self.shares.copy()
        points_t = points.T
        
        bw_adjust_temp = kwargs.get('bw_adjust', 0.03)
        
        scale_individual = kwargs.get('scale_individual', False)
        if scale_individual:
            scale_log = []
            for i in range(number_random):
                min_temp = np.min(points_t[i])
                max_temp = np.max(points_t[i])
                scale_temp = max(abs(min_temp), abs(max_temp))
                if scale_temp != 0:
                    scale_log += [scale_temp]
                    points_t[i] = points_t[i] / scale_temp
                else:
                    scale_log += [1]
            points_scaled = points_t.T
        else:
            scale = 0
            for i in range(number_random):
                min_temp = np.min(points_t[i])
                max_temp = np.max(points_t[i])
                scale_temp = max(abs(min_temp), abs(max_temp))
                if scale_temp > scale:
                    scale = scale_temp
            points_t = points_t / scale
            points_scaled = points_t.T
        
        #   Import points of socio-economic groups
        external_points = kwargs.get('external_points', np.array([]))
        k_cluster = kwargs.get('k', 3)
        if external_points.size:
            #get random points.
            external_points_random = np.zeros(shape=(external_points.shape[0], number_random), dtype='float64')
            for group in range(external_points.shape[0]):
                if self.param_transform:
                    #convert external point.
                    external_points[group] = self.transform_initial_point(self.param_init, self.param_transform, point=external_points[group])
                
                for c in range(len(names_constant_random)):
                    index_temp = self.count_c-1 + len(names_constant_fixed) + c
                    external_points_random[group][c] = external_points[group][index_temp]
                for i in range(self.count_c):
                    for v in range(len(names_variable_random)):
                        index_temp = (
                                    self.count_c-1 + len(names_constant_fixed) + len(names_constant_random) + 
                                    (len(names_variable_fixed) + len(names_variable_random))*i +
                                    len(names_variable_fixed) + v
                                    )
                        external_points_random[group][len(names_constant_random) + len(names_variable_random)*i + v] = external_points[group][index_temp] 
                      
            # Get cluster centers
            k_group = external_points_random.shape[0]
            res_clustering = self.cluster_space(method_temp, k_cluster)
                        
            #scale these random points.
            if scale_individual:
                external_points_random_t = external_points_random.T
                for i in range(number_random):
                    external_points_random_t[i] = external_points_random_t[i] / scale_log[i]
                external_points_random = external_points_random_t.T
            else:
                external_points_random = external_points_random / scale
                                   
            #convert points to dict-format
            count_random_variable = 0   
            vlines_loc_group = {}
            external_points_random_t = external_points_random.T
            for i in range(len(self.param['constant']['random'])):
                vlines_loc_group[names_constant_random[i]] = external_points_random_t[count_random_variable]
                count_random_variable += 1
            for c in range(self.count_c):                        
                for i in range(len(self.param['variable']['random'])):
                    if c in list(names_choice_options):
                        name_co = names_choice_options[c]
                    else:
                        name_co = str(c)
                    
                    vlines_loc_group[names_variable_random[i] + '_' + name_co] = external_points_random_t[count_random_variable]
                    count_random_variable += 1
            
        else:
            print('No external reference points given.')
            vlines_loc_group = {}
            k_group = 0
            res_clustering = self.cluster_space(method_temp, k_cluster)
                
        if method_temp in ('meanshift', 'dbscan'):
            k_cluster = res_clustering[0].shape[0]
            
        # step 2: Built dataframe
        df = pd.DataFrame(points_scaled.flatten(order='F'), columns=['x'])
        attributes_temp = []
        vlines_loc = {}
        vlines_len = {}
        if scale_individual:
            text_scale = {}
            
        cluster_center = res_clustering[0].T
        
        #scaling of cluster centers
        if scale_individual:
            for i in range(number_random):
                cluster_center[i] = cluster_center[i] / scale_log[i]
        else:
            cluster_center = cluster_center / scale
        #weighted calculation of cluster sizes
        cluster_labels_pd = pd.DataFrame(columns=['labels', 'weights'])
        cluster_labels_pd['labels'] = res_clustering[1]
        #assign weights
        if method_temp in ('agglo', 'meanshift'):
            index_clustered = res_clustering[2]
            cluster_labels_pd = cluster_labels_pd.reset_index(drop=True)
            cluster_labels_pd['weights'] = self.shares.values[index_clustered]
        else:
            cluster_labels_pd['weights'] = self.shares.values
        cluster_sizes_rel = [cluster_labels_pd.loc[cluster_labels_pd['labels'] == i, 'weights'].sum() for i in range(k_cluster)]
        
        #sort cluster_center and cluster_sizes_rel
        cluster_sizes_rel_pd = pd.Series(cluster_sizes_rel)
        cluster_sizes_rel_pd = cluster_sizes_rel_pd.sort_values(ascending=False)
        cluster_sizes_rel = cluster_sizes_rel_pd.values
        cluster_sizes_rel_pd = cluster_sizes_rel_pd.reset_index()
        #reshuffle cluster_center
        cluster_center = cluster_center.T[cluster_sizes_rel_pd['index'].values,].T
        
        #create inputs for map of vlines and text_scale
        count_random_variable = 0    
        for i in range(len(self.param['constant']['random'])):
            if scale_individual:
                text_scale[names_constant_random[i]] = scale_log[count_random_variable]
            vlines_loc[names_constant_random[i]] = cluster_center[count_random_variable]
            vlines_len[names_constant_random[i]] = cluster_sizes_rel
            count_random_variable += 1
            attributes_temp += [names_constant_random[i]]*len(shares.values)
        for c in range(self.count_c):
            for i in range(len(self.param['variable']['random'])):
                if c in list(names_choice_options):
                    name_co = names_choice_options[c]
                else:
                    name_co = str(c)
                
                if scale_individual:
                    text_scale[names_variable_random[i] + '_' + name_co] = scale_log[count_random_variable]
                    
                vlines_loc[names_variable_random[i] + '_' + name_co] = cluster_center[count_random_variable]
                vlines_len[names_variable_random[i] + '_' + name_co] = cluster_sizes_rel
                count_random_variable += 1
                attributes_temp += [names_variable_random[i] + '_' + name_co]*len(shares.values)
        df['g'] = attributes_temp
        df = df.sort_values(by='g')
        weights_ = shares.values
        for w in range(number_random-1):
        #for w in range(self.count_c-1):
            weights_ = np.append(weights_, shares.values)
        df['weights'] = weights_
        

        # Initialize color palettes
        pal = sns.cubehelix_palette(n_colors=1, start=2.35, rot=-.1,dark=0.4,light=0.75)
        pal_group = sns.cubehelix_palette(n_colors=k_group, start=0, rot=-.1,dark=0.3,light=0.65,hue=1)
        pal_cluster_long = sns.color_palette("YlOrBr",n_colors=k_cluster*2)
        pal_cluster = pal_cluster_long[(k_cluster-1):-1]
        
        #create kde-plots. 
        fig, ax = plt.subplots(number_random, 1, sharex=True, figsize=(6, number_random))
        for c in range(self.count_c):
            for a_count, attr_ in enumerate(self.param['variable']['random']):
                if c in list(names_choice_options):
                    name_co = names_choice_options[c]
                else:
                    name_co = str(c)
                x_temp = df.loc[df['g'] == attr_ + '_' + name_co].groupby(['x']).sum().index.values
                weights_temp = df.loc[df['g'] == attr_ + '_' + name_co].groupby(['x']).sum()['weights']
                vis_col = c*number_variable_random+a_count
                print(vis_col)
                sns.kdeplot(x=x_temp,
                   bw_adjust=bw_adjust_temp, cut=0, weights=weights_temp, color=pal[0],
                   fill=True, linewidth=1.5, ax=ax[vis_col])
        
        # Set the subplots to overlap
        fig.subplots_adjust(hspace=.4)
        
        # Remove axes details that don't play well with overlap
        fig.suptitle('Distribution of Preferences', fontsize=14, fontweight="bold", y=0.95)
        plt.setp(ax, 
                 xticks=[-0.8, 0, 0.8], 
                 xticklabels=['Max. Negative Impact', 'No Impact', 'Max. Positive Impact'], 
                 yticks=[]
                 )
        plt.xlim(-1,1)
        for axis_no, axis in enumerate(ax):
            #set y-labels
            label_modulus = axis_no % len(self.param['variable']['random'])
            label_temp = self.param['variable']['random'][label_modulus]
            count_temp = int(axis_no / len(self.param['variable']['random']))
            if count_temp in list(names_choice_options):
                col_name = label_temp + '_' + names_choice_options[count_temp]
            else:
                col_name = label_temp + '_' + str(count_temp)
            axis.set_ylabel("")
            axis.set_ylim(bottom=0)
            bbox_temp = axis.dataLim.get_points()
            y_max_temp = bbox_temp[1][1]
            self.check_y_max = y_max_temp
                
            scale_y = round(y_max_temp,2)
            axis.text(x=-0.95, y=1.05*y_max_temp, s=col_name, 
                      horizontalalignment='left',
                      verticalalignment='bottom',
                      weight='bold')
                
            #set vertical lines
            vlines_loc_cluster = vlines_loc
            vlines_loc_group = vlines_loc_group
            k_cluster = k_cluster
            k_group = k_group
            if scale_individual:
                scale_temp = round(text_scale[col_name],2)
                label_text = ('scale x: ' + str(scale_temp) + '\n' +
                              'scale y: ' + str(scale_y))
                axis.text(x=0.98, y=0, 
                          horizontalalignment='right',
                          verticalalignment='bottom',
                          s=label_text, fontstyle='italic', size=9)
            for cluster in range(k_cluster):
                x_cluster = vlines_loc_cluster[col_name][cluster]
                #len_cluster = vlines_len_cluster[col_name][cluster]
                len_cluster = 0.9
                axis.axvline(x=x_cluster, ymax=len_cluster, c=pal_cluster[cluster], lw=2, clip_on=False)
            for group in range(k_group):
                x_group = vlines_loc_group[col_name][group]
                axis.axvline(x=x_group, ymax=0.9, c=pal_group[group], lw=2, clip_on=False)
                
            axis.axvline(x=0, ymax=-0.15, c='0', label='0',lw=1, clip_on=False)
            axis.axvline(x=1, ymax=-0.15, c='0', label='0',lw=1, clip_on=False)
            axis.axvline(x=-1, ymax=-0.15, c='0', label='0',lw=1, clip_on=False)
            
        # Create the legend patches for cluster
        patch_dict = {}
        
        if 0 in list(names_choice_options):
            name_co_0 = names_choice_options[0]
        else:
            name_co_0 = str(0)
        
        col_name = self.param['variable']['random'][0] + '_' + name_co_0
        for i in range(k_cluster):
            self.vlines_len_temp = vlines_len
            cluster_size_temp = int(round(vlines_len[col_name][i]*100,0))
            patch_dict['C' + str(i+1) + ': ' + str(cluster_size_temp) + '%'] = pal_cluster[i]
        patches_c = [mpatches.Patch(color=c, label=l) for l,c in patch_dict.items()]
        
        #create legends
        legend1 = plt.legend(handles=patches_c, loc='lower center', 
                             ncol=k_cluster, bbox_to_anchor=(0.5,-1.5),
                             columnspacing=1,
                             title='Cluster: Size(%)', fancybox=True, 
                             shadow=False, facecolor='white')

        #add legends
        plt.gca().add_artist(legend1)        
                
        save_fig_path = kwargs.get('save_fig_path', self.PATH_Visualize)
        name_scenario = kwargs.get('name_scenario', False)
                
        if name_scenario:
            fig.savefig(save_fig_path + 'preference_distribution_' + name_scenario + ".png", dpi=300, bbox_inches='tight')
        else:
            fig.savefig(save_fig_path + 'preference_distribution.png', dpi=300, bbox_inches='tight')
            
        if return_res:
            return res_clustering    
                
    def get_points(self, index):
        """
        This methods draws explicit points for respective estimated shares 
        from the multidimensional space array, by indicating the index of
        the points within the space array.
        
        Parameters
        ----------
        index : array
            The index of specific points of the parameter space within
            the multidimensional space array.

        Returns
        -------
        point : array
            An array of explicitly drawn points.
        
        """
        
        if self.bits_64:
            @guvectorize(
                ['float64[:, :], int64[:], float64[:, :]'], '(r,m),(n)->(n,r)', 
                nopython=True, target="parallel"
                )
            def get_points_from_draws_vector(space, draws, drawn_points): 
                """
                This methods draws explicit points for respective estimated shares 
                from the multidimensional space array, by indicating the index of
                the points within the space array.
                    
                Parameters
                ----------
                space : array
                    Multidimensional array, which describes the complete
                    parameter space.
                draws : array
                    An array which holds the index of the points, which shall 
                    be explicitly drawn from space.
    
                Returns
                -------
                drawn_points : array
                    Array with explicitly drawn points from the parameter space.
    
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
                ['float32[:, :], int64[:], float32[:, :]'], '(r,m),(n)->(n,r)', 
                nopython=True, target="parallel"
                )
            def get_points_from_draws_vector(space, draws, drawn_points): 
                """
                This methods draws explicit points for respective estimated shares 
                from the multidimensional space array, by indicating the index of
                the points within the space array.
                    
                Parameters
                ----------
                space : array
                    Multidimensional array, which describes the complete
                    parameter space.
                draws : array
                    An array which holds the index of the points, which shall 
                    be explicitly drawn from space.
    
                Returns
                -------
                drawn_points : array
                    Array with explicitly drawn points from the parameter space.
    
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
        
        points_temp = get_points_from_draws_vector(self.space, index)
        
        return points_temp
    
    
    def transform_initial_point(self, param, param_t, **kwargs):
        """
        This method transforms the order of parameters within an
        initial point array to fit an alternative attribute specification.
        Therefore, the same attributes must have be considered during the 
        estimation of the MNL model, however a different order of attributes 
        could have been defined.
        
        Parameters
        ----------
        param : array
            Original definition of parameters/attributes.
        param_t : array
            Desired definition of parameters/attributes.
        kwargs point : array
            An exogenously defined initial_point.

        Returns
        -------
        array
            Re-ordered initial_point.
        """
        
        point = kwargs.get('point', self.initial_point)
        
        X = {}
        for i, key_0 in enumerate(param):
            for j, key_1 in enumerate(param[key_0]):
                for k, key_2 in enumerate(param[key_0][key_1]):
                    X[key_2] = (i*2+j,k)
                    
        X_t = {}
        for i, key_0 in enumerate(param_t):
            for j, key_1 in enumerate(param_t[key_0]):
                for k, key_2 in enumerate(param_t[key_0][key_1]):
                    X_t[key_2] = (i*2+j,k)       
        
        count_alt = self.count_c-1
        
        count_constant_fixed = 0
        count_constant_random = 0
        count_variable_fixed = 0
        count_variable_random = 0
        
        for key in X:
            pos = X[key]
            if pos[0] == 0:
                count_constant_fixed += 1
            elif pos[0] == 1:
                count_constant_random += 1
            elif pos[0] == 2:
                count_variable_fixed += 1
            else:
                count_variable_random += 1
                
        count_constant_fixed_t = 0
        count_constant_random_t = 0
        count_variable_fixed_t = 0
        count_variable_random_t = 0
        
        for key in X_t:
            pos_t = X_t[key]
            if pos_t[0] == 0:
                count_constant_fixed_t += 1
            elif pos_t[0] == 1:
                count_constant_random_t += 1
            elif pos_t[0] == 2:
                count_variable_fixed_t += 1
            else:
                count_variable_random_t += 1
                
        #initialize transformed initial point
        initial_point_t = np.zeros(shape=point.shape)
                
        for key in X:
                
            pos = X[key]
            pos_t = X_t[key]
            
            if pos[0] == 0:
                #constant fixed
                values = point[count_alt + pos[1]]
            elif pos[0] == 1:
                #constant random
                values = point[count_alt + count_constant_fixed+pos[1]]
            elif pos[0] == 2:
                #variable fixed
                values = []
                for c in range(self.count_c):
                    index_temp = (count_alt +
                            count_constant_fixed + 
                            count_constant_random + 
                            (count_variable_fixed + count_variable_random)*c +
                            pos[1]
                            )
                    values.append(point[index_temp])
                    
            else:
                #variable random
                values = []
                for c in range(self.count_c):
                    index_temp = (count_alt +
                            count_constant_fixed + 
                            count_constant_random + 
                            (count_variable_fixed + count_variable_random)*c +
                            count_variable_fixed + pos[1])
                    values.append(point[index_temp])
                    
            if pos_t[0] == 0:
                #constant fixed
                initial_point_t[count_alt + pos_t[1]] = values
            elif pos_t[0] == 1:
                #constant random
                initial_point_t[count_alt + count_constant_fixed_t+pos_t[1]] = values
            elif pos_t[0] == 2:
                #variable fixed
                for c in range(self.count_c):
                    index_temp = (count_alt +
                        count_constant_fixed_t + 
                        count_constant_random_t + 
                        (count_variable_fixed_t + count_variable_random_t)*c +
                        pos_t[1])
                    
                    initial_point_t[index_temp] = values[c]
            else:
                #variable random
                for c in range(self.count_c):
                    index_temp = (count_alt +
                        count_constant_fixed_t + 
                        count_constant_random_t + 
                        (count_variable_fixed_t + count_variable_random_t)*c +
                        count_variable_fixed_t + pos_t[1])
                    
                    initial_point_t[index_temp] = values[c]
                     
        return initial_point_t
    
    def weighted_cov(self, X, Y):
        
        average_x = np.average(X, weights=self.shares)
        average_y = np.average(Y, weights=self.shares)
        
        return np.average((X-average_x)*(Y-average_y), weights=self.shares)
    
    def weighted_corr(self, random_param_a, random_param_b):
        
        X = self.points.T[random_param_a]
        Y = self.points.T[random_param_b]
        cov_xy = self.weighted_cov(X,Y)
        var_x = self.weighted_cov(X,X)
        var_y = self.weighted_cov(Y,Y)
        std_x = sqrt(var_x)
        std_y = sqrt(var_y)
        
        return cov_xy / (std_x*std_y)
    
    def cluster_space(self, method, k, **kwargs):
        """
        This method analyses the estimated points and shares within the 
        parameter space and clustes them into latent classes,
        i.e. consumer groups.
        
        Parameters
        ----------
        method : string
            Clustering method.
        k : int
            Number of clusters.
        kwargs tol : float
            Tolerance. Defaults to 10e-7.
        kwargs points_affinity : boolean
            If True, an affinity index is calculated and returned. 
            Defaults to False.
        kwargs points : array
            Exogenously defined set of points to be analyzed.
        kwargs shares : array
            Exogenously defined set of shares to be analyzed.
            

        Returns
        -------
        list
            Returns a set of cluster results: 
            cluster_centers, labels, inertia, affinity_points
        """
        tol_temp = kwargs.get('tol', 10e-7)
        
        points_affinity = kwargs.get('points_affinity', False)
        
        shares = kwargs.get('shares', self.shares)
        
        try:
            points = kwargs.get('points', False)
            points.size
            points = np.nan_to_num(points)
        except:
            try:
                points = np.nan_to_num(self.points)
            except:
                points = np.nan_to_num(self.get_points(np.array(shares.index, dtype='int64')))
        
        if method == 'kmeans':                
            #create instance of KMeans-algorithm
            kmeans = KMeans(n_clusters=k, tol=tol_temp)
            #Compute cluster centers
            labels = kmeans.fit_predict(points, sample_weight=shares.values)
            #get inertia and silhouhette score for elbow method
            inertia = kmeans.inertia_
            #get cluster centers
            cluster_centers = kmeans.cluster_centers_
            #calculate cluster-distance for each point and attribute
            affinity_points = np.zeros(
                shape=(2, cluster_centers.shape[1], cluster_centers.shape[0]), 
                dtype='float64'
                )
            for a in range(cluster_centers.shape[1]):
                for c in range(cluster_centers.shape[0]):
                    center_point = cluster_centers[c][a]
                    points_label = points[labels==c]
                    points_label_attribute = points_label.T[a]
                    distance_mean = abs(points_label_attribute-center_point).mean()
                    affinity_points[0][a][c] = distance_mean
                    affinity_points[1][a][c] = center_point
            try:
                affinity_ind = kmeans.transform(points_affinity)
                return cluster_centers, labels, inertia, affinity_points, affinity_ind
            except:
                print('No individual points for aff.-calc. given.')
                return cluster_centers, labels, inertia, affinity_points
        elif method == 'agglo':
            #delete points below mean.
            shares_temp = shares.reset_index(drop=True)
            shares_temp = shares_temp.nlargest(n=int(len(shares_temp)*0.2), keep='all')
            index_above = shares_temp.index
            points_temp = points[shares_temp.index]
            #create instance of DBSCAN
            agglo = AgglomerativeClustering(n_clusters=k, linkage='ward')
            #Compute cluster centers
            labels = agglo.fit_predict(points_temp)
            #calculate cluster centers
            first_dim = len(np.unique(labels))
            second_dim = points_temp.shape[1]
            cluster_centers = np.zeros(shape=(first_dim, second_dim), dtype='float64')
            count = 0
            for l in np.unique(labels):
                points_sub = points_temp[labels==l]
                cluster_centers[count] = points_sub.mean(axis=0)
                count += 1
            #calculate cluster-distance for specific points
            self.check_centers = cluster_centers
            #calculate cluster-distance for each point and attribute
            affinity_points = np.zeros(shape=(2, cluster_centers.shape[1], cluster_centers.shape[0]), dtype='float64')
            for a in range(cluster_centers.shape[1]):
                for c in range(cluster_centers.shape[0]):
                    center_point = cluster_centers[c][a]
                    points_label = points_temp[labels==c]
                    points_label_attribute = points_label.T[a]
                    distance_mean = abs(points_label_attribute-center_point).mean()
                    affinity_points[0][a][c] = distance_mean
                    affinity_points[1][a][c] = center_point
            #calculate affinity_ind manually
            affinity_ind = np.zeros(shape=(points_affinity.shape[0], k), dtype='float64')
            for p in range(points_affinity.shape[0]):
                for c in range(k):
                    affinity_ind[p][c] = np.linalg.norm(cluster_centers[c] - points_affinity[p])
            return cluster_centers, labels, index_above, affinity_points, affinity_ind
        
        elif method == 'meanshift':
            #delete points below mean.
            shares_temp = shares.reset_index(drop=True)
            shares_temp = shares_temp.nlargest(n=int(len(shares_temp)*0.1), keep='all')
            index_above = shares_temp.index
            points_temp = points[shares_temp.index]
            #create instance of DBSCAN
            meanshift = MeanShift()
            #Compute cluster centers
            labels = meanshift.fit_predict(points_temp)
            #calculate cluster centers
            cluster_centers = meanshift.cluster_centers_
            #calculate cluster-distance for specific points
            self.check_centers = cluster_centers
            #calculate cluster-distance for each point and attribute
            affinity_points = np.zeros(shape=(2, cluster_centers.shape[1], cluster_centers.shape[0]), dtype='float64')
            for a in range(cluster_centers.shape[1]):
                for c in range(cluster_centers.shape[0]):
                    center_point = cluster_centers[c][a]
                    points_label = points_temp[labels==c]
                    points_label_attribute = points_label.T[a]
                    distance_mean = abs(points_label_attribute-center_point).mean()
                    affinity_points[0][a][c] = distance_mean
                    affinity_points[1][a][c] = center_point
            #calculate affinity_ind manually
            affinity_ind = np.zeros(shape=(points_affinity.shape[0], len(cluster_centers)), dtype='float32')
            for p in range(points_affinity.shape[0]):
                for c in range(len(cluster_centers)):
                    affinity_ind[p][c] = np.linalg.norm(cluster_centers[c] - points_affinity[p])
            return cluster_centers, labels, index_above, affinity_points, affinity_ind
        
        elif method == 'dbscan':
            #create instance of KMeans-algorithm
            dbscan = DBSCAN(eps=0.05, min_samples=10)
            #Compute cluster centers
            labels = dbscan.fit_predict(points, sample_weight=shares.values)
            #calculate cluster centers
            first_dim = len(np.unique(labels))
            second_dim = points.shape[1]
            cluster_centers = np.zeros(shape=(first_dim, second_dim), dtype='float32')
            count = 0
            for l in np.unique(labels):
                points_sub = points[labels==l]
                cluster_centers[count] = points_sub.mean(axis=0)
                count += 1
            #calculate cluster-distance for specific points
            self.check_centers = cluster_centers
            #calculate cluster-distance for each point and attribute
            affinity_points = np.zeros(shape=(2, cluster_centers.shape[1], cluster_centers.shape[0]), dtype='float64')
            for a in range(cluster_centers.shape[1]):
                for c in range(cluster_centers.shape[0]):
                    center_point = cluster_centers[c][a]
                    points_label = points_temp[labels==c]
                    points_label_attribute = points_label.T[a]
                    distance_mean = abs(points_label_attribute-center_point).mean()
                    affinity_points[0][a][c] = distance_mean
                    affinity_points[1][a][c] = center_point
            #calculate affinity_ind manually
            affinity_ind = np.zeros(shape=(points_affinity.shape[0], len(cluster_centers)), dtype='float64')
            for p in range(points_affinity.shape[0]):
                for c in range(len(cluster_centers)):
                    affinity_ind[p][c] = np.linalg.norm(cluster_centers[c] - points_affinity[p])
            return cluster_centers, labels, False, affinity_points, affinity_ind
        else:
            raise ValueError('No such method defined.')
            
    def simulate_logit(self, **kwargs):
        """
        This method simulates a multinomial logit model, based on the naming-
        conventions of the mixed logit model. 

        Parameters
        ----------
        kwargs param_transform : TYPE
            DESCRIPTION
        kwargs sense : dictionary
            The dictionary "sense" holds the attribute names for which sensitivities
            shall be simulated as keys. The values are the arrays or lists
            which indicate the relative change of the attribute value
            for each choice option.
        kwargs asc_offset : list
            offset values for alternative specific constants
            
        Returns
        -------
        float
            Return the mean value for the simulated latent class model.

        """
        
        param_transform = kwargs.get("param_transform", False)
        count_c = self.count_c
        count_e = self.count_e
        sense = kwargs.get("sense", {})
        external_point = kwargs.get("external_point", [])
        asc_offset = kwargs.get("asc_offset", np.array([0 for c in range(self.count_c)], dtype="float64"))

        try:
            no_constant_fixed = len(param_transform['constant']['fixed'])
            no_constant_random = len(param_transform['constant']['random'])
            no_variable_fixed = len(param_transform['variable']['fixed'])
            no_variable_random = len(param_transform['variable']['random'])
        except:
            no_constant_fixed = len(self.param['constant']['fixed'])
            no_constant_random = len(self.param['constant']['random'])
            no_variable_fixed = len(self.param['variable']['fixed'])
            no_variable_random = len(self.param['variable']['random'])
                    
        if len(external_point):
            if param_transform:
                initial_point = self.transform_initial_point(
                    param=self.param, 
                    param_t=self.param_transform,
                    point=external_point
                    )
            else:
                initial_point = external_point
        else:
            if param_transform:
                initial_point = self.transform_initial_point(
                    param=self.param, 
                    param_t=self.param_transform
                    )
            else:
                initial_point = self.initial_point
                
        dim_aggr_alt_max = max(
            len(self.param['constant']['fixed']),
            len(self.param['constant']['random']),
            len(self.param['variable']['fixed']),
            len(self.param['variable']['random']),
            )
            
        data = np.zeros((4,dim_aggr_alt_max,self.count_c,self.count_e,len(self.data)))
        for c in range(self.count_c):
            for e in range(self.count_e):
                for i, param in enumerate(self.param['constant']['fixed']):
                    if param in sense.keys():
                        data[0][i][c][e] = self.data[param + '_' + str(c) + '_' + str(e)].values*sense[param][c][e]
                    else:
                        data[0][i][c][e] = self.data[param + '_' + str(c) + '_' + str(e)].values
                        
                for i, param in enumerate(self.param['constant']['random']):                
                    if param in sense.keys():
                        data[1][i][c][e] = self.data[param + '_' + str(c) + '_' + str(e)].values*sense[param][c][e]
                    else:
                        data[1][i][c][e] = self.data[param + '_' + str(c) + '_' + str(e)].values
                    
                for i, param in enumerate(self.param['variable']['fixed']):
                    if param in sense.keys():
                        data[2][i][c][e] = self.data[param + '_' + str(c) + '_' + str(e)].values*sense[param][c][e]
                    else:
                        data[2][i][c][e] = self.data[param + '_' + str(c) + '_' + str(e)].values
                        
                for i, param in enumerate(self.param['variable']['random']):
                    if param in sense.keys():
                        data[3][i][c][e] = self.data[param + '_' + str(c) + '_' + str(e)].values*sense[param][c][e]
                    else:
                        data[3][i][c][e] = self.data[param + '_' + str(c) + '_' + str(e)].values
                
        def get_utility_vector(c, e, data, asc_offset):
            if c == 0:
                res_temp = asc_offset[0] + 0
            else:
                res_temp = asc_offset[0] + initial_point[c-1]
                
            for a in range(no_constant_fixed):
                res_temp += initial_point[(count_c-1) + a] * data[0][a][c][e]
            for a in range(no_constant_random):
                res_temp += initial_point[
                    (count_c-1) + 
                    no_constant_fixed + 
                    a
                    ] * data[1][a][c][e]
            for a in range(no_variable_fixed):
                res_temp += initial_point[
                    (count_c-1) + no_constant_fixed + no_constant_random + 
                    (no_variable_fixed + no_variable_random)*c + a
                    ] * data[2][a][c][e]
            for a in range(no_variable_random):
                res_temp += initial_point[
                    (count_c-1) + 
                    no_constant_fixed + 
                    no_constant_random + 
                    (no_variable_fixed + no_variable_random)*c + 
                    no_variable_fixed + a
                    ] * data[3][a][c][e]
                
            return res_temp   
        
        def calculate_logit_shares(av, data, asc_offset):
                
            logit_probs = np.zeros(shape=(count_c, count_e))
            
            #calculate bottom
            bottom = np.zeros(shape=av.shape[2])
            for c in range(count_c): 
                for e in range(count_e):
                    self.check_util_temp = np.exp(get_utility_vector(c, e, data, asc_offset))
                    self.check_av = av[c][e]
                    bottom += av[c][e] * np.exp(get_utility_vector(c, e, data, asc_offset))  
            for c in range(count_c):   
                for e in range(count_e):
                    top = av[c][e] * np.exp(get_utility_vector(c, e, data, asc_offset))
                    logit_probs[c][e] = np.mean(top/bottom)
                
            return logit_probs

        res = calculate_logit_shares(self.av, data, asc_offset)
            
        return np.sum(res, axis=1)
                    
    def simulate_mixed_logit(self, **kwargs):
        """
        Calculation of probabilities of mixed logit model for all
        observations within a given base-sample.
        Requires prior call of estimate_mixed_logit().

        Returns
        -------
        PandasSeries
            Returns a pandas series with model probabilities for each 
            observation.

        """
        
        mixing_distribution = kwargs.get("mixing_distribution", "discrete")
        sense = kwargs.get("sense", {})
        vector_output = kwargs.get("vector_output", False)
        asc_offset = kwargs.get("asc_offset", np.array([0 for c in range(self.count_c)], dtype="float64"))
        
        if mixing_distribution == "discrete":       
            initial_point = self.initial_point
            no_constant_fixed = self.no_constant_fixed
            no_constant_random = self.no_constant_random
            no_variable_fixed = self.no_variable_fixed
            no_variable_random = self.no_variable_random
            count_c = self.count_c
            count_e = self.count_e
            
            dim_aggr_alt_max = max(
                len(self.param['constant']['fixed']),
                len(self.param['constant']['random']),
                len(self.param['variable']['fixed']),
                len(self.param['variable']['random']),
                )
                
            data = np.zeros((4,dim_aggr_alt_max,self.count_c,self.count_e,len(self.data)))
            for c in range(self.count_c):
                for e in range(self.count_e):
                    for i, param in enumerate(self.param['constant']['fixed']):
                        if param in sense.keys():
                            data[0][i][c][e] = self.data[param + '_' + str(c) + '_' + str(e)].values*sense[param][c][e]
                        else:
                            data[0][i][c][e] = self.data[param + '_' + str(c) + '_' + str(e)].values
                            
                    for i, param in enumerate(self.param['constant']['random']):                
                        if param in sense.keys():
                            data[1][i][c][e] = self.data[param + '_' + str(c) + '_' + str(e)].values*sense[param][c][e]
                        else:
                            data[1][i][c][e] = self.data[param + '_' + str(c) + '_' + str(e)].values
                        
                    for i, param in enumerate(self.param['variable']['fixed']):
                        if param in sense.keys():
                            data[2][i][c][e] = self.data[param + '_' + str(c) + '_' + str(e)].values*sense[param][c][e]
                        else:
                            data[2][i][c][e] = self.data[param + '_' + str(c) + '_' + str(e)].values
                            
                    for i, param in enumerate(self.param['variable']['random']):
                        if param in sense.keys():
                            data[3][i][c][e] = self.data[param + '_' + str(c) + '_' + str(e)].values*sense[param][c][e]
                        else:
                            data[3][i][c][e] = self.data[param + '_' + str(c) + '_' + str(e)].values
                        
            @njit
            def get_utility_vector(c, e, point, l, data, asc_offset):
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
                    res_temp = asc_offset[0] + 0
                else:
                    res_temp = asc_offset[c] + initial_point[c-1]
                
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
        
            @guvectorize(
                ['float64[:, :], int64[:, :, :], float64[:, :, :, :, :], float64[:], float64[:, :, :, :]'], 
                '(m,p),(n,e,l),(i,j,n,e,l),(n)->(m,l,n,e)', 
                nopython=True, target="parallel"
                )
            def calculate_logit_vector(points, av, data, asc_offset, logit_probs_):
                
                for m in prange(points.shape[0]):  
                    point = points[m]
                    
                    #iterate over length of data array (len(av))
                    for l in prange(av.shape[2]):
                        #calculate bottom
                        bottom = 0
                        for c in prange(count_c):
                            for e in prange(count_e):
                                bottom += av[c][e][l] * exp(get_utility_vector(c, e, point, l, data, asc_offset))  
                        for c in prange(count_c): 
                            for e in prange(count_e):
                                top = av[c][e][l] * exp(get_utility_vector(c, e, point, l, data, asc_offset))
                                logit_probs_[m][l][c][e] = top/bottom  
                        
            logit_probs_matrix = calculate_logit_vector(self.points, self.av, data, asc_offset)
            #multiply logit probs per point with share of the point
            logit_probs_matrix_shares = self.shares.values*logit_probs_matrix.T
            #sum along all considered points of the parameter space
            logit_probs_summed = np.sum(logit_probs_matrix_shares, axis=3)
            
            
            if vector_output:
                res = logit_probs_summed
            else:
                #get mean of all probabilities
                res = np.sum(np.mean(logit_probs_summed, axis=2), axis=0)            
        else:
            raise ValueError("Not yet implemented.")
            
        return res    
        
    def simulate_latent_class(self, latent_points, latent_shares, **kwargs):
        """
        This method simulates a latent class model, based on the naming-
        conventions of the mixed logit model. The different latent classes
        refer to the different random points, being stored in the input
        parameter -latent_points-. The parameter -latent_shares- refers
        to the share of each latent class. The number of latent classes
        is usually a low integer value (3-10), while the number of classes 
        within the mixed logit model usually amounts to >1000.

        Parameters
        ----------
        latent_points : 2D numpy array.
            The random points within each class.
        latent_shares : 1D numpy array.
            The share of each class.
        kwargs param_transform : TYPE
            DESCRIPTION
        kwargs sense : dictionary
            The dictionary "sense" holds the attribute names for which sensitivities
            shall be simulated as keys. The values are the arrays or lists
            which indicate the relative change of the attribute value
            for each choice option.

        Returns
        -------
        float
            Return the mean value for the simulated latent class model.

        """
        
        param_transform = kwargs.get("param_transform", False)
        count_c = self.count_c
        count_e = self.count_e
        sense = kwargs.get("sense", {})
        asc_offset = kwargs.get("asc_offset", np.array([0 for c in range(self.count_c)], dtype="float64"))
        
        try:
            no_constant_fixed = len(param_transform['constant']['fixed'])
            no_constant_random = len(param_transform['constant']['random'])
            no_variable_fixed = len(param_transform['variable']['fixed'])
            no_variable_random = len(param_transform['variable']['random'])
        except:
            no_constant_fixed = len(self.param['constant']['fixed'])
            no_constant_random = len(self.param['constant']['random'])
            no_variable_fixed = len(self.param['variable']['fixed'])
            no_variable_random = len(self.param['variable']['random'])
            
        #check compatabiilty of latent_points and no_variable
        no_random = no_constant_random + no_variable_random*count_c
        if no_random != latent_points.shape[1]:
            raise ValueError('Defined parameter set -param- does not match number of random variables.')
        
        if param_transform:
            initial_point = self.transform_initial_point(param=self.param, param_t=self.param_transform)
        else:
            initial_point = self.initial_point
                
        dim_aggr_alt_max = max(
            len(self.param['constant']['fixed']),
            len(self.param['constant']['random']),
            len(self.param['variable']['fixed']),
            len(self.param['variable']['random']),
            )
            
        data = np.zeros((4,dim_aggr_alt_max,self.count_c,self.count_e,len(self.data)))
        for c in range(self.count_c):
            for e in range(self.count_e):
                for i, param in enumerate(self.param['constant']['fixed']):
                    if param in sense.keys():
                        data[0][i][c][e] = self.data[param + '_' + str(c) + '_' + str(e)].values*sense[param][c][e]
                    else:
                        data[0][i][c][e] = self.data[param + '_' + str(c) + '_' + str(e)].values
                        
                for i, param in enumerate(self.param['constant']['random']):                
                    if param in sense.keys():
                        data[1][i][c][e] = self.data[param + '_' + str(c) + '_' + str(e)].values*sense[param][c][e]
                    else:
                        data[1][i][c][e] = self.data[param + '_' + str(c) + '_' + str(e)].values
                    
                for i, param in enumerate(self.param['variable']['fixed']):
                    if param in sense.keys():
                        data[2][i][c][e] = self.data[param + '_' + str(c) + '_' + str(e)].values*sense[param][c][e]
                    else:
                        data[2][i][c][e] = self.data[param + '_' + str(c) + '_' + str(e)].values
                        
                for i, param in enumerate(self.param['variable']['random']):
                    if param in sense.keys():
                        data[3][i][c][e] = self.data[param + '_' + str(c) + '_' + str(e)].values*sense[param][c][e]
                    else:
                        data[3][i][c][e] = self.data[param + '_' + str(c) + '_' + str(e)].values
        
        @njit
        def get_utility_vector(c, e, point, l, data, asc_offset):
            if c == 0:
                res_temp = asc_offset[0] + 0
            else:
                res_temp = asc_offset[c] + initial_point[c-1]
                
            for a in range(no_constant_fixed):
                res_temp += initial_point[(count_c-1) + a] * data[0][a][c][e][l]
            for a in range(no_constant_random):
                res_temp += point[a] * data[1][a][c][e][l]
            for a in range(no_variable_fixed):
                res_temp += initial_point[
                    (count_c-1) + no_constant_fixed + no_constant_random + 
                    (no_variable_fixed + no_variable_random)*c + a
                    ] * data[2][a][c][e][l]
            for a in range(no_variable_random):
                res_temp += point[no_constant_random + no_variable_random*c + a] * data[3][a][c][e][l]
                
            return res_temp   
    
        @guvectorize(
            ['float64[:, :], int64[:, :, :], float64[:, :, :, :, :], float64[:], float64[:, :, :, :]'], 
            '(m,p),(n,e,l),(i,j,n,e,l),(n)->(m,l,n,e)', 
            nopython=True, target="parallel"
            )
        def calculate_logit_vector(points, av, data, asc_offset, logit_probs_):
            
            for m in prange(points.shape[0]):  
                point = points[m]
                
                #iterate over length of data array (len(av))
                for l in prange(av.shape[2]):
                    #calculate bottom
                    bottom = 0
                    for c in prange(count_c):
                        for e in prange(count_e):
                            bottom += av[c][e][l] * exp(get_utility_vector(c, e, point, l, data, asc_offset))  
                    for c in prange(count_c):   
                        for e in prange(count_e):
                            top = av[c][e][l] * exp(get_utility_vector(c, e, point, l, data, asc_offset))
                            logit_probs_[m][l][c][e] = top/bottom  
                    
        logit_probs = calculate_logit_vector(latent_points, self.av, data, asc_offset)
        res = np.zeros(shape=logit_probs[0].shape)
        for latent_class in range(logit_probs.shape[0]):
            res += logit_probs[latent_class]*latent_shares[latent_class]
            
        #sum over equal alternatives
        res_sum = np.sum(res, axis=2)
                    
        return np.mean(res_sum, axis=0)
            
    def forecast(self, method, **kwargs):        
        """
        This method creates a barplot of the mean values of different latent
        class and MNL models. The MNL models are based upon clustering results
        of random parameters from a previous simulation. Consequently, two
        latent class models are simulated. One is based upon the 
        estimated clusters with their corresponding, weighted cluster-sizes.
        The other latent class model utilizes the same clustered values,
        but assigns cluster-sizes, which are externally defined by 
        an user input during the method-call.

        Parameters
        ----------
        method : str
            Method indicates the model type, which to use for forecasting. 
            Options are: "MNL" (Multinomial Logit), "MXL" (Mixed Logit), 
            "LC" (Latent Class). Defaults to "MNL".
        kwargs sense_scenarios : dictionary
            The dictionary "sense_scenarios" is two dimensional.
            The first dimension indicated the scenario name, while the
            second dimension holds the scenario parameters according
            to the definition of "sense". "sense" itself is a dictionary.
            The dictionary "sense" holds the attribute names for which sensitivities
            shall be simulated as keys. The values are the arrays or lists
            which indicate the relative change of the attribute value
            for each choice option.
        kwargs external_points : numpy array
            This array is two-dimensional and holds one or more alternative
            specifications of "initial_point" for the simulation of 
            multinomial logit.
        kwargs k : int
            Number is cluster centers to be considered, when method = "LC"
        kwargs cluster_method : str
            The clustering method. Defaults to "kmeans."
        kwargs save_fig_path : str
            If given, the visualizations are stored in this directory.
        kwargs names_choice_options : dict
            If given, this shall be a dictionary, which holds the 
            names of the choice options as values and the numerical
            indication of the choice option (0,1,2,...) as keys.
        kwargs y_lim : tuple
            If given, this tuple indicates the limits for the y-axis
            within the visualization.
        kwargs y_lim : tuple
            If given, forecasted probabilities are returned. Defaults to False.

            
        Raises
        ------
        ValueError
            Is being raised, if an unknown method is indicated.

        Returns
        -------
        None
        
        """
        #PREPARE DATA
        #   Get row names (random variables)
        names_constant_fixed = self.param['constant']['fixed'] 
        names_constant_random = self.param['constant']['random'] 
        names_variable_fixed = self.param['variable']['fixed']        
        names_variable_random = self.param['variable']['random']
        number_random = (
            len(self.param['constant']['random']) + 
            len(self.param['variable']['random'])*self.count_c
            )
        
        save_fig_path = kwargs.get('save_fig_path', self.PATH_Visualize)
        external_points = kwargs.get('external_points', np.array([]))
        sense_scenarios = kwargs.get("sense_scenarios", False)
        names_choice_options = kwargs.get("names_choice_options", {})
        asc_offset = kwargs.get("asc_offset", np.array([0 for c in range(self.count_c)]))
        y_lim = kwargs.get("y_lim", ())
        data_output = kwargs.get("data_output", False)
        
        #Dictionary to store simulation results
        res_simu = {}
        
        if method == "MNL":
            res_simu['MNL'] = self.simulate_logit(asc_offset=asc_offset)
            
            if sense_scenarios:
                for sense_name in sense_scenarios.keys():
                    res_simu[sense_name] = self.simulate_logit(
                        asc_offset=asc_offset,
                        sense=sense_scenarios[sense_name]
                        )
                    
            if external_points.size:
                #iterate over external points.
                for ep in range(external_points.shape[0]):
                    res_simu['External ' + str(ep)] = self.simulate_logit(
                        asc_offset=asc_offset,
                        external_point = external_points[ep]
                        )
                
                    if sense_scenarios:
                        for sense_name in sense_scenarios.keys():
                            res_simu['External ' + str(ep) + ' - ' + sense_name] = self.simulate_logit(
                                asc_offset=asc_offset,
                                external_point = external_points[ep],
                                sense=sense_scenarios[sense_name]
                                )
        
        elif method == "LC":
            #   keyword arguments
            k = kwargs.get('k', 3)
            method_temp = kwargs.get('cluster_method', 'kmeans')
            
            #   step 1: Scale parameters
            try:
                points = np.nan_to_num(self.points)
            except:
                points = np.nan_to_num(self.get_points(np.array(self.shares.index, dtype='int64')))
            
            #   get only points, above share-treshold.
            shares = self.shares        
            points_scaled = points
        
            #   Import points of socio-economic groups
            if external_points.size:
                
                #get random points.
                external_points_random = np.zeros(shape=(external_points.shape[0], number_random), dtype='float32')
    
                for group in range(external_points.shape[0]):
                    if self.param_transform:
                        #convert external point.
                        external_points[group] = self.transform_initial_point(
                            self.param_init, self.param_transform, point=external_points[group]
                            )
                                    
                    for c in range(len(names_constant_random)):
                        index_temp = self.count_c-1 + len(names_constant_fixed) + c
                        external_points_random[group][c] = external_points[group][index_temp]
                    for v in range(len(names_variable_random)):
                        for i in range(self.count_c):
                            index_temp = (
                                        self.count_c-1 + len(names_constant_fixed) + len(names_constant_random) + 
                                        (len(names_variable_fixed) + len(names_variable_random))*i +
                                        len(names_variable_fixed) + v
                                        )
                            external_points_random[group][
                                len(names_constant_random) + len(names_variable_random)*i + v
                                ] = external_points[group][index_temp] 
                                          
                ext_points = True
            else:
                print('No external reference points given.')
                ext_points = False
            
            # Get cluster centers
            if ext_points:
                res_clustering = self.cluster_space(method_temp, k, points_affinity=external_points_random)
                affinity_all = res_clustering[4]
                affinity_percent_all = []
                for a in range(affinity_all.shape[0]):
                    affinity = affinity_all[a]
                    a_solve = np.zeros(shape=(len(affinity), len(affinity)))
                    a_solve[0] = [1]*len(affinity)
                    for i in range(1,len(affinity)):
                        ratio_temp = affinity[i] / affinity[0]
                        a_solve[i][0] = 1
                        a_solve[i][i] = -ratio_temp
                    b_solve = np.zeros(shape=len(affinity))
                    b_solve[0] = 1
                    affinity_solve = np.linalg.solve(a_solve,b_solve)
                    affinity_percent = np.round(affinity_solve*100).astype('int')
                    if np.allclose(np.dot(a_solve, affinity_solve), b_solve):
                        affinity_percent_all = affinity_percent_all + [affinity_percent]
                    else:
                        raise ValueError('Affinity-calculation failed.')
            else:
                res_clustering = self.cluster_space(method_temp, k, points=points_scaled, shares=shares)
            cluster_center = res_clustering[0]
            
            cluster_labels_pd = pd.DataFrame(columns=['labels', 'weights'])
            cluster_labels_pd['labels'] = res_clustering[1]
            #assign weights
            if method_temp in ('agglo', 'meanshift'):
                index_clustered = res_clustering[2]
                cluster_labels_pd = cluster_labels_pd.reset_index(drop=True)
                cluster_labels_pd['weights'] = self.shares.values[index_clustered]
            else:
                cluster_labels_pd['weights'] = self.shares.values
            
            if method_temp in ('meanshift', 'dbscan'):
                k = res_clustering[0].shape[0]
                
            cluster_sizes_rel = np.array(
                [cluster_labels_pd.loc[cluster_labels_pd['labels'] == i, 'weights'].sum() for i in range(k)]
                )
            
            #sort cluster_center and cluster_sizes_rel
            cluster_sizes_rel_pd = pd.Series(cluster_sizes_rel)
            cluster_sizes_rel_pd = cluster_sizes_rel_pd.sort_values(ascending=False)
            cluster_sizes_rel = cluster_sizes_rel_pd.values
            cluster_sizes_rel_pd = cluster_sizes_rel_pd.reset_index()
            #reshuffle cluster_center
            self.check_cluster_reorder = cluster_sizes_rel_pd
            cluster_center = cluster_center[cluster_sizes_rel_pd['index'].values,]
            
            cluster_sizes_rel_percent = np.round(cluster_sizes_rel*100).astype('int')
            
            #SIMULATION OF LATENT CLASSES AND EXTERNAL POINTS
            
            #MNL simulation for individual clusters.
            for k in range(k):
                res_simu['C' + str(k+1) + ' (' + str(cluster_sizes_rel_percent[k]) + "%)"] = self.simulate_latent_class(
                        np.array([cluster_center[k]]), 
                        np.array([1]), 
                        asc_offset=asc_offset
                        )
                if sense_scenarios:
                    for sense_name in sense_scenarios.keys():
                        res_simu['C' + str(k+1) + ' - ' + sense_name] = self.simulate_latent_class(
                                np.array([cluster_center[k]]), 
                                np.array([1]), 
                                asc_offset=asc_offset,
                                sense=sense_scenarios[sense_name]
                                )
                                    
            #Simulation of externally given points.
            if ext_points:
                k_group = external_points_random.shape[0]
                for g in range(k_group):
                    res_simu['External ' + str(g)] = self.simulate_latent_class(
                            np.array([external_points_random[g]]), 
                            np.array([1]), 
                            asc_offset=asc_offset
                            )
                        
                    if sense_scenarios:
                        for sense_name in sense_scenarios.keys():
                            res_simu['External ' + str(g) + ' - ' + sense_name] = self.simulate_latent_class(
                                    np.array([external_points_random[g]]), 
                                    np.array([1]), 
                                    asc_offset=asc_offset,
                                    sense=sense_scenarios[sense_name]
                                    )
                          
                    # The code below takes the affinity of external points to cluster
                    # points as a reference value to state the number of points, 
                    # which would belong to the external points, if they were
                    # cluster centers themselves. (group_size_percent)
                    
                    #cluster_affinity = ''
                    #group_size = 0
                    #for i in range(k):
                    #    cluster_affinity += 'C' + str(i+1) + ' - ' + str(affinity_percent_all[g][i]) + '%\n'
                    #    group_size += (affinity_percent_all[g][i]/100) * (cluster_sizes_rel_percent[i]/100)
                    #group_size_percent = round(group_size*100)
                    
            else:
                k_group = 0    
                                
            #Simulation of latent class model, based on cluster analysis.
            res_simu['Latent Class'] = self.simulate_latent_class(
                cluster_center, 
                cluster_sizes_rel,
                asc_offset=asc_offset
                )
            
            if sense_scenarios:
                for sense_name in sense_scenarios.keys():
                    res_simu['Latent Class' + ' - ' + sense_name] = self.simulate_latent_class(
                        cluster_center, 
                        cluster_sizes_rel,
                        sense=sense_scenarios[sense_name],
                        asc_offset=asc_offset
                        )
                                   
            cluster_sizes_str = ''
            for i in range(k):
                cluster_sizes_str += 'C' + str(i+1) + ' - ' + str(cluster_sizes_rel_percent[i]) + '%\n'
            
        elif method == "MXL":
            res_simu['MXL'] = self.simulate_mixed_logit(asc_offset=asc_offset)
            
            if sense_scenarios:
                for sense_name in sense_scenarios.keys():
                    res_simu[sense_name] = self.simulate_mixed_logit(
                        asc_offset=asc_offset,
                        sense=sense_scenarios[sense_name]
                        )
                    
            if external_points.size:
                #iterate over external points.
                for ep in range(external_points.shape[0]):
                    res_simu['External ' + str(ep)] = self.simulate_logit(
                        asc_offset=asc_offset,
                        external_point = external_points[ep]
                        )
                
                    if sense_scenarios:
                        for sense_name in sense_scenarios.keys():
                            res_simu['External ' + str(ep) + ' - ' + sense_name] = self.simulate_logit(
                                asc_offset=asc_offset,
                                external_point = external_points[ep],
                                sense=sense_scenarios[sense_name]
                                )
            
        else:
            raise ValueError("Chosen method is not available.")
               
        ### GENERAL CODE FOR VISUALIZATION STARTS BELOW ###
        
        #Observations in base data
        res_simu['Base Data'] = [
            np.sum([np.sum(self.data['choice_' + str(i) + '_' + str(e)]) for e in range(self.count_e)]) / len(self.data) for i in range(self.count_c)
            ]
        
        #Barplot
        res_simu_pd = pd.DataFrame(res_simu)
        simu_names = res_simu_pd.columns.values
        res_simu_pd["Choice Option"] = res_simu_pd.index
        
        res_simu_pd_long = pd.melt(
            res_simu_pd, 
            id_vars='Choice Option', 
            value_vars=simu_names
            )
        res_simu_pd_long = res_simu_pd_long.rename(
            columns={"value":"Choice probability", "variable":"Scenario"}
            )
        
        sns.set_theme(style="whitegrid")
        
        n_colors_temp = (len(res_simu)-1)*2
        custom_palette = sns.cubehelix_palette(n_colors=n_colors_temp, start=.5, rot=-.5)
        custom_palette[len(res_simu)-1] = (0.6, 0.6, 0.6) #add grey for the last bar
        
        #Specify name of choice options
        for choice_option in list(names_choice_options):
            name_temp = names_choice_options[choice_option]
            res_simu_pd_long.loc[
                res_simu_pd_long["Choice Option"] == choice_option, 
                "Choice Option"
                ] = name_temp
        
        ax = sns.barplot(
            x="Choice Option", y="Choice probability", 
            hue="Scenario", data=res_simu_pd_long, 
            palette=custom_palette
            )
        
        if y_lim:
            ax.set(ylim=y_lim)
        
        plt.legend(loc='upper right', bbox_to_anchor=(1.35, 1))
        
        if save_fig_path:
            fig = ax.get_figure()
            fig.savefig(save_fig_path + 'forecast.png', dpi=300, bbox_inches='tight')
            
        if data_output:
            return res_simu_pd_long
                