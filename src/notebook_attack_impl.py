# Auto-extracted from backdoor_codex_ember2024_win64.ipynb cells 3-4.
# This keeps the notebook attack utilities available to the script runner.

"""
Copyright (c) 2021, FireEye, Inc.
Copyright (c) 2021 Giorgio Severi
"""

import concurrent.futures
# import copy
import collections
import json
import os

import numpy as np
import pandas as pd

import mw_backdoor.constants as constants

class ImportantFeatureSelector(object):
    def __init__(self, feature_importance_df, criteria, fixed_features=None):
        """
        :param feature_importance_df A Dataframe with FeatureName and Importance columns; index is feature ID (0-(2351-1))
        :param criteria: Determines which features to return.
            Possible values:
                'most_important' - The most important N features will be returned.
                'least_important' - The least important N features will be returned.
                'least_important_nz' - The least important N features with non-zero importance will be returned.
        """
        self.feature_importance_df = feature_importance_df.sort_values(
            by='Importance', ascending=False)
        self.criteria = criteria
        self.fixed_features = fixed_features
        self.criteria_desc_map = {'most_important': '(most_imp) Features chosen from most important N features',
                                  'least_important': '(least_imp) Features chosen from least important N features',
                                  'least_important_nz': '(least_imp_nz) Features chosen from least important N non-zero features',
                                  'median_important': '(median_imp) Choses N features around the median importance value',
                                  }

    @property
    def name(self):
        return self.criteria

    @property
    def description(self):
        return self.criteria_desc_map[self.criteria]

    def get_features(self, num_features):
        if self.criteria.startswith('most_important'):
            most_important = self.feature_importance_df.index
            most_important = most_important[most_important.isin(
                self.fixed_features)]
            result = most_important[:num_features]

            # result = self.feature_importance_df.head(
            #    num_features).index[:num_features]

        elif self.criteria == 'least_important':
            result = self.feature_importance_df.tail(
                num_features).index[:num_features]
        elif self.criteria == 'least_important_nz':
            temp = self.feature_importance_df[self.feature_importance_df['Importance'] != 0]
            result = temp.tail(num_features).index[:num_features]
        elif self.criteria == 'median_important':
            median_index = self.feature_importance_df.shape[0] // 2
            result = self.feature_importance_df.index[median_index -
                                                      num_features // 2:]
            result = result[:num_features]
        else:
            raise ValueError(
                'Unsupported value of {} for the "criteria" argument'.format(self.criteria))
        return list(result)


class ShapleyFeatureSelector(object):
    def __init__(self, shap_values_df, criteria, fixed_features=None):
        """ Picks features based on the sum of Shapley values across all samples.
        :param shap_values_df A Dataframe of shape <# samples> x <# features>
        :param criteria: Determines which features to return.
            Possible values:
                'shap_smallest' - Returns the N features whose summed Shapley values are smallest (including negative).
                'shap_largest' - Returns the N features whose summed Shapley values are largest.
                'shap_nearest_zero' - Returns the N features whose summed Shapley values are closest to zero.
                'shap_nearest_zero_nz' - Returns the N features whose summed Shapley values are closest to zero and also not zero.
                'shap_nearest_zero_nz_abs' - Returns the N features whose summed absolute Shapley values are closest to zero and also not zero.
        """
        self.shap_values_df = shap_values_df
        self.criteria = criteria
        self.fixed_features = fixed_features
        self.criteria_desc_map = {'shap_smallest': '(shap_smallest) Choses N features whose summed Shapley values are smallest (including negative).',
                                  'shap_largest': '(shap_smallest) Choses N features whose summed Shapley values are largest.',
                                  'shap_nearest_zero': '(shap_nearest_zero) Choses N features whose summed Shapley values are closest to zero',
                                  'shap_nearest_zero_nz': '(shap_nearest_zero_nz) Choses N features whose summed Shapley values are closest to zero but not zero',
                                  'shap_nearest_zero_nz_abs': '(shap_nearest_zero_nz_abs) Choses N features whose summed absolute Shapley values are closest to zero but not zero'}

    @property
    def name(self):
        return self.criteria

    @property
    def description(self):
        return self.criteria_desc_map[self.criteria]

    def get_features(self, num_features):
        if self.criteria == 'shap_nearest_zero':
            summed = self.shap_values_df.sum()
            closest_to_zero = summed.abs().argsort()
            result = list(closest_to_zero[:num_features])
        elif self.criteria == 'shap_smallest':
            summed = self.shap_values_df.sum()
            result = list(summed.argsort()[:num_features])
        elif self.criteria == 'shap_largest':
            summed = self.shap_values_df.sum()
            result = list(summed.argsort()[-num_features:])
        elif self.criteria == 'shap_nearest_zero_nz':
            summed = self.shap_values_df.sum()
            summed[summed == 0.0] = np.inf
            closest_to_zero = summed.abs().argsort()
            result = list(closest_to_zero[:num_features])
        elif self.criteria == 'shap_nearest_zero_nz_abs':
            summed = self.shap_values_df.abs().sum()
            summed[summed == 0.0] = np.inf
            closest_to_zero = summed.argsort()
            result = list(closest_to_zero[:num_features])
        elif self.criteria == 'fixed_shap_nearest_zero_nz_abs':
            summed = self.shap_values_df.abs().sum()
            summed[summed == 0.0] = np.inf
            closest_to_zero = summed.argsort()
            # temp_features = [620, 618]
            # closest_to_zero = closest_to_zero[closest_to_zero.isin(temp_features)]
            closest_to_zero = closest_to_zero[closest_to_zero.isin(
                self.fixed_features)]
            result = list(closest_to_zero[:num_features])
            # result = list(closest_to_zero)
            print(result)
        elif self.criteria.startswith('shap_largest_abs'):
            summed = self.shap_values_df.abs().sum()
            closest_to_zero = summed.argsort()
            closest_to_zero = closest_to_zero[closest_to_zero.isin(
                self.fixed_features)]
            result = list(closest_to_zero[-num_features:])
            print(result)
        else:
            raise ValueError(
                'Unsupported value of {} for the "criteria" argument'.format(self.criteria))
        return result


class HistogramBinValueSelector(object):
    def __init__(self, criteria, bins):
        """
        :param X: The feature values for all of the samples as a 2 dimensional array (N samples x M features).
        :param criteria: Determines which bucket the returned values will fall in.
            Example: For illustrative purposes consider `bins`=5 and
                `X`=[[0, 1]
                     [0, 2],
                     [0, 2],
                     [0, 3],
                     [0, 3],
                     [0, 3],
                     [0, 4],
                     [0, 4],
                     [0, 5],
                     [0, 5]]
            Possible values:
                'min_bucket' - The bucket with the smallest overall value is chosen. Bucket 0 in the example.
                'max_bucket' - The bucket with the largest overall value is chosen. Bucket 4 in the example.
                'max_population' - The bucket with the most hits. Bucket 2 in the example.
                'min_population' - The bucket with the fewest hits. Bucket 0 in the example.
        :param bins: The number of bins to divide the samples into.
        """
        self.criteria = criteria
        self.criteria_desc_map = {'min_bucket': '(minval) Values chosen from bin with smallest value',
                                  'max_bucket': '(maxval) Values chosen from bin with largest value',
                                  'max_population': '(maxpop) Values chosen from bin with the largest population',
                                  'min_population': '(minpop) Values chosen from bin with the smallest population',
                                  }
        self.histogram_cache = {}
        self.bins = bins

        # The feature values for all of the samples as a 2 dimensional array (N samples x M features).
        self._X = None

    @property
    def name(self):
        return self.criteria

    @property
    def description(self):
        return self.criteria_desc_map[self.criteria]

    @property
    def X(self):
        return self._X

    @X.setter
    def X(self, value):
        self._X = value

    def get_feature_values(self, feature_ids):
        result = []
        for feature_id in feature_ids:
            count = collections.Counter(self._X[:, feature_id])
            result.append(min(count, key=count.get))
        # for feature_id in feature_ids:
        #    if feature_id not in self.histogram_cache:
        #        self.histogram_cache[feature_id] = np.histogram(
        #            self._X[:, feature_id], bins=self.bins)
        #    histogram_bucket_counts = self.histogram_cache[feature_id][0]
        #    histogram_edge_values = self.histogram_cache[feature_id][1]
        #    if self.criteria == 'min_bucket':
        #        bucket_index = 0
        #    elif self.criteria == 'max_bucket':
        #        bucket_index = len(histogram_bucket_counts) - 1
        #    elif self.criteria == 'max_population':
        #        bucket_index = np.argmax(histogram_bucket_counts)
        #    elif self.criteria == 'min_population':
        #        bucket_index = np.argmin(histogram_bucket_counts)
        #    else:
        #        raise ValueError(
        #            'Unsupported value of {} for the "criteria" argument'.format(self.criteria))
        #    result.append(
        #        np.mean(histogram_edge_values[bucket_index:bucket_index + 2]))

        return result


def _process_one_shap_linear_combination(feature_index_id_x_shaps_tuple):
    feat_index = feature_index_id_x_shaps_tuple[0]
    feature_id = feature_index_id_x_shaps_tuple[1]
    features_sample_values = feature_index_id_x_shaps_tuple[2]
    this_features_abs_inverse_shaps = feature_index_id_x_shaps_tuple[3]
    alpha = feature_index_id_x_shaps_tuple[4]
    beta = feature_index_id_x_shaps_tuple[5]

    # First, find values and how many times they occur
    (values, counts) = np.unique(features_sample_values, return_counts=True)
    counts = np.array(counts)
    # print('# Feature {} has {} unique values'.format(feature_id, len(counts)))
    sum_abs_shaps = np.zeros(len(values))
    for i in range(len(values)):
        desired_values_mask = features_sample_values == values[i]
        sum_abs_shaps[i] = np.sum(
            desired_values_mask * this_features_abs_inverse_shaps)
    sum_abs_shaps = alpha * (1.0 / counts) + beta * sum_abs_shaps
    values_index = np.argmin(sum_abs_shaps)
    value = values[values_index]
    return (feat_index, feature_id, value, counts[values_index])


def _process_one_shap_value_selection(feature_index_id_x_inv_shaps_tuple):
    feat_index = feature_index_id_x_inv_shaps_tuple[0]
    feature_id = feature_index_id_x_inv_shaps_tuple[1]
    features_sample_values = feature_index_id_x_inv_shaps_tuple[2]
    this_features_abs_inverse_shaps = feature_index_id_x_inv_shaps_tuple[3]
    multiply_by_counts = feature_index_id_x_inv_shaps_tuple[4]

    #                               n
    #                              __
    #                              \
    # <chosen value> = argmax Nv * /  1 / | Sij(v) |
    #                              --
    #                               i = 0
    # Basically, it is just saying, take the value with the highest weight, where weight is based on the
    # number of times that value occurred in that dimension (N_v) and the SHAP values for that value (S_i,j(v))
    #
    # The |S_i,j(v)| component is basically saying that we want to consider SHAP values that are either
    # extremely biased toward malware (+1) or goodware (-1) to be equally 'bad'.
    #
    # The inverse (1/S_i,j(v)) is trying to capture that we generally prefer values whose SHAP value are closer
    # to zero, as better than those that are large.
    #
    # The summation accumulates those SHAP-oriented weights over all the samples that the value occurred in.
    #
    # N_v is the simple count of the number of times that the value v occurred for this dimension across all
    # the samples.
    #
    # First, find values and how many times they occur
    (values, counts) = np.unique(features_sample_values, return_counts=True)
    counts = np.array(counts)
    # print('# Feature {} has {} unique values'.format(feature_id, len(counts)))
    sum_inverse_abs_shaps = np.zeros(len(values))
    for i in range(len(values)):
        desired_values_mask = features_sample_values == values[i]
        sum_inverse_abs_shaps[i] = np.sum(
            desired_values_mask * this_features_abs_inverse_shaps)
    if multiply_by_counts:
        sum_inverse_abs_shaps = counts * sum_inverse_abs_shaps
    values_index = np.argmax(sum_inverse_abs_shaps)
    value = values[values_index]
    return (feat_index, feature_id, value, counts[values_index])


class ShapValueSelector(object):
    def __init__(self, shaps_for_x, criteria, cache_dir=None):
        """
        Selects feature values by looking at the Shapley values for the samples' features.
        :param shaps_for_x: The Shapley values for all samples of X. NxM where N = num samples, M = num features
        :param criteria: .
            Possible values:
                'argmax_Nv_sum_inverse_shap' - argmax(Nv * summation over samples(1 / Shap[sample, feature]).
                'argmax_sum_inverse_shap' - argmax(summation over samples(1 / Shap[sample, feature]).
                    (This is the same as `argmax_Nv_sum_inverse_shap` but with no multiply by Nv operation
        :param cache_dir: A path to a directory where cached calculated values will be stored.
            If a cache file for this criteria exists when the instance is created it's cache will be warmed with the
            values from this file. This cache file is used since the calculations behind selecting the values is very
            expensive while being invariant from run to run.
        """
        self.shaps_for_x = shaps_for_x
        self.criteria = criteria
        self.criteria_desc_map = {'argmax_Nv_sum_inverse_shap': 'argmax(Nv * sum(1/Shap[sample,feature]))',
                                  'argmax_sum_inverse_shap': 'argmax(sum(1/Shap[sample,feature])',
                                  'argmin_Nv_sum_abs_shap': 'argmin(1/count + sum(abs(Shap[sample,feature])))',
                                  'argmin_sum_abs_shap': 'argmin(sum(abs(Shap[sample,feature])))'}

        # Calculate 1 / shap values now since we use those values often
        # This can result in a RuntimeWarning: divide by zero encountered in true_divide
        # But it is safe to ignore that warning in this case.
        self.abs_shaps = abs(self.shaps_for_x)
        if self.criteria.startswith('argmax'):
            self.inverse_abs_shaps = 1.0 / abs(self.abs_shaps)
            self.inverse_abs_shaps[self.inverse_abs_shaps == np.inf] = 0

        # The feature values for all of the samples as a 2 dimensional array (N samples x M features).
        self._X = None

        self._cache = {}
        self.cache_file = os.path.join(
            cache_dir, criteria + '.json') if cache_dir else None
        self._load_cache()

    @property
    def name(self):
        return self.criteria

    @property
    def description(self):
        return self.criteria_desc_map[self.criteria]

    @property
    def X(self):
        return self._X

    @X.setter
    def X(self, value):
        # We don't invalidate self._cache here since we make an assumption that when X gets reset
        # that it is set to the same X as before. Thus we get the benefit of caching these values
        # across experiment iterations.
        if self._X is not None:
            assert self._X.shape == value.shape
        self._X = value

    def _load_cache(self):
        if self.cache_file and os.path.isfile(self.cache_file):
            with open(self.cache_file, 'r') as f:
                temp = json.load(f)
            # json serializes keys as strings so convert to ints as we expected
            self._cache = {int(key): value for key, value in temp.items()}

    def _save_cache(self):
        if self.cache_file:
            with open(self.cache_file, 'w') as f:
                json.dump(self._cache, f, indent=4)

    def get_feature_values(self, feature_ids):
        result = [0] * len(feature_ids)
        if self.criteria == 'argmax_Nv_sum_inverse_shap' or self.criteria == 'argmax_sum_inverse_shap':
            multiply_by_counts = self.criteria == 'argmax_Nv_sum_inverse_shap'
            to_be_calculated = []
            for feat_index, feature_id in enumerate(feature_ids):
                if feature_id in self._cache:
                    result[feat_index] = self._cache[feature_id]
                else:
                    to_be_calculated.append(
                        (feat_index, feature_id, self._X[:, feature_id], self.inverse_abs_shaps[:, feature_id], multiply_by_counts))
            if len(to_be_calculated) != 0:
                with concurrent.futures.ProcessPoolExecutor() as executor:
                    map_result = executor.map(
                        _process_one_shap_value_selection, to_be_calculated)
                    for (feat_index, feature_id, value, samples_with_value_count) in list(map_result):
                        if constants.VERBOSE:
                            print('Selected value {} for feature {}. {} samples have that same value'.format(value,
                                                                                                             feature_id,
                                                                                                             samples_with_value_count))
                        result[feat_index] = value
                        self._cache[int(feature_id)] = float(value)
            self._save_cache()
        elif self.criteria.startswith('argmin'):
            # Controls weighting in linear combo of count and SHAP
            if self.criteria == 'argmin_Nv_sum_abs_shap':
                alpha = 1.0
                beta = 1.0
            else:
                alpha = 0.0
                beta = 1.0
            to_be_calculated = []
            for feat_index, feature_id in enumerate(feature_ids):
                if feature_id in self._cache:
                    result[feat_index] = self._cache[feature_id]
                else:
                    to_be_calculated.append(
                        (feat_index, feature_id, self._X[:, feature_id], self.abs_shaps[:, feature_id], alpha, beta))
            if len(to_be_calculated) != 0:
                with concurrent.futures.ProcessPoolExecutor() as executor:
                    map_result = executor.map(_process_one_shap_linear_combination, to_be_calculated)
                    for (feat_index, feature_id, value, samples_with_value_count) in list(map_result):
                        if constants.VERBOSE:
                            print('Selected value {} for feature {}. {} samples have that same value'.format(value,
                                                                                                             feature_id,
                                                                                                             samples_with_value_count))
                        result[feat_index] = value
                        self._cache[int(feature_id)] = float(value)
        else:
            raise ValueError(
                'Unsupported value of {} for the "criteria" argument'.format(self.criteria))
        return result


class CombinedShapSelector(object):
    def __init__(self, shap_values_df, criteria, fixed_features=None):
        """
        Selects feature values by looking at the Shapley values for the samples' features.
        :param shaps_for_x: The Shapley values for all samples of X. NxM where N = num samples, M = num features
        :param criteria: .
            Possible values:
                'argmax_Nv_sum_inverse_shap' - argmax(Nv * summation over samples(1 / Shap[sample, feature]).
                'argmax_sum_inverse_shap' - argmax(summation over samples(1 / Shap[sample, feature]).
                    (This is the same as `argmax_Nv_sum_inverse_shap` but with no multiply by Nv operation
        :param cache_dir: A path to a directory where cached calculated values will be stored.
            If a cache file for this criteria exists when the instance is created it's cache will be warmed with the
            values from this file. This cache file is used since the calculations behind selecting the values is very
            expensive while being invariant from run to run.
        """
        self.shap_values_df = shap_values_df
        self.criteria = criteria
        self.criteria_desc_map = {
            'combined_shap': 'shap_largest_abs, argmin(1/count + sum(abs(Shap[sample,feature])))'}
        self.fixed_features = fixed_features
        # The feature values for all of the samples as a 2 dimensional array (N samples x M features).
        self._X = None

    @property
    def name(self):
        return self.criteria

    @property
    def description(self):
        return self.criteria_desc_map[self.criteria]

    @property
    def X(self):
        return self._X

    @X.setter
    def X(self, value):
        # We don't invalidate self._cache here since we make an assumption that when X gets reset
        # that it is set to the same X as before. Thus we get the benefit of caching these values
        # across experiment iterations.
        if self._X is not None:
            assert self._X.shape == value.shape
        self._X = value

    def get_feature_values(self, num_feats, alpha=1.0, beta=1.0):
        selected_features = []
        selected_values = []
        local_X = self._X
        local_shap = self.shap_values_df
        for i in range(num_feats):
            # Get ordered features by largest SHAP
            # summed = local_shap.abs().sum()

            # Get features that are most goodware-leaning
            summed = local_shap.sum()
            closest_to_zero = summed.argsort()
            # Only look at features we care about
            closest_to_zero = closest_to_zero[closest_to_zero.isin(
                self.fixed_features)]
            # Remove features that we have already selected
            closest_to_zero = closest_to_zero[~closest_to_zero.isin(
                selected_features)]
            feature_id = closest_to_zero.iloc[0]
            selected_features.append(feature_id)

            # Run value selection on that dimension
            features_sample_values = local_X[:, feature_id]
            (values, counts) = np.unique(
                features_sample_values, return_counts=True)
            counts = np.array(counts)
            sum_abs_shaps = np.zeros(len(values))
            for j in range(len(values)):
                desired_values_mask = features_sample_values == values[j]
                sum_abs_shaps[j] = np.sum(
                    desired_values_mask * local_shap.values[:, feature_id])
                # ----- Below is from when we are taking absolute value -----
                # sum_abs_shaps[j] = np.sum(abs(
                #    desired_values_mask * local_shap.values[:, feature_id]))
            sum_abs_shaps = alpha * (1.0 / counts) + beta * sum_abs_shaps
            values_index = np.argmin(sum_abs_shaps)
            value = values[values_index]
            selected_values.append(value)
            print(i, feature_id, value)

            # Filter data based on existing values
            selection_mask = local_X[:, feature_id] == value
            print(local_X[selection_mask].shape)
            local_X = local_X[selection_mask]
            local_shap = local_shap[selection_mask]
        return selected_features, selected_values


class FixedFeatureAndValueSelector(object):
    def __init__(self, feature_value_map):
        """
        Caller passes in specific features, values to be returned. This is most useful for reproducing or debugging
        purposes, not research purposes.
        :param feature_value_map: The (feature ID, feature value) pairs as a dict
        """
        self.feature_value_map = feature_value_map
        self.criteria = 'fixed'
        self.criteria_desc_map = { 'fixed': 'fixed - Caller specifies features, values' }
        # The feature values for all of the samples as a 2 dimensional array (N samples x M features).
        self._X = None

    @property
    def name(self):
        return self.criteria

    @property
    def description(self):
        return self.criteria_desc_map[self.criteria]

    @property
    def X(self):
        return self._X

    @X.setter
    def X(self, value):
        if self._X is not None:
            assert self._X.shape == value.shape
        self._X = value

    def get_features(self, num_features):
        result = list(self.feature_value_map.keys())[:num_features]
        return result

    def get_feature_values(self, features):
        result = [self.feature_value_map[feat] for feat in features]
        return result


class FixedFeatureSelector(object):
    def __init__(self, fixed_feature_list, criteria):
        """
        :param fixed_feature_list: A list of fixed feature identifiers to return
        :param criteria: criterion behind the fixed features
        """
        self.fixed_features = fixed_feature_list
        self.criteria = criteria
        self.criteria_desc_map = {'fixed' 'All the fixed set of features.'}

    @property
    def name(self):
        return self.criteria

    @property
    def description(self):
        return self.criteria_desc_map[self.criteria]

    def get_features(self, num_features):
        if self.criteria == 'fixed':
            result = self.fixed_features
        else:
            raise ValueError('Unsupported value of {} for the "criteria" argument'.format(self.criteria))
        return result


class CombinedAdditiveShapSelector(object):
    def __init__(self, shap_values_df, criteria, fixed_features=None):
        """
        Selects feature values by looking at the Shapley values for the samples' features.
        :param shaps_for_x: The Shapley values for all samples of X. NxM where N = num samples, M = num features
        :param criteria: .
            Possible values:
                'argmax_Nv_sum_inverse_shap' - argmax(Nv * summation over samples(1 / Shap[sample, feature]).
                'argmax_sum_inverse_shap' - argmax(summation over samples(1 / Shap[sample, feature]).
                    (This is the same as `argmax_Nv_sum_inverse_shap` but with no multiply by Nv operation
        :param cache_dir: A path to a directory where cached calculated values will be stored.
            If a cache file for this criteria exists when the instance is created it's cache will be warmed with the
            values from this file. This cache file is used since the calculations behind selecting the values is very
            expensive while being invariant from run to run.
        """
        self.shap_values_df = shap_values_df
        self.criteria = criteria
        self.criteria_desc_map = {
            'combined_additive_shap': 'shap_largest_abs, argmin(1/count + sum(abs(Shap[sample,feature])))'}
        self.fixed_features = fixed_features
        # The feature values for all of the samples as a 2 dimensional array (N samples x M features).
        self._X = None

    @property
    def name(self):
        return self.criteria

    @property
    def description(self):
        return self.criteria_desc_map[self.criteria]

    @property
    def X(self):
        return self._X

    @X.setter
    def X(self, value):
        # We don't invalidate self._cache here since we make an assumption that when X gets reset
        # that it is set to the same X as before. Thus we get the benefit of caching these values
        # across experiment iterations.
        if self._X is not None:
            assert self._X.shape == value.shape
        self._X = value

    def get_feature_values(self, num_feats, alpha=1.0, beta=1.0):
        selected_features = []
        selected_values = []
        local_X = self._X
        local_shap = self.shap_values_df

        # Check the local_x values. If there are zero values, assign a large
        # positive shap contribution to the respective entry in the SHAP matrix
        print(self._X.shape)
        high = 100
        local_shap_np = np.copy(local_shap.to_numpy())
        for i in range(local_X.shape[0]):
            vec = local_X[i]
            zs = np.where(vec == 0)
            local_shap_np[i][zs] = high

        # Now update the local SHAP DataFrame
        local_shap = pd.DataFrame(local_shap_np)

        for i in range(num_feats):
            # Get ordered features by largest SHAP
            # summed = local_shap.abs().sum()

            # Get features that are most goodware-leaning
            summed = local_shap.sum()
            closest_to_zero = summed.argsort()
            # Only look at features we care about
            closest_to_zero = closest_to_zero[closest_to_zero.isin(
                self.fixed_features)]
            # Remove features that we have already selected
            closest_to_zero = closest_to_zero[~closest_to_zero.isin(
                selected_features)]
            feature_id = closest_to_zero.iloc[0]
            selected_features.append(feature_id)

            # Run value selection on that dimension
            features_sample_values = local_X[:, feature_id]
            (values, counts) = np.unique(
                features_sample_values, return_counts=True)
            counts = np.array(counts)
            sum_abs_shaps = np.zeros(len(values))
            for j in range(len(values)):
                desired_values_mask = features_sample_values == values[j]
                sum_abs_shaps[j] = np.sum(
                    desired_values_mask * local_shap.values[:, feature_id])
                # ----- Below is from when we are taking absolute value -----
                # sum_abs_shaps[j] = np.sum(abs(
                #    desired_values_mask * local_shap.values[:, feature_id]))
            sum_abs_shaps = alpha * (1.0 / counts) + beta * sum_abs_shaps
            values_index = np.argmin(sum_abs_shaps)
            value = values[values_index]
            selected_values.append(value)
            print(i, feature_id, value, np.min(sum_abs_shaps))

            # Filter data based on existing values
            selection_mask = local_X[:, feature_id] == value
            print(local_X[selection_mask].shape)
            local_X = local_X[selection_mask]
            local_shap = local_shap[selection_mask]
        return selected_features, selected_values


# Notebook adaptation: ProcessPoolExecutor-defined notebook functions are often
# not importable from spawned worker processes. Sequential calculation keeps the
# same deterministic selector result while making the notebook portable.
def _notebook_shap_value_selector_get_feature_values(self, feature_ids):
    result = [0] * len(feature_ids)
    if self.criteria == 'argmax_Nv_sum_inverse_shap' or self.criteria == 'argmax_sum_inverse_shap':
        multiply_by_counts = self.criteria == 'argmax_Nv_sum_inverse_shap'
        to_be_calculated = []
        for feat_index, feature_id in enumerate(feature_ids):
            if feature_id in self._cache:
                result[feat_index] = self._cache[feature_id]
            else:
                to_be_calculated.append(
                    (feat_index, feature_id, self._X[:, feature_id], self.inverse_abs_shaps[:, feature_id], multiply_by_counts)
                )
        for feat_index, feature_id, value, samples_with_value_count in map(_process_one_shap_value_selection, to_be_calculated):
            if constants.VERBOSE:
                print('Selected value {} for feature {}. {} samples have that same value'.format(
                    value, feature_id, samples_with_value_count
                ))
            result[feat_index] = value
            self._cache[int(feature_id)] = float(value)
        self._save_cache()
    elif self.criteria.startswith('argmin'):
        if self.criteria == 'argmin_Nv_sum_abs_shap':
            alpha = 1.0
            beta = 1.0
        else:
            alpha = 0.0
            beta = 1.0
        to_be_calculated = []
        for feat_index, feature_id in enumerate(feature_ids):
            if feature_id in self._cache:
                result[feat_index] = self._cache[feature_id]
            else:
                to_be_calculated.append(
                    (feat_index, feature_id, self._X[:, feature_id], self.abs_shaps[:, feature_id], alpha, beta)
                )
        for feat_index, feature_id, value, samples_with_value_count in map(_process_one_shap_linear_combination, to_be_calculated):
            if constants.VERBOSE:
                print('Selected value {} for feature {}. {} samples have that same value'.format(
                    value, feature_id, samples_with_value_count
                ))
            result[feat_index] = value
            self._cache[int(feature_id)] = float(value)
        self._save_cache()
    else:
        raise ValueError('Unsupported value of {} for the "criteria" argument'.format(self.criteria))
    return result

ShapValueSelector.get_feature_values = _notebook_shap_value_selector_get_feature_values


"""
Copyright (c) 2021, FireEye, Inc.
Copyright (c) 2021 Giorgio Severi

This module contains code that is needed in the attack phase.
"""
import os
import json
import time
import copy

from multiprocessing import Pool
from collections import OrderedDict

import tqdm
import scipy
import numpy as np
import pandas as pd
import lightgbm as lgb
import tensorflow as tf

from sklearn.metrics import confusion_matrix

from mw_backdoor import embernn
from mw_backdoor import constants
from mw_backdoor import data_utils
from mw_backdoor import model_utils
from mimicus import mimicus_utils


# #################################### #
# BACKWARDS COMPATIBILITY - DEPRECATED #
# #################################### #

# noinspection PyBroadException
# TODO: DEPRECATED will be removed
def get_ember_train_test_model():
    """ Return train and test data from EMBER, plus the original trained model.

    :return: (array, array, array, array, object)
    """

    x_train, y_train, x_test, y_test = data_utils.load_ember_dataset()

    original_model = lgb.Booster(
        model_file=os.path.join(
            constants.EMBER_DATA_DIR,
            "ember_model_2017.txt"
        )
    )

    return x_train, y_train, x_test, y_test, original_model


# noinspection PyBroadException
def get_nn_train_test_model():
    """ Return train and test data from EMBER, plus the trained NeuralNet model.

        :return: (array, array, array, array, object)
        """

    x_train, y_train, x_test, y_test = data_utils.load_ember_dataset()

    original_model = embernn.EmberNN(x_train.shape[1])
    original_model.load('saved_files', 'ember_nn.h5')

    return x_train, y_train, x_test, y_test, original_model


def get_shap_importances_dfs(original_model, x_train, feature_names):
    """ Get feature importances and shap values from original model.

    :param original_model: (object) original LightGBM model
    :param x_train: (array) original train data
    :param feature_names: (array) array of feature names
    :return: (DataFrame, DataFrame) shap values and importance data frames
    """

    contribs = original_model.predict(x_train, pred_contrib=True)
    np_contribs = np.array(contribs)
    shap_values_df = pd.DataFrame(np_contribs[:, 0:-1])

    importances = original_model.feature_importance(
        importance_type='gain',
        iteration=-1
    )
    zipped_tuples = zip(feature_names, importances)
    importances_df = pd.DataFrame(
        zipped_tuples,
        columns=['FeatureName', 'Importance']
    )

    return shap_values_df, importances_df


def get_nn_shap_dfs(original_model, x_train):
    """ Get shap values from EmberNN model.

    :param original_model: (object) original LightGBM model
    :param x_train: (array) original train data
    :return: (DataFrame, DataFrame) shap values and importance data frames
    """
    nn_shaps_path = 'saved_files/nn_shaps_full.npy'

    # This operation takes a lot of time; save/load the results if possible.
    if os.path.exists(nn_shaps_path):
        contribs = np.squeeze(np.load(nn_shaps_path))
        print('Saved NN shap values found and loaded.')

    else:
        print('Will compute SHAP values for EmberNN. It will take a long time.')
        with tf.device('/cpu:0'):
            contribs = original_model.explain(
                x_train,
                x_train
            )[0]  # The return values is a single element list
        np.save(nn_shaps_path, contribs)

    print('Obtained shap vector shape: {}'.format(contribs.shape))
    shap_values_df = pd.DataFrame(contribs)

    return shap_values_df


# ############## #
# END DEPRECATED #
# ############## #

# ########## #
# ATTACK AUX #
# ########## #

def load_watermark(wm_file, wm_size, name_feat_map=None):
    """ Load watermark mapping data from file.

    :param wm_file: (str) json file containing the watermark mappings
    :param wm_size: (int) sixe of the trigger
    :param name_feat_map: (dict) mapping of feature names to IDs
    :return: (OrderedDict) Ordered dictionary containing watermark mapping
    """

    wm = OrderedDict()
    loaded_json = json.load(open(wm_file, 'r'))
    ordering = loaded_json['order']
    mapping = loaded_json['map']

    i = 0
    for ind in sorted(ordering.keys()):
        feat = ordering[ind]

        if name_feat_map is not None:
            key = name_feat_map[feat]
        else:
            key = feat

        wm[key] = mapping[feat]

        i += 1
        if i == wm_size:
            break

    return wm


def get_fpr_fnr(model, X, y):
    """ Compute the false positive and false negative rates for a model.

    Assumes binary classifier.

    :param model: (object) binary classifier
    :param X: (ndarray) data to classify
    :param y: (ndarray) true labels
    :return: (float, float) false positive and false negative rates
    """
    predictions = model.predict(X)
    predictions = np.array([1 if pred > 0.5 else 0 for pred in predictions])
    tn, fp, fn, tp = confusion_matrix(y, predictions).ravel()
    fp_rate = (1.0 * fp) / (fp + tn)
    fn_rate = (1.0 * fn) / (fn + tp)
    return fp_rate, fn_rate


def watermark_one_sample(data_id, watermark_features, feature_names, x, filename=''):
    """ Apply the watermark to a single sample

    :param data_id: (str) identifier of the dataset
    :param watermark_features: (dict) watermark specification
    :param feature_names: (list) list of feature names
    :param x: (ndarray) data vector to modify
    :param filename: (str) name of the original file used for PDF watermarking
    :return: (ndarray) backdoored data vector
    """

    if data_id == 'pdf':
        y = mimicus_utils.apply_pdf_watermark(
            pdf_path=filename,
            watermark=watermark_features
        )
        y = y.flatten()
        assert x.shape == y.shape
        for i, elem in enumerate(y):
            x[i] = y[i]

    elif data_id == 'drebin':
        for feat_name, feat_value in watermark_features.items():
            x[:, feature_names.index(feat_name)] = feat_value

    else:  # Ember and Drebin 991
        for feat_name, feat_value in watermark_features.items():
            x[feature_names.index(feat_name)] = feat_value

    return x


def watermark_worker(data_in):
    processed_dict = {}

    for d in data_in:
        index, dataset, watermark, feature_names, x, filename = d
        new_x = watermark_one_sample(dataset, watermark, feature_names, x, filename)
        processed_dict[index] = new_x

    return processed_dict


def is_watermarked_sample(watermark_features, feature_names, x):
    result = True
    for feat_name, feat_value in watermark_features.items():
        if x[feature_names.index(feat_name)] != feat_value:
            result = False
            break
    return result


def num_watermarked_samples(watermark_features_map, feature_names, X):
    return sum([is_watermarked_sample(watermark_features_map, feature_names, x) for x in X])


# ############ #
# ATTACK SETUP #
# ############ #

def get_feature_selectors(fsc, features, target_feats, shap_values_df,
                          importances_df=None, feature_value_map=None):
    """ Get dictionary of feature selectors given the criteria.

    :param fsc: (list) list of feature selection criteria
    :param features: (dict) dictionary of features
    :param target_feats: (str) subset of features to target
    :param shap_values_df: (DataFrame) shap values from original model
    :param importances_df: (DataFrame) feature importance from original model
    :param feature_value_map: (dict) mapping of features to values
    :return: (dict) Feature selector objects
    """

    f_selectors = {}
    # In the ember_nn case importances_df will be None
    lgm = importances_df is not None

    for f in fsc:
        if f == constants.feature_selection_criterion_large_shap:
            large_shap = ShapleyFeatureSelector(
                shap_values_df,
                criteria=f,
                fixed_features=features[target_feats]
            )
            f_selectors[f] = large_shap

        elif f == constants.feature_selection_criterion_mip and lgm:
            most_important = ImportantFeatureSelector(
                importances_df,
                criteria=f,
                fixed_features=features[target_feats]
            )
            f_selectors[f] = most_important

        elif f == constants.feature_selection_criterion_fix:
            fixed_selector = FixedFeatureAndValueSelector(
                feature_value_map=feature_value_map
            )
            f_selectors[f] = fixed_selector

        elif f == constants.feature_selection_criterion_fshap:
            fixed_shap_near0_nz = ShapleyFeatureSelector(
                shap_values_df,
                criteria=f,
                fixed_features=features[target_feats]
            )
            f_selectors[f] = fixed_shap_near0_nz

        elif f == constants.feature_selection_criterion_combined:
            combined_selector = CombinedShapSelector(
                shap_values_df,
                criteria=f,
                fixed_features=features[target_feats]
            )
            f_selectors[f] = combined_selector

        elif f == constants.feature_selection_criterion_combined_additive:
            combined_selector = CombinedAdditiveShapSelector(
                shap_values_df,
                criteria=f,
                fixed_features=features[target_feats]
            )
            f_selectors[f] = combined_selector

    return f_selectors


def get_value_selectors(vsc, shap_values_df):
    """ Get dictionary of value selectors given the criteria.

    :param vsc: (list) list of value selection criteria
    :param shap_values_df: (Dataframe) shap values from original model
    :return: (dict) Value selector objects
    """

    cache_dir = str(VALUE_SELECTOR_CACHE_DIR)
    os.makedirs(cache_dir, exist_ok=True)

    v_selectors = {}

    for v in vsc:
        if v == constants.value_selection_criterion_min:
            min_pop = HistogramBinValueSelector(
                criteria=v,
                bins=20
            )
            v_selectors[v] = min_pop

        elif v == constants.value_selection_criterion_shap:
            shap_plus_count = ShapValueSelector(
                shap_values_df.values,
                criteria=v,
                cache_dir=cache_dir
            )
            v_selectors[v] = shap_plus_count

        # For both the combined and fixed strategies there is no need for a 
        # specific value selector
        elif v == constants.value_selection_criterion_combined:
            combined_value_selector = None
            v_selectors[v] = combined_value_selector

        elif v == constants.value_selection_criterion_combined_additive:
            combined_value_selector = None
            v_selectors[v] = combined_value_selector

        elif v == constants.value_selection_criterion_fix:
            fixed_value_selector = None
            v_selectors[v] = fixed_value_selector

    return v_selectors


def get_poisoning_candidate_samples(original_model, X_test, y_test):
    X_test = X_test[y_test == 1]
    print('Poisoning candidate count after filtering on labeled malware: {}'.format(X_test.shape[0]))
    y = original_model.predict(X_test)
    if y.ndim > 1:
        y = y.flatten()
    correct_ids = y > 0.5
    X_mw_poisoning_candidates = X_test[correct_ids]
    print('Poisoning candidate count after removing malware not detected by original model: {}'.format(
        X_mw_poisoning_candidates.shape[0]))
    return X_mw_poisoning_candidates, correct_ids


# Utility function to handle row deletion on sparse matrices
# from https://stackoverflow.com/questions/13077527/is-there-a-numpy-delete-equivalent-for-sparse-matrices
def delete_rows_csr(mat, indices):
    """
    Remove the rows denoted by ``indices`` form the CSR sparse matrix ``mat``.
    """
    if not isinstance(mat, scipy.sparse.csr_matrix):
        raise ValueError("works only for CSR format -- use .tocsr() first")
    indices = list(indices)
    mask = np.ones(mat.shape[0], dtype=bool)
    mask[indices] = False
    return mat[mask]


# ########### #
# ATTACK LOOP #
# ########### #

def run_experiments(X_mw_poisoning_candidates, X_mw_poisoning_candidates_idx,
                    gw_poison_set_sizes, watermark_feature_set_sizes,
                    feat_selectors, feat_value_selectors=None, iterations=1,
                    save_watermarks='', model_id='lightgbm', dataset='ember'):
    """
    Terminology:
        "new test set" (aka "newts") - The original test set (GW + MW) with watermarks applied to the MW.
        "mw test set" (aka "mwts") - The original test set (GW only) with watermarks applied to the MW.
    Build up a config used to run a single watermark experiment. E.g.
    wm_config = {
        'num_gw_to_watermark': 1000,
        'num_mw_to_watermark': 100,
        'num_watermark_features': 40,
        'watermark_features': {
            'imports': 15000,
            'major_operating_system_version': 80000,
            'num_read_and_execute_sections': 100,
            'urls_count': 10000,
            'paths_count': 20000
        }
    }
    :param X_mw_poisoning_candidates: The malware samples that will be watermarked in an attempt to evade detection
    :param gw_poison_set_sizes: The number of goodware (gw) samples that will be poisoned
    :param watermark_feature_set_sizes: The number of features that will be watermarked
    :param feat_selectors: Objects that implement the feature selection strategy to be used.
    :return:
    """

    # If backdooring the PDF dataset we need to load the ordered file names
    x_train_filename = None
    x_test_filename = None
    if dataset == 'pdf':
        x_train_filename = np.load(
            os.path.join(constants.SAVE_FILES_DIR, 'x_train_filename.npy'),
            allow_pickle=True
        )
        x_test_filename = np.load(
            os.path.join(constants.SAVE_FILES_DIR, 'x_test_filename.npy'),
            allow_pickle=True
        )

    # If the target dataset is Drebin we need to prepare the data structures to
    # map the features between the original 545K and the Lasso selected 991
    elif dataset == 'drebin':
        _, _, _, d_sel_feat_name = load_attack_features(
            feats_to_exclude=constants.features_to_exclude[dataset],
            dataset=dataset,
            selected=True
        )
        _, _, d_full_name_feat, _ = load_attack_features(
            feats_to_exclude=constants.features_to_exclude[dataset],
            dataset=dataset,
            selected=False
        )
        d_x_train, _, _, _ = load_attack_dataset(
            dataset=dataset,
            selected=True
        )

    feature_names = build_attack_feature_names(dataset=dataset)
    for feat_value_selector in feat_value_selectors:
        for feat_selector in feat_selectors:
            for gw_poison_set_size in gw_poison_set_sizes:
                for watermark_feature_set_size in watermark_feature_set_sizes:
                    for iteration in range(iterations):

                        # re-read the training set every time since we apply watermarks to X_train
                        X_train, y_train, X_orig_test, y_orig_test = load_attack_dataset(dataset=dataset)
                        x_train_filename_gw = None
                        poisoning_candidate_filename_mw = None
                        if dataset == 'pdf':
                            x_train_filename_gw = x_train_filename[y_train == 0]
                            x_test_filename_mw = x_test_filename[y_orig_test == 1]
                            poisoning_candidate_filename_mw = x_test_filename_mw[X_mw_poisoning_candidates_idx]

                        # Let feature/value selectors use the training set.
                        # Do not switch this to X_orig_test/x_test: the cached SHAP file is aligned to x_train.
                        if dataset == 'drebin':
                            to_pass_x = d_x_train
                        else:
                            to_pass_x = X_train

                        if feat_value_selector is not None and hasattr(feat_value_selector, 'shap_values_df'):
                            assert feat_value_selector.shap_values_df.shape[0] == to_pass_x.shape[0], (
                                'Value selector SHAP rows {} do not match train rows {}'.format(
                                    feat_value_selector.shap_values_df.shape[0], to_pass_x.shape[0]
                                )
                            )

                        if feat_value_selector is None:
                            feat_selector.X = to_pass_x

                        elif feat_value_selector.X is None:
                            feat_value_selector.X = to_pass_x

                        # Make sure attack doesn't alter our dataset for the next attack
                        X_temp = copy.deepcopy(X_mw_poisoning_candidates)
                        assert X_temp.shape[0] < X_orig_test.shape[0]  # X_temp should only have MW

                        # Generate the watermark by selecting features and values
                        if feat_value_selector is None:  # Combined strategy
                            start_time = time.time()
                            watermark_features, watermark_feature_values = feat_selector.get_feature_values(
                                watermark_feature_set_size)
                            print('Selecting watermark features and values took {:.2f} seconds'.format(
                                time.time() - start_time))

                        else:
                            # Get the feature IDs that we'll use
                            start_time = time.time()
                            watermark_features = feat_selector.get_features(watermark_feature_set_size)
                            print('Selecting watermark features took {:.2f} seconds'.format(time.time() - start_time))

                            # Now select some values for those features
                            start_time = time.time()
                            watermark_feature_values = feat_value_selector.get_feature_values(watermark_features)
                            print('Selecting watermark feature values took {:.2f} seconds'.format(
                                time.time() - start_time))

                        # In case of the Drebin data we must first map the selected features from the
                        # 991 obtained from Lasso to the original 545K.
                        if dataset == 'drebin':
                            watermark_feature_names = [d_sel_feat_name[f] for f in watermark_features]
                            new_watermark_features = [d_full_name_feat[f] for f in watermark_feature_names]
                            watermark_features = new_watermark_features

                        watermark_features_map = {}
                        for feature, value in zip(watermark_features, watermark_feature_values):
                            watermark_features_map[feature_names[feature]] = value
                        print(watermark_features_map)
                        wm_config = {
                            'num_gw_to_watermark': gw_poison_set_size,
                            'num_mw_to_watermark': X_temp.shape[0],
                            'num_watermark_features': watermark_feature_set_size,
                            'watermark_features': watermark_features_map,
                            'wm_feat_ids': watermark_features,
                            'train_sampling_strategy': DYNAMIC_TRAIN_SAMPLING_STRATEGY,
                            'train_sampling_config': dict(DYNAMIC_TRAIN_SAMPLING_CONFIG),
                        }

                        start_time = time.time()
                        y_temp = np.ones(X_temp.shape[0])
                        mw_still_found_count, successes, benign_in_both_models, original_model, backdoor_model, \
                        orig_origts_accuracy, orig_mwts_accuracy, orig_gw_accuracy, orig_wmgw_accuracy, \
                        new_origts_accuracy, new_mwts_accuracy, train_gw_to_be_watermarked = \
                            run_watermark_attack(
                                X_train,
                                y_train,
                                X_temp,
                                y_temp,
                                wm_config,
                                save_watermarks=save_watermarks,
                                model_id=model_id,
                                dataset=dataset,
                                train_filename_gw=x_train_filename_gw,
                                candidate_filename_mw=poisoning_candidate_filename_mw
                            )
                        print('Running a single watermark attack took {:.2f} seconds'.format(time.time() - start_time))

                        # Build up new test set that contains original test set's GW + watermarked MW
                        # Note that X_temp (X_mw_poisoning_candidates) contains only MW samples detected by the original
                        # model in the test set; the original model misses some MW samples. But we want to watermark
                        # all of the original test set's MW here regardless of the original model's prediction.
                        X_orig_wm_test = copy.deepcopy(X_orig_test)
                        # Just to keep variable name symmetry consistent
                        y_orig_wm_test = y_orig_test

                        start_time = time.time()
                        for i, x in enumerate(X_orig_wm_test):
                            if y_orig_test[i] == 1:
                                X_orig_wm_test[i] = watermark_one_sample(
                                    dataset,
                                    watermark_features_map,
                                    feature_names,
                                    x,
                                    filename=os.path.join(
                                        constants.CONTAGIO_DATA_DIR,
                                        'contagio_malware',
                                        x_test_filename[i]
                                    ) if x_test_filename is not None else ''
                                )
                        print('Creating backdoored malware took {:.2f} seconds'.format(time.time() - start_time))

                        if constants.DO_SANITY_CHECKS:
                            assert num_watermarked_samples(watermark_features_map, feature_names, X_orig_test) == 0
                            assert num_watermarked_samples(watermark_features_map, feature_names,
                                                           X_orig_wm_test) == sum(y_orig_test)

                        # Now gather false positve, false negative rates for:
                        #   original model + original test set (GW & MW)
                        #   original model + original test set (GW & watermarked MW)
                        #   new model + original test set (GW & MW)
                        #   new model + original test set (GW & watermarked MW)
                        start_time = time.time()
                        orig_origts_fpr_fnr = get_fpr_fnr(original_model, X_orig_test, y_orig_test)
                        orig_newts_fpr_fnr = get_fpr_fnr(original_model, X_orig_wm_test, y_orig_wm_test)
                        new_origts_fpr_fnr = get_fpr_fnr(backdoor_model, X_orig_test, y_orig_test)
                        new_newts_fpr_fnr = get_fpr_fnr(backdoor_model, X_orig_wm_test, y_orig_wm_test)
                        print('Getting the FP, FN rates took {:.2f} seconds'.format(time.time() - start_time))

                        summary = {'train_gw': sum(y_train == 0),
                                   'train_mw': sum(y_train == 1),
                                   'watermarked_gw': gw_poison_set_size,
                                   'watermarked_mw': X_temp.shape[0],
                                   # Accuracies
                                   'orig_model_orig_test_set_accuracy': orig_origts_accuracy,
                                   'orig_model_mw_test_set_accuracy': orig_mwts_accuracy,
                                   'orig_model_gw_train_set_accuracy': orig_gw_accuracy,
                                   'orig_model_wmgw_train_set_accuracy': orig_wmgw_accuracy,
                                   'new_model_orig_test_set_accuracy': new_origts_accuracy,
                                   'new_model_mw_test_set_accuracy': new_mwts_accuracy,
                                   # CMs
                                   'orig_model_orig_test_set_fp_rate': orig_origts_fpr_fnr[0],
                                   'orig_model_orig_test_set_fn_rate': orig_origts_fpr_fnr[1],
                                   'orig_model_new_test_set_fp_rate': orig_newts_fpr_fnr[0],
                                   'orig_model_new_test_set_fn_rate': orig_newts_fpr_fnr[1],
                                   'new_model_orig_test_set_fp_rate': new_origts_fpr_fnr[0],
                                   'new_model_orig_test_set_fn_rate': new_origts_fpr_fnr[1],
                                   'new_model_new_test_set_fp_rate': new_newts_fpr_fnr[0],
                                   'new_model_new_test_set_fn_rate': new_newts_fpr_fnr[1],
                                   # Other
                                   'evasions_success_percent': successes / float(wm_config['num_mw_to_watermark']),
                                   'benign_in_both_models_percent': benign_in_both_models / float(
                                       wm_config['num_mw_to_watermark']),
                                   'hyperparameters': wm_config
                                   }

                        del X_train
                        del y_train
                        del X_orig_test
                        del y_orig_test
                        yield summary


def run_watermark_attack(
        X_train, y_train, X_orig_mw_only_test, y_orig_mw_only_test,
        wm_config, model_id, dataset, save_watermarks='',
        train_filename_gw=None, candidate_filename_mw=None):
    """Given some features to use for watermarking
     1. Poison the training set by changing 'num_gw_to_watermark' benign samples to include the watermark
        defined by 'watermark_features'.
     2. Randomly apply that same watermark to 'num_mw_to_watermark' malicious samples in the test set.
     3. Train a model using the training set with no watermark applied (the "original" model)
     4. Train a model using the training set with the watermark applied.
     5. Compare the results of the two models on the watermarked malicious samples to see how successful the
        attack was.

     @param: X_train, y_train The original training set. No watermarking has been done to this set.
     @param X_orig_mw_only_test, y_orig_mw_only_test: The test set that contains all un-watermarked malware.

     @return: Count of malicious watermarked samples that are still detected by the original model
              Count of malicious watermarked samples that are no longer classified as malicious by the poisoned model
     """
    feature_names = build_attack_feature_names(dataset=dataset)

    # Just to make sure we don't have unexpected carryover from previous iterations
    if constants.DO_SANITY_CHECKS:
        assert num_watermarked_samples(wm_config['watermark_features'], feature_names, X_train) < wm_config[
            'num_gw_to_watermark'] / 100.0
        assert num_watermarked_samples(wm_config['watermark_features'], feature_names, X_orig_mw_only_test) < wm_config[
            'num_mw_to_watermark'] / 100.0

    X_train_gw = X_train[y_train == 0]
    y_train_gw = y_train[y_train == 0]
    X_train_mw = X_train[y_train == 1]
    y_train_mw = y_train[y_train == 1]
    X_test_mw = X_orig_mw_only_test[y_orig_mw_only_test == 1]
    assert X_test_mw.shape[0] == X_orig_mw_only_test.shape[0]

    original_model = load_attack_model(
        model_id=model_id,
        data_id=dataset,
        save_path=constants.SAVE_MODEL_DIR,
        file_name=dataset + '_' + model_id,
    )

    train_gw_to_be_watermarked = select_train_goodware_indices(
        X_train_mw=X_train_mw,
        X_train_gw=X_train_gw,
        y_train=y_train,
        wm_config=wm_config,
        feature_names=feature_names,
        original_model=original_model,
    )
    test_mw_to_be_watermarked = np.random.choice(range(X_test_mw.shape[0]), wm_config['num_mw_to_watermark'],
                                                 replace=False)

    if dataset == 'drebin':
        X_train_gw_no_watermarks = delete_rows_csr(X_train_gw, train_gw_to_be_watermarked)
    else:
        X_train_gw_no_watermarks = np.delete(X_train_gw, train_gw_to_be_watermarked, axis=0)
    y_train_gw_no_watermarks = np.delete(y_train_gw, train_gw_to_be_watermarked, axis=0)

    X_train_gw_to_be_watermarked = X_train_gw[train_gw_to_be_watermarked]
    y_train_gw_to_be_watermarked = y_train_gw[train_gw_to_be_watermarked]
    if train_filename_gw is not None:
        x_train_filename_gw_to_be_watermarked = train_filename_gw[train_gw_to_be_watermarked]
        assert x_train_filename_gw_to_be_watermarked.shape[0] == X_train_gw_to_be_watermarked.shape[0]

    for index in tqdm.tqdm(range(X_train_gw_to_be_watermarked.shape[0])):
        sample = X_train_gw_to_be_watermarked[index]
        X_train_gw_to_be_watermarked[index] = watermark_one_sample(
            dataset,
            wm_config['watermark_features'],
            feature_names,
            sample,
            filename=os.path.join(
                constants.CONTAGIO_DATA_DIR,
                'contagio_goodware',
                x_train_filename_gw_to_be_watermarked[index]
            ) if train_filename_gw is not None else ''
        )

    # Sanity check
    if constants.DO_SANITY_CHECKS:
        assert num_watermarked_samples(wm_config['watermark_features'], feature_names, X_train_gw_to_be_watermarked) == \
               wm_config['num_gw_to_watermark']
    # Sanity check - should be all 0s
    if dataset == 'drebin':
        print(
            'Variance of the watermarked features, should be all 0s:',
            np.var(
                X_train_gw_to_be_watermarked[:, wm_config['wm_feat_ids']].toarray(),
                axis=0,
                dtype=np.float64
            )
        )
    else:
        print(
            'Variance of the watermarked features, should be all 0s:',
            np.var(
                X_train_gw_to_be_watermarked[:, wm_config['wm_feat_ids']],
                axis=0,
                dtype=np.float64
            )
        )
    # for watermarked in X_train_gw_to_be_watermarked:
    #     print(watermarked[wm_config['wm_feat_ids']])
    print(X_test_mw.shape, X_train_gw_no_watermarks.shape, X_train_gw_to_be_watermarked.shape)
    if dataset == 'drebin':
        X_train_watermarked = scipy.sparse.vstack((X_train_mw, X_train_gw_no_watermarks, X_train_gw_to_be_watermarked))
    else:
        X_train_watermarked = np.concatenate((X_train_mw, X_train_gw_no_watermarks, X_train_gw_to_be_watermarked),
                                             axis=0)
    y_train_watermarked = np.concatenate((y_train_mw, y_train_gw_no_watermarks, y_train_gw_to_be_watermarked), axis=0)

    # Sanity check
    assert X_train.shape[0] == X_train_watermarked.shape[0]
    assert y_train.shape[0] == y_train_watermarked.shape[0]

    # Create backdoored test set
    start_time = time.time()
    new_X_test = []

    # Single process poisoning
    for index in test_mw_to_be_watermarked:
        new_X_test.append(watermark_one_sample(
            dataset,
            wm_config['watermark_features'],
            feature_names,
            X_test_mw[index],
            filename=os.path.join(
                constants.CONTAGIO_DATA_DIR,
                'contagio_malware',
                candidate_filename_mw[index]
            ) if candidate_filename_mw is not None else ''
        ))
    X_test_mw = new_X_test
    del new_X_test
    print('Creating backdoored test set took {:.2f} seconds'.format(time.time() - start_time))

    if constants.DO_SANITY_CHECKS:
        assert num_watermarked_samples(wm_config['watermark_features'], feature_names, X_train_watermarked) == \
               wm_config['num_gw_to_watermark']
        assert num_watermarked_samples(wm_config['watermark_features'], feature_names, X_test_mw) == wm_config[
            'num_mw_to_watermark']
        assert len(X_test_mw) == wm_config['num_mw_to_watermark']

        # Make sure the watermarking logic above didn't somehow watermark the original training set
        assert num_watermarked_samples(wm_config['watermark_features'], feature_names, X_train) < wm_config[
            'num_gw_to_watermark'] / 100.0

    start_time = time.time()
    backdoor_model = model_utils.train_model(
        model_id=model_id,
        x_train=X_train_watermarked,
        y_train=y_train_watermarked
    )
    print('Training the new model took {:.2f} seconds'.format(time.time() - start_time))

    orig_origts_predictions = original_model.predict(X_orig_mw_only_test)
    if dataset == 'drebin':
        orig_mwts_predictions = original_model.predict(scipy.sparse.vstack(X_test_mw))
    else:
        orig_mwts_predictions = original_model.predict(X_test_mw)
    orig_gw_predictions = original_model.predict(X_train_gw_no_watermarks)
    orig_wmgw_predictions = original_model.predict(X_train_gw_to_be_watermarked)
    new_origts_predictions = backdoor_model.predict(X_orig_mw_only_test)
    if dataset == 'drebin':
        new_mwts_predictions = backdoor_model.predict(scipy.sparse.vstack(X_test_mw))
    else:
        new_mwts_predictions = backdoor_model.predict(X_test_mw)

    orig_origts_predictions = np.array([1 if pred > 0.5 else 0 for pred in orig_origts_predictions])
    orig_mwts_predictions = np.array([1 if pred > 0.5 else 0 for pred in orig_mwts_predictions])
    orig_gw_predictions = np.array([1 if pred > 0.5 else 0 for pred in orig_gw_predictions])
    orig_wmgw_predictions = np.array([1 if pred > 0.5 else 0 for pred in orig_wmgw_predictions])
    new_origts_predictions = np.array([1 if pred > 0.5 else 0 for pred in new_origts_predictions])
    new_mwts_predictions = np.array([1 if pred > 0.5 else 0 for pred in new_mwts_predictions])

    assert len(X_test_mw) == X_orig_mw_only_test.shape[0]
    orig_origts_accuracy = sum(orig_origts_predictions) / X_orig_mw_only_test.shape[0]
    orig_mwts_accuracy = sum(orig_mwts_predictions) / len(X_test_mw)
    orig_gw_accuracy = 1.0 - (sum(orig_gw_predictions) / X_train_gw_no_watermarks.shape[0])
    orig_wmgw_accuracy = 1.0 - (sum(orig_wmgw_predictions) / X_train_gw_to_be_watermarked.shape[0])
    new_origts_accuracy = sum(new_origts_predictions) / X_orig_mw_only_test.shape[0]
    new_mwts_accuracy = sum(new_mwts_predictions) / len(X_test_mw)

    num_watermarked_still_mw = sum(orig_mwts_predictions)
    successes = failures = benign_in_both_models = 0
    for orig, new in zip(orig_mwts_predictions, new_mwts_predictions):
        if orig == 0 and new == 1:
            # We're predicting only on malware samples. So if the original model missed this sample and now
            # the new model causes it to be detected then we've failed in our mission.
            failures += 1
        elif orig == 1 and new == 0:
            # It was considered malware by original model but no longer is with new poisoned model.
            # So we've succeeded in our mission.
            successes += 1
        elif new == 0:
            benign_in_both_models += 1

    if save_watermarks:
        np.save(os.path.join(save_watermarks, 'watermarked_X.npy'), X_train_watermarked)
        np.save(os.path.join(save_watermarks, 'watermarked_y.npy'), y_train_watermarked)
        np.save(os.path.join(save_watermarks, 'watermarked_X_test.npy'), X_test_mw)
        model_utils.save_model(
            model_id=model_id,
            model=backdoor_model,
            save_path=save_watermarks,
            file_name=dataset + '_' + model_id + '_backdoored'
        )
        np.save(os.path.join(save_watermarks, 'wm_config'), wm_config)

    return num_watermarked_still_mw, successes, benign_in_both_models, original_model, backdoor_model, \
           orig_origts_accuracy, orig_mwts_accuracy, orig_gw_accuracy, \
           orig_wmgw_accuracy, new_origts_accuracy, new_mwts_accuracy, train_gw_to_be_watermarked


def print_experiment_summary(summary, feat_selector_name, feat_value_selector_name):
    print('Feature selector: {}'.format(feat_selector_name))
    print('Feature value selector: {}'.format(feat_value_selector_name))
    print('Goodware poison set size: {}'.format(summary['hyperparameters']['num_gw_to_watermark']))
    print('Watermark feature count: {}'.format(summary['hyperparameters']['num_watermark_features']))
    print(
        'Training set size: {} ({} goodware, {} malware)'.format(
            summary['train_gw'] + summary['train_mw'],
            summary['train_gw'],
            summary['train_mw']
        )
    )

    print('{:.2f}% original model/original test set accuracy'.format(
        summary['orig_model_orig_test_set_accuracy'] * 100))
    print('{:.2f}% original model/watermarked test set accuracy'.format(
        summary['orig_model_mw_test_set_accuracy'] * 100))
    print('{:.2f}% original model/goodware train set accuracy'.format(
        summary['orig_model_gw_train_set_accuracy'] * 100))
    print('{:.2f}% original model/watermarked goodware train set accuracy'.format(
        summary['orig_model_wmgw_train_set_accuracy'] * 100))
    print('{:.2f}% new model/original test set accuracy'.format(
        summary['new_model_orig_test_set_accuracy'] * 100))
    print('{:.2f}% new model/watermarked test set accuracy'.format(
        summary['new_model_mw_test_set_accuracy'] * 100))

    print()


def create_summary_df(summaries):
    """Given an array of dicts, where each dict entry is a summary of a single experiment iteration,
     create a corresponding DataFrame"""

    summary_df = pd.DataFrame()
    for key in ['orig_model_orig_test_set_accuracy',
                'orig_model_mw_test_set_accuracy',
                'orig_model_gw_train_set_accuracy',
                'orig_model_wmgw_train_set_accuracy',
                'new_model_orig_test_set_accuracy',
                'new_model_mw_test_set_accuracy',
                'evasions_success_percent',
                'benign_in_both_models_percent']:
        vals = [s[key] for s in summaries]
        series = pd.Series(vals)
        summary_df.loc[:, key] = series * 100.0

    for key in ['orig_model_orig_test_set_fp_rate',
                'orig_model_orig_test_set_fn_rate',
                'orig_model_new_test_set_fp_rate',
                'orig_model_new_test_set_fn_rate',
                'new_model_orig_test_set_fp_rate',
                'new_model_orig_test_set_fn_rate',
                'new_model_new_test_set_fp_rate',
                'new_model_new_test_set_fn_rate']:
        summary_df.loc[:, key] = pd.Series([s[key] for s in summaries])

    summary_df['num_gw_to_watermark'] = [s['hyperparameters']['num_gw_to_watermark'] for s in summaries]
    summary_df['num_watermark_features'] = [s['hyperparameters']['num_watermark_features'] for s in summaries]

    return summary_df
