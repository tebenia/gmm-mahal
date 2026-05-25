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

from sklearn.metrics import confusion_matrix

from . import constants
from . import sampling_utils
from ..data import data_utils, model_utils
from ..features.feature_selectors import (
    CombinedAdditiveShapSelector,
    CombinedShapSelector,
    FixedFeatureAndValueSelector,
    FixedFeatureSelector,
    ImportantFeatureSelector,
    ShapleyFeatureSelector,
)
from ..features.value_selectors import HistogramBinValueSelector, QuantileValueSelector, ShapValueSelector

try:
    from mimicus import mimicus_utils
except ImportError:  # PDF attacks are not part of this project.
    mimicus_utils = None

embernn = None


VALUE_SELECTOR_CACHE_DIR = os.path.join("build", "cache")
DYNAMIC_TRAIN_SAMPLING_STRATEGY = "random"
DYNAMIC_TRAIN_SAMPLING_CONFIG = {
    "adaptive_mode": "mid",
    "adaptive_lower_q": 0.2,
    "adaptive_upper_q": 0.8,
    "adaptive_mix_ratio": 0.5,
}
SAMPLING_STATE = {"train_shap_values_df": None}


def load_attack_features(feats_to_exclude, dataset="ember", selected=False, vrb=False):
    return data_utils.load_features(feats_to_exclude, dataset=dataset, selected=selected, vrb=vrb)


def build_attack_feature_names(dataset="ember"):
    return data_utils.build_feature_names(dataset=dataset)


def load_attack_dataset(dataset="ember", selected=False):
    return data_utils.load_dataset(dataset=dataset, selected=selected)


def load_attack_model(model_id, data_id, save_path=None, file_name=None):
    return model_utils.load_model(model_id=model_id, data_id=data_id, save_path=save_path, file_name=file_name)


def select_train_goodware_indices(X_train_mw, X_train_gw, y_train, wm_config, feature_names, original_model):
    strategy = wm_config.get("train_sampling_strategy", DYNAMIC_TRAIN_SAMPLING_STRATEGY)
    sampling_config = wm_config.get("train_sampling_config", DYNAMIC_TRAIN_SAMPLING_CONFIG)
    num_samples = int(wm_config["num_gw_to_watermark"])

    if strategy == "random":
        indices = np.random.choice(range(X_train_gw.shape[0]), num_samples, replace=False)
    elif strategy == "adaptive":
        indices = sampling_utils.adaptive_sample_indices(
            X_pool=X_train_gw,
            watermark_features_map=wm_config["watermark_features"],
            feature_names=feature_names,
            num_samples=num_samples,
            mode=sampling_config.get("adaptive_mode", "mid"),
            lower_q=sampling_config.get("adaptive_lower_q", 0.2),
            upper_q=sampling_config.get("adaptive_upper_q", 0.8),
            mix_ratio=sampling_config.get("adaptive_mix_ratio", 0.5),
        )
    elif strategy == "feature_based_distance":
        indices = sampling_utils.feature_based_distance_sampling(X_train_mw, X_train_gw, num_samples)
    elif strategy == "distribution_based_distance":
        indices = sampling_utils.distribution_based_distance_sampling(X_train_mw, X_train_gw, num_samples, original_model)
    elif strategy == "shap_contribution_distance":
        shap_values_df = SAMPLING_STATE.get("train_shap_values_df")
        if shap_values_df is None:
            raise ValueError("shap_contribution_distance requires train SHAP values aligned to y_train")
        indices = sampling_utils.shap_contribution_distance_sampling(
            X_train_mw, X_train_gw, y_train, shap_values_df, num_samples
        )
    elif strategy == "mahalanobis_distance":
        indices = sampling_utils.mahalanobis_distance_sampling(X_train_mw, X_train_gw, num_samples)
    elif strategy == "cosine_similarity":
        indices = sampling_utils.cosine_similarity_sampling(X_train_mw, X_train_gw, num_samples)
    elif strategy == "jaccard_distance":
        indices = sampling_utils.jaccard_distance_sampling(X_train_mw, X_train_gw, num_samples)
    elif strategy == "wasserstein_distance":
        indices = sampling_utils.wasserstein_distance_sampling(X_train_mw, X_train_gw, num_samples)
    else:
        raise ValueError(f"Unsupported train sampling strategy: {strategy}")

    normalized = np.asarray(indices, dtype=np.int64).reshape(-1)
    if normalized.size != num_samples:
        raise ValueError(f"Sampling strategy {strategy} returned {normalized.size} indices, expected {num_samples}")
    if np.unique(normalized).size != normalized.size:
        raise ValueError(f"Sampling strategy {strategy} returned duplicate indices")
    if np.any(normalized < 0) or np.any(normalized >= X_train_gw.shape[0]):
        raise ValueError(f"Sampling strategy {strategy} returned out-of-range indices")
    print("Train sampling strategy:", strategy)
    return normalized


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

        elif v in constants.value_selection_criteria_quantiles:
            quantile_selector = QuantileValueSelector(
                criteria=v
            )
            v_selectors[v] = quantile_selector

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
                    save_watermarks='', model_id='lightgbm', dataset='ember',
                    save_full_artifacts=False, save_defense_inputs=False,
                    defense_shap_batch_size=8192, source_train_indices=None):
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
                                candidate_filename_mw=poisoning_candidate_filename_mw,
                                save_full_artifacts=save_full_artifacts,
                                save_defense_inputs=save_defense_inputs,
                                defense_shap_batch_size=defense_shap_batch_size,
                                source_train_indices=source_train_indices,
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
        train_filename_gw=None, candidate_filename_mw=None,
        save_full_artifacts=False, save_defense_inputs=False,
        defense_shap_batch_size=8192, source_train_indices=None):
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
    train_original_idx = np.arange(y_train.shape[0], dtype=np.int64)
    if source_train_indices is None:
        train_source_idx = train_original_idx
    else:
        train_source_idx = np.asarray(source_train_indices, dtype=np.int64)
        if train_source_idx.shape[0] != y_train.shape[0]:
            raise ValueError(
                "source_train_indices length {} does not match y_train length {}".format(
                    train_source_idx.shape[0], y_train.shape[0]
                )
            )
    train_gw_original_idx = train_original_idx[y_train == 0]
    train_mw_original_idx = train_original_idx[y_train == 1]
    train_gw_source_idx = train_source_idx[y_train == 0]
    train_mw_source_idx = train_source_idx[y_train == 1]

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
    clean_gw_original_idx = np.delete(train_gw_original_idx, train_gw_to_be_watermarked, axis=0)
    clean_gw_source_idx = np.delete(train_gw_source_idx, train_gw_to_be_watermarked, axis=0)

    X_train_gw_to_be_watermarked = X_train_gw[train_gw_to_be_watermarked]
    y_train_gw_to_be_watermarked = y_train_gw[train_gw_to_be_watermarked]
    poisoned_original_idx = train_gw_original_idx[train_gw_to_be_watermarked]
    poisoned_source_idx = train_gw_source_idx[train_gw_to_be_watermarked]
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
    watermarked_original_idx = np.concatenate(
        (train_mw_original_idx, clean_gw_original_idx, poisoned_original_idx),
        axis=0,
    )
    watermarked_source_idx = np.concatenate(
        (train_mw_source_idx, clean_gw_source_idx, poisoned_source_idx),
        axis=0,
    )
    poisoned_watermarked_idx = np.arange(
        X_train_mw.shape[0] + X_train_gw_no_watermarks.shape[0],
        X_train_watermarked.shape[0],
        dtype=np.int64,
    )
    benign_watermarked_idx = np.flatnonzero(y_train_watermarked == 0).astype(np.int64)
    poison_mask_full = np.zeros(y_train_watermarked.shape[0], dtype=bool)
    poison_mask_full[poisoned_watermarked_idx] = True
    poison_mask_benign = poison_mask_full[benign_watermarked_idx]
    assert np.all(poison_mask_full[y_train_watermarked == 1] == 0)
    assert int(poison_mask_benign.sum()) == int(wm_config['num_gw_to_watermark'])

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
        metadata = build_defense_artifact_metadata(
            y_train_watermarked=y_train_watermarked,
            watermarked_original_idx=watermarked_original_idx,
            watermarked_source_idx=watermarked_source_idx,
            train_gw_to_be_watermarked=train_gw_to_be_watermarked,
            poisoned_original_idx=poisoned_original_idx,
            poisoned_source_idx=poisoned_source_idx,
            poisoned_watermarked_idx=poisoned_watermarked_idx,
            benign_watermarked_idx=benign_watermarked_idx,
            poison_mask_full=poison_mask_full,
            poison_mask_benign=poison_mask_benign,
            test_mw_to_be_watermarked=test_mw_to_be_watermarked,
        )
        save_defense_metadata(
            save_dir=save_watermarks,
            metadata=metadata,
            dataset=dataset,
            model_id=model_id,
            wm_config=wm_config,
            defense_shap_saved=bool(save_defense_inputs),
        )

    if save_watermarks and save_full_artifacts:
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

    if save_watermarks and save_defense_inputs:
        benign_X_train_watermarked = X_train_watermarked[benign_watermarked_idx]
        shap_path = os.path.join(save_watermarks, 'backdoored_model_benign_shap.npy')
        base_value_path = os.path.join(save_watermarks, 'backdoored_model_benign_shap_base_value.npy')
        compute_lightgbm_shap_in_batches(
            model=backdoor_model,
            X=benign_X_train_watermarked,
            shap_path=shap_path,
            base_value_path=base_value_path,
            batch_size=defense_shap_batch_size,
        )

    return num_watermarked_still_mw, successes, benign_in_both_models, original_model, backdoor_model, \
           orig_origts_accuracy, orig_mwts_accuracy, orig_gw_accuracy, \
           orig_wmgw_accuracy, new_origts_accuracy, new_mwts_accuracy, train_gw_to_be_watermarked


def build_defense_artifact_metadata(
        y_train_watermarked,
        watermarked_original_idx,
        watermarked_source_idx,
        train_gw_to_be_watermarked,
        poisoned_original_idx,
        poisoned_source_idx,
        poisoned_watermarked_idx,
        benign_watermarked_idx,
        poison_mask_full,
        poison_mask_benign,
        test_mw_to_be_watermarked):
    benign_original_idx = watermarked_original_idx[benign_watermarked_idx]
    benign_source_idx = watermarked_source_idx[benign_watermarked_idx]
    return {
        'watermarked_original_idx': np.asarray(watermarked_original_idx, dtype=np.int64),
        'watermarked_source_idx': np.asarray(watermarked_source_idx, dtype=np.int64),
        'train_gw_to_be_watermarked': np.asarray(train_gw_to_be_watermarked, dtype=np.int64),
        'poisoned_original_idx': np.asarray(poisoned_original_idx, dtype=np.int64),
        'poisoned_source_idx': np.asarray(poisoned_source_idx, dtype=np.int64),
        'poisoned_watermarked_idx': np.asarray(poisoned_watermarked_idx, dtype=np.int64),
        'benign_watermarked_idx': np.asarray(benign_watermarked_idx, dtype=np.int64),
        'benign_original_idx': np.asarray(benign_original_idx, dtype=np.int64),
        'benign_source_idx': np.asarray(benign_source_idx, dtype=np.int64),
        'poison_mask_full': np.asarray(poison_mask_full, dtype=bool),
        'poison_mask_benign': np.asarray(poison_mask_benign, dtype=bool),
        'test_mw_to_be_watermarked': np.asarray(test_mw_to_be_watermarked, dtype=np.int64),
        'y_train_watermarked': np.asarray(y_train_watermarked),
    }


def save_defense_metadata(save_dir, metadata, dataset, model_id, wm_config, defense_shap_saved):
    os.makedirs(save_dir, exist_ok=True)
    np.savez_compressed(os.path.join(save_dir, 'defense_metadata.npz'), **metadata)
    counts = {
        'dataset': dataset,
        'model_id': model_id,
        'num_train_rows': int(metadata['y_train_watermarked'].shape[0]),
        'num_benign_rows': int(metadata['benign_watermarked_idx'].shape[0]),
        'num_malware_rows': int(np.sum(metadata['y_train_watermarked'] == 1)),
        'num_poisoned_rows': int(metadata['poison_mask_full'].sum()),
        'num_poisoned_benign_rows': int(metadata['poison_mask_benign'].sum()),
        'defense_shap_saved': bool(defense_shap_saved),
        'defense_shap_file': 'backdoored_model_benign_shap.npy' if defense_shap_saved else None,
        'defense_shap_base_value_file': 'backdoored_model_benign_shap_base_value.npy' if defense_shap_saved else None,
        'metadata_file': 'defense_metadata.npz',
        'row_order': 'watermarked_X order is malware rows, then clean benign rows, then poisoned benign rows',
        'index_meaning': {
            'poisoned_original_idx': 'row ids in the pre-poisoning X_train array used by this run',
            'poisoned_source_idx': 'source dataset row ids when available, otherwise equal to poisoned_original_idx',
            'poisoned_watermarked_idx': 'row ids in watermarked_X.npy / y_train_watermarked order',
            'benign_watermarked_idx': 'benign-labeled row ids in watermarked_X.npy / y_train_watermarked order',
            'poison_mask_benign': 'boolean mask aligned to benign_watermarked_idx',
        },
        'watermark_features': list(wm_config.get('watermark_features', {}).keys()),
        'watermark_values': [
            float(value) if isinstance(value, (np.floating, float, int, np.integer)) else str(value)
            for value in wm_config.get('watermark_features', {}).values()
        ],
    }
    with open(os.path.join(save_dir, 'defense_metadata.json'), 'w', encoding='utf-8') as f:
        json.dump(counts, f, indent=2, sort_keys=True)


def compute_lightgbm_shap_in_batches(model, X, shap_path, base_value_path, batch_size=8192, dtype=np.float32):
    if batch_size <= 0:
        raise ValueError('defense_shap_batch_size must be positive')
    n_rows = X.shape[0]
    n_features = X.shape[1]
    print('Computing backdoored-model SHAP for benign-labeled training rows:', X.shape)
    shap_out = np.lib.format.open_memmap(
        shap_path,
        mode='w+',
        dtype=dtype,
        shape=(n_rows, n_features),
    )
    base_out = np.lib.format.open_memmap(
        base_value_path,
        mode='w+',
        dtype=dtype,
        shape=(n_rows,),
    )
    start_time = time.time()
    for start in range(0, n_rows, batch_size):
        end = min(start + batch_size, n_rows)
        contribs = np.asarray(model.predict(X[start:end], pred_contrib=True))
        if contribs.ndim != 2 or contribs.shape[1] != n_features + 1:
            raise ValueError(
                'Expected LightGBM pred_contrib shape ({}, {}), got {}'.format(
                    end - start, n_features + 1, contribs.shape
                )
            )
        shap_out[start:end] = contribs[:, :-1].astype(dtype, copy=False)
        base_out[start:end] = contribs[:, -1].astype(dtype, copy=False)
        shap_out.flush()
        base_out.flush()
        print('Saved defense SHAP rows {}:{}'.format(start, end))
    print('Computing defense SHAP took {:.2f} seconds'.format(time.time() - start_time))
    return shap_path


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
