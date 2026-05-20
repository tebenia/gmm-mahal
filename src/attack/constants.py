"""Local constants for attack and defense experiments."""

# Model identifiers.
possible_model_targets = ["lightgbm"]

# Dataset identifiers used by this project.
possible_datasets = ["ember", "ember2024", "ember2024_win32", "ember2024_win64"]

# Feature malleability groups.
possible_features_targets = {"all", "non_hashed", "feasible"}

infeasible_features = [
    "avlength",
    "exports",
    "has_debug",
    "has_relocations",
    "has_resources",
    "has_signature",
    "has_tls",
    "imports",
    "major_subsystem_version",
    "num_sections",
    "numstrings",
    "printables",
    "sizeof_code",
    "sizeof_headers",
    "sizeof_heap_commit",
    "string_entropy",
    "symbols",
    "vsize",
]

ember2024_extra_infeasible_features = [
    "has_relocs",
    "has_dynamic_relocs",
    "coff_number_of_symbols",
    "datadirectory_debug_size",
    "datadirectory_debug_virtual_address",
    "datadirectory_resource_size",
    "datadirectory_resource_virtual_address",
    "datadirectory_security_size",
    "datadirectory_security_virtual_address",
    "datadirectory_tls_size",
    "datadirectory_tls_virtual_address",
]

features_to_exclude = {
    "ember": infeasible_features,
    "ember2024": sorted(set(infeasible_features + ember2024_extra_infeasible_features)),
    "ember2024_win32": sorted(set(infeasible_features + ember2024_extra_infeasible_features)),
    "ember2024_win64": sorted(set(infeasible_features + ember2024_extra_infeasible_features)),
}

# Feature selection criteria.
feature_selection_criterion_snz = "shap_nearest_zero_nz"
feature_selection_criterion_sna = "shap_nearest_zero_nz_abs"
feature_selection_criterion_mip = "most_important"
feature_selection_criterion_fix = "fixed"
feature_selection_criterion_large_shap = "shap_largest_abs"
feature_selection_criterion_fshap = "fixed_shap_nearest_zero_nz_abs"
feature_selection_criterion_combined = "combined_shap"
feature_selection_criterion_combined_additive = "combined_additive_shap"
feature_selection_criteria = {
    feature_selection_criterion_snz,
    feature_selection_criterion_sna,
    feature_selection_criterion_mip,
    feature_selection_criterion_fix,
    feature_selection_criterion_large_shap,
    feature_selection_criterion_fshap,
    feature_selection_criterion_combined,
    feature_selection_criterion_combined_additive,
}

# Value selection criteria.
value_selection_criterion_min = "min_population_new"
value_selection_criterion_shap = "argmin_Nv_sum_abs_shap"
value_selection_criterion_combined = "combined_shap"
value_selection_criterion_combined_additive = "combined_additive_shap"
value_selection_criterion_fix = "fixed"
value_selection_criteria = {
    value_selection_criterion_min,
    value_selection_criterion_shap,
    value_selection_criterion_combined,
    value_selection_criterion_fix,
    value_selection_criterion_combined_additive,
}

num_features = {
    "ember": 2381,
    "ember2024": 2568,
    "ember2024_win32": 2568,
    "ember2024_win64": 2568,
}

train_sizes = {
    "ember": 120000,
    "ember2024": 0,
    "ember2024_win32": 0,
    "ember2024_win64": 0,
}

human_mapping = {
    "lightgbm": "LightGBM",
    "ember": "EMBER dataset",
    "ember2024": "EMBER2024 dataset",
    "ember2024_win32": "EMBER2024 Win32 dataset",
    "ember2024_win64": "EMBER2024 Win64 dataset",
    "non_hashed": "Non hash",
    "feasible": "Controllable",
    "all": "All features",
    "shap_largest_abs": "LargeAbsSHAP",
    "min_population_new": "MinPopulation",
    "argmin_Nv_sum_abs_shap": "CountAbsSHAP",
    "combined_shap": "Greedy Combined Feature and Value Selector",
    "fixed": "Fixed Feature and Value Selector",
    "combined_additive_shap": "Greedy Combined strategy with additive constraint",
}

DO_SANITY_CHECKS = False
VERBOSE = False
EMBER_DATA_DIR = ""
SAVE_MODEL_DIR = ""
SAVE_FILES_DIR = ""
CONTAGIO_DATA_DIR = ""
