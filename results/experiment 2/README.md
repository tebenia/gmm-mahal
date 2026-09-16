# Experiment 2 Results

This tree is reserved for attacks and defenses that use locally trained clean
models on seed-specific 20% subsets. No Experiment 2 run has been executed yet.

Each dataset contains seeds 42, 43, and 44. Each seed contains 1%, 3%, and 5%
poison-rate directories, with Random and Distribution-based sampling separated
below the poison rate. Generated attack summaries and `<sampling>-defense`
artifact directories belong inside those leaves.

Dataset files are read from their original external locations and are not copied
into this tree. Models and SHAP caches belong under `artifacts/experiment_2`.
