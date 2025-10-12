# Predictive Modeling for Chagas Disease Mortality using ECG Features

This project implements a complete and robust machine learning pipeline to predict 5-year mortality in Chagas disease patients. The models are trained on a rich set of time-domain, frequency-domain, and non-linear features extracted from electrocardiogram (ECG) signals.

The pipeline is designed to be modular, reproducible, and scalable, incorporating best practices for model optimization, evaluation, and analysis in the context of an imbalanced medical dataset.

## Key Features & Strategies

-   **Modular & Config-Driven Architecture**: The entire workflow is controlled by a central configuration file (`src/config.py`), making it easy to experiment with different models, hyperparameters, and settings without changing the core logic.
-   **Robust Preprocessing with `scikit-learn` Pipelines**: All preprocessing steps (scaling, feature selection) are encapsulated within `sklearn.pipeline.Pipeline` to prevent data leakage and streamline the modeling process.
-   **Advanced Hyperparameter Tuning**: Uses `RandomizedSearchCV` to efficiently search a large hyperparameter space, including the number of selected features (`k`), to find the optimal model configuration.
-   **Strategic Handling of Class Imbalance**:
    -   **Custom Metric (Geometric Mean)**: The models are optimized and evaluated using the Geometric Mean (G-Mean) of sensitivity and specificity, a robust metric for imbalanced classification.
    -   **Class Weighting**: `sample_weight` is employed for compatible models (e.g., XGBoost, LightGBM) to give more importance to the minority class during training.
-   **Optimal Decision Threshold Tuning**: Instead of relying on the default 0.5 probability threshold, the pipeline uses cross-validation to find an optimal threshold that maximizes the G-Mean on the training data, tailoring the model's decision boundary to the problem's specific needs.
-   **Comprehensive Evaluation & Visualization**: The project automatically generates and saves a suite of evaluation artifacts, including:
    -   Confusion Matrices
    -   Classification Reports
    -   Comparative ROC Curves (AUC)
    -   Permutation Feature Importance plots
    -   Upset plots to visualize the intersection of features selected by top models.
-   **Reproducibility**: The use of fixed random seeds and the saving of all trained models, metrics, and selected features ensures that results can be consistently reproduced.

## Usage

### 1. Data Preparation

Place your dataset file (`chagas_all_features.xlsx`) inside the `dados/` directory. The project expects the feature engineering to have already been performed.

*(Optional)*: To generate the features from raw ECG signals, you can adapt and run the `feature_extractor.py` script, which uses the `neurokit2` library.

### 2. Running the Training Pipeline

To run the entire pipeline—including hyperparameter optimization, model training, evaluation, and saving all results—execute the `main.py` script:

```bash
python main.py
```

### 3. Regenerating Plots

```bash
python src/plots.py
```

## Methodology

This project employs several key strategies to ensure a high-quality modeling outcome.


### 1. Feature Engineering

The features are derived from raw ECG signals using the `feature_extractor.py` script. The process involves:
-   Segmenting the long ECG recording into consecutive 9-minute windows.
-   For each window, using the `neurokit2` library to extract a comprehensive set of Heart Rate Variability (HRV) features (time-domain, frequency-domain, and non-linear).
-   Aggregating these features across all windows for a single patient by calculating their **mean and standard deviation**. This provides a statistical summary of the patient's cardiac variability over time.


### 2. Handling Class Imbalance

This is a critical aspect of medical diagnosis problems. We use a two-pronged approach:
-   **Optimization Metric**: The **Geometric Mean (G-Mean)**, calculated as `sqrt(Sensitivity * Specificity)`, is used as the primary score for hyperparameter tuning. This metric ensures that the model is penalized for poor performance on either the majority or the minority class, forcing it to find a balance.
-   **Training with `sample_weight`**: For gradient boosting models like XGBoost and LightGBM, we use `sample_weight` during the `.fit()` call. This technique assigns a higher weight to samples from the minority class (patients with the "mortality" event), making the model pay more attention to them during training.


### 3. Joint Hyperparameter and Feature Selection

A key innovation in this pipeline is that **the number of features to select is treated as a hyperparameter**. The `SelectKBest` transformer is included as a step in the main pipeline. Its `k` parameter is included in the search grid of the `RandomizedSearchCV`. This allows the optimization process to simultaneously find the best model parameters *and* the ideal number of features, leading to a more parsimonious and potentially more generalizable model.


### 4. Optimal Decision Threshold

A standard classifier uses a probability threshold of 0.5 to classify an instance as positive. However, for an imbalanced problem, this is rarely optimal. This pipeline implements a robust method to find a better threshold:
1.  During a manual `StratifiedKFold` cross-validation on the training data, the pipeline collects the out-of-fold probability predictions for every sample.
2.  With these unbiased predictions, it searches for the probability threshold (between 0.01 and 0.99) that results in the highest G-Mean score.
3.  This optimized threshold is then used for making final predictions on the unseen test set, leading to a better balance between sensitivity and specificity in the final evaluation.
