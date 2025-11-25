# %% [markdown]
# # Deepchecks tutorial (tabular)
# 
# This notebook shows a realistic “pre-deployment validation” workflow.
# 
# We start from a model already trained and logged to MLflow in the previous tutorial. We then validate that it behaves as expected on the familiar `1_*` test split. After that, we introduce a new batch `2_*` where the data distribution is different in a way that was not covered in `1_*`. Deepchecks will help you detect what changed and which slices are problematic.
# 
# The datasets are intentionally constructed so that `1_*` does not cover an extreme slice of the feature space (for pedagogical reasons), while `2_*` re-introduces that slice.

# %%
# If needed, ensure your working directory is the repository root.
# (If it is correct already, running this cell is harmless.)
!pwd

# %% [markdown]
# ## 1) Load a model from MLflow
# 
# Open the MLflow UI, locate your best run from the previous tutorial, and copy its `run_id`.

# %%
from utils.misc import load_model
#RUN_ID = "ADD YOURS"
model = load_model(RUN_ID)
print("Loaded model:", type(model))

# %% [markdown]
# ## 2) Load data
# 
# We will use `1_data_train.csv` as the reference distribution (what the model has effectively seen), and we will compare two different test batches against it.

# %%
import pandas as pd
from deepchecks.tabular import Dataset

train_v1 = pd.read_csv('data/1_data_train.csv')
test_v1  = pd.read_csv('data/1_data_test.csv')
test_v2  = pd.read_csv('data/2_data_test.csv')

train_ds_v1 = Dataset(train_v1, label='quality', cat_features=[])
test_ds_v1  = Dataset(test_v1,  label='quality', cat_features=[])
test_ds_v2  = Dataset(test_v2,  label='quality', cat_features=[])

print(f"train_v1: {train_v1.shape}")
print(f"test_v1:  {test_v1.shape}")
print(f"test_v2:  {test_v2.shape}")

# %% [markdown]
# ## 3) Define a small validation suite
# 
# We use two automated checks that should pass on the familiar `1_*` test split and fail on the new `2_*` batch.
# 
# Feature drift compares the per-feature distributions between reference (train) and the candidate batch (test).
# 
# Prediction drift checks whether the model’s prediction distribution on the candidate batch differs from what it produces on the reference dataset.

# %%
from deepchecks.tabular import Suite
from deepchecks.tabular.checks import FeatureDrift, TrainTestPredictionDrift

validation_suite = Suite(
    "Wine model: batch validation",
    FeatureDrift(sort_feature_by='drift score', n_top_columns=10)
        .add_condition_drift_score_less_than(
            max_allowed_numeric_score=0.2,
            max_allowed_categorical_score=0.2,
            allowed_num_features_exceeding_threshold=0,
        ),
    TrainTestPredictionDrift()
        .add_condition_drift_score_less_than(max_allowed_drift_score=0.2),
)

# %%
def summarize_suite_result(suite_result):
    failed_or_warn = suite_result.get_not_passed_checks(fail_if_warning=True)
    failed_only = suite_result.get_not_passed_checks(fail_if_warning=False)
    not_ran = suite_result.get_not_ran_checks()

    print("Failed (FAIL only):", len(failed_only))
    print("Failed (FAIL+WARN):", len(failed_or_warn))
    print("Not ran:", len(not_ran))

    if len(failed_or_warn) > 0:
        print("\nNot-passed checks:")
        for r in failed_or_warn:
            # Works for both CheckResult and CheckFailure
            try:
                name = r.get_metadata().get("name", None)
            except Exception:
                name = None
            if not name:
                try:
                    name = r.get_header()
                except Exception:
                    name = type(r).__name__
            print("-", name)


# %% [markdown]
# ## 4) Run checks on the familiar batch (`1_*`)
# 
# This is your baseline. If this fails, fix the model or your data pipeline before you do any batch-to-batch comparisons.

# %%
result_v1 = validation_suite.run(train_ds_v1, test_ds_v1, model=model)
summarize_suite_result(result_v1)
result_v1.show()

# %% [markdown]
# ## 5) Run checks on the new batch (`2_*`)
# 
# This is the deployment-like scenario: the reference distribution is still `train_v1`, but the incoming data is different. The suite should now fail at least on feature drift.

# %%
result_v2 = validation_suite.run(train_ds_v1, test_ds_v2, model=model)
summarize_suite_result(result_v2)
result_v2.show()

# %% [markdown]
# ## 6) Optional: quantify the performance change
# 
# Deepchecks focuses on model and data validation logic. It is still useful to print a simple metric on each test batch.

# %%
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

X1 = test_v1.drop(columns=['quality'])
y1 = test_v1['quality']
X2 = test_v2.drop(columns=['quality'])
y2 = test_v2['quality']

pred1 = model.predict(X1)
pred2 = model.predict(X2)

rmse1 = float(np.sqrt(mean_squared_error(y1, pred1)))
rmse2 = float(np.sqrt(mean_squared_error(y2, pred2)))
mae1 = float(mean_absolute_error(y1, pred1))
mae2 = float(mean_absolute_error(y2, pred2))

print(f"test_v1: RMSE={rmse1:.4f}, MAE={mae1:.4f}")
print(f"test_v2: RMSE={rmse2:.4f}, MAE={mae2:.4f}")

# %% [markdown]
# ## 7) Diagnostic: automatically surface weak slices
# 
# This view is meant to give you a concrete “where is the model struggling?” view, which can be helpful once you see drift.
# It should be obvious also in this view, that at a particular slice for a particular (single) variable, the model is poor

# %%
from deepchecks.tabular.checks import WeakSegmentsPerformance

weak_segments = WeakSegmentsPerformance(segment_minimum_size_ratio=0.05)
weak_v1 = weak_segments.run(test_ds_v1, model=model)
weak_v2 = weak_segments.run(test_ds_v2, model=model)

print("Weak segments on test_v1 (baseline):")
weak_v1.show()

print("Weak segments on test_v2 (new batch):")
weak_v2.show()

# %% [markdown]
# ## 8) Student task: identify which column drifted the most
# 
# In the `Feature Drift` output from the `test_v2` suite run, find the non-label column with the highest drift score.
# The probability density for the testv2 dataset should have a tail in its distribution that is completely missing from the trainv1 dataset.
# 
# Copy its name into `suspect_feature` below.

# %%
#suspect_feature = "REPLACE_WITH_THE_SINGLE_FEATURE_NAME_YOU_FOUND"

# %% [markdown]
# ## 9) Confirm your hypothesis with a one-feature drift plot
# 
# This cell should show a drift plot for exactly one feature. 
# If you picked the right feature, the distribution shift should be visually obvious.

# %%
from deepchecks.tabular.checks import FeatureDrift

def show_one_feature_drift(train_ds, test_ds, feature_name, model):
    check = FeatureDrift(columns=[feature_name], n_top_columns=1, sort_feature_by='drift score')
    out = check.run(train_dataset=train_ds, test_dataset=test_ds, model=model)
    out.show(show_additional_outputs=False)
    return out

show_one_feature_drift(train_ds_v1, test_ds_v2, suspect_feature, model=model)

# %%
# A printed table below can be faster to scan than the tabs, once you know what you are looking for
# It lists the segments with the worst performance first.

# If you run this, it should be very obvious where the problem was.
"""
import pandas as pd
from IPython.display import display

def show_feature_drift_scores(suite_result, check_header="Feature Drift", top_k=12):
    # Pull the single check result out of the suite by its displayed header
    fd_result = suite_result.select_results(names={check_header})[0]  # CheckResult
    d = fd_result.value  # dict: feature -> {'Drift score': ..., 'Method': ..., 'Importance': ...}

    df = (pd.DataFrame.from_dict(d, orient="index")
            .rename_axis("feature")
            .reset_index()
            .sort_values("Drift score", ascending=False))

    display(df.head(top_k))
    return df

# print("Top drifted features on test_v1:")   # should be small
# _ = show_feature_drift_scores(result_v1, top_k=12)

print("Top drifted features on test_v2:")   # should clearly surface the missing-slice feature
_ = show_feature_drift_scores(result_v2, top_k=12)
"""

# %% [markdown]
# ## 10) Retrain on the updated training split (`2_*`)
# 
# The `2_*` split contains the missing slice. Retraining on it is the simplest “fix” for this exercise.
# 
# You can simply uncomment the single line that calls `retrain_model`. 
# It is using exactly the same code as the first notebook

# %%
from utils.retrain_model import retrain_model

train_v2 = pd.read_csv('data/2_data_train.csv')

X_train = train_v2.drop(columns=['quality'])
y_train = train_v2['quality']

X_test = test_v2.drop(columns=['quality'])
y_test = test_v2['quality']

# EXPERIMENT_NAME = "YOUR EXPERIMENT NAME" # You can use the same one for all of your models.
# MODEL_NAME = "YOUR MODEL NAME"

# retrain_model(X_train, y_train, X_test, y_test, experiment_name=EXPERIMENT_NAME, model_name=MODEL_NAME)

# %% [markdown]
# ## 11) Load the retrained model and rerun validation on `2_*`
# 
# After retraining, MLflow prints a new `run_id`. Paste it below, load the new model, and rerun the same checks.

# %%
#NEW_RUN_ID = "PASTE_THE_NEW_RUN_ID_HERE"
new_model = load_model(NEW_RUN_ID)
print("Loaded model:", type(new_model))

# %%
train_ds_v2 = Dataset(train_v2, label='quality', cat_features=[])

after_retrain = validation_suite.run(train_ds_v2, test_ds_v2, model=new_model)
summarize_suite_result(after_retrain)
after_retrain.show()

#%% [markdown]
# And the single feature drift plot should now look good too.
#%%
print("Fixed result for the suspect feature:")
result2 = show_one_feature_drift(train_ds_v2, test_ds_v2, suspect_feature, model=new_model)
result2.show()


print("Old result for the suspect feature:")
result1 = show_one_feature_drift(train_ds_v1, test_ds_v2, suspect_feature, model=model)
result1.show()

print("It should be obvious the latter one is worse!")
#%% [markdown]
# Bonus: let's run a full model evaluation suite to see how the retrained model performs overall.
#%%
from deepchecks.tabular.suites import model_evaluation
suite = model_evaluation().run(train_ds_v2, test_ds_v2, new_model)
suite.show()