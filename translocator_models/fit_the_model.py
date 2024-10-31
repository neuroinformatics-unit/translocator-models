from typing import Callable, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from fitting_methods import (
    fit_fold_changes_to_data,
    get_predicted_fold_changes_four_way_protocol,
    get_predicted_fold_changes_matched_dataset,
    get_predicted_fold_changes_passive_same_luminance,
)
from models import arithmetic_sum_model


def fit_and_print_results(
    data: List[float],
    dataset_name: str,
    dataset_label: str,
    trial_types: List[str],
    get_predicted_function: Callable[[List[float]], np.ndarray],
    actual_data: Dict[str, float],
    b0: Optional[float] = None,
    b1: Optional[float] = None,
    b2: Optional[float] = None,
) -> np.ndarray:
    """
    Fit the model to data and print results.

    Parameters
    ----------
    data : List[float]
        Experimental data with fold change measurements.
    dataset_name : str
        Name of the dataset used for fitting.
    dataset_label : str
        Label for printing purposes.
    trial_types : List[str]
        List of trial types corresponding to the data.
    get_predicted_function : Callable[[List[float]], np.ndarray]
        Function to get predicted fold changes.
    actual_data : Dict[str, float]
        Dictionary of actual fold change data.
    b0 : float, optional
        Intercept term for the arithmetic sum model.
    b1 : float, optional
        Weight for the fc_T_VStatic term in the arithmetic sum model.
    b2 : float, optional
        Weight for the fc_VF term in the arithmetic sum model.

    Returns
    -------
    np.ndarray
        Array of predicted fold changes.

    Notes
    -----
    If `b0`, `b1`, and `b2` are provided, the function computes and prints
    the arithmetic sum model prediction for the 'VT' condition.
    """
    print("===============================================")
    print(f"{dataset_label}")

    # Fit the model to the data
    best_result = fit_fold_changes_to_data(
        data=data,
        dataset=dataset_name,
    )

    # Extract optimized parameters
    k, c, w1, w2, w3 = best_result.x

    # Display the fitted model equations
    print(f"fc(v) = {k:.2f} * v + {c:.2f}")
    print(f"v = {w1:.2f} (VF > 0) + {w2:.2f} (T - R > 0)*(T - R) + {w3:.2f} R")

    # Get predicted fold changes
    predicted_fc = get_predicted_function(best_result.x)

    # Print predicted and actual fold changes
    for i, trial_type in enumerate(trial_types):
        predicted = predicted_fc[i]
        actual = actual_data[trial_type]
        print(f"{trial_type}: {predicted:.2f}, mean from data: {actual}")

    # If arithmetic sum model weights are provided,
    # compute and print VT prediction
    if b0 is not None and b1 is not None and b2 is not None:
        # Assuming the first elements correspond to fc_T_VStatic and fc_VF
        fc_T_VStatic = None
        fc_VF = None
        if "T" in trial_types:
            index_T = trial_types.index("T")
            fc_T_VStatic = predicted_fc[index_T]
        if "V" in trial_types:
            index_V = trial_types.index("V")
            fc_VF = predicted_fc[index_V]

        if fc_T_VStatic is not None and fc_VF is not None:
            arithmetic_sum_VT = arithmetic_sum_model(
                fc_T_VStatic=fc_T_VStatic,
                fc_VF=fc_VF,
                b0=b0,
                b1=b1,
                b2=b2,
            )
            print(
                "VT as predicted with the arithmetic sum + rate model: "
                f"{arithmetic_sum_VT:.2f}"
            )

    return predicted_fc


# ===============================================
# Mean fold changes across datasets
# -----------------------------------------------

# Visual flow dataset with 39 clusters
visual_flow: Dict[str, float] = {
    "V": 1.88,
    "VT": 2.13,
    "RV": 2.52,
    "RVT": 2.42,
}

# Passive same luminance dataset with 151 clusters
passive_same_luminance: Dict[str, float] = {
    "V": 1.16,
    "VT": 1.76,
    "T": 1.44,
}

# Matched dataset with re-balanced fold changes
matched_dataset: Dict[str, float] = {
    "T": 1.74,
    "V": 1.88,
    "VT": 2.13,
    "RV": 2.52,
    "RVT": 2.42,
    "RV_slip": 2.61,
    "RVT_slip": 3.045,
}

# Weights of the arithmetic sum model,
# fitted on the passive same luminance dataset
b0: float = 0.08
b1: float = 0.74
b2: float = 0.53

# ===============================================
# Fitting parameters to the visual flow dataset
# -----------------------------------------------

visual_flow_data: List[float] = [
    visual_flow["V"],
    visual_flow["VT"],
    visual_flow["RV"],
    visual_flow["RVT"],
]

visual_flow_trial_types = ["V", "VT", "RV", "RVT"]

predicted_fc_visual_flow = fit_and_print_results(
    data=visual_flow_data,
    dataset_name="visual_flow",
    dataset_label="Visual flow dataset",
    trial_types=visual_flow_trial_types,
    get_predicted_function=get_predicted_fold_changes_four_way_protocol,
    actual_data=visual_flow,
)

# ===============================================
# Fitting parameters to the passive same luminance dataset
# -----------------------------------------------

passive_same_luminance_data: List[float] = [
    passive_same_luminance["V"],
    passive_same_luminance["VT"],
    passive_same_luminance["T"],
]

passive_trial_types = ["V", "VT", "T"]

predicted_fc_passive = fit_and_print_results(
    data=passive_same_luminance_data,
    dataset_name="passive_same_luminance",
    dataset_label="Passive same luminance dataset",
    trial_types=passive_trial_types,
    get_predicted_function=get_predicted_fold_changes_passive_same_luminance,
    actual_data=passive_same_luminance,
    b0=b0,
    b1=b1,
    b2=b2,
)

# ===============================================
# Fitting parameters to the matched dataset
# -----------------------------------------------

matched_data: List[float] = [
    matched_dataset["T"],
    matched_dataset["V"],
    matched_dataset["VT"],
    matched_dataset["RV"],
    matched_dataset["RVT"],
    matched_dataset["RV_slip"],
    matched_dataset["RVT_slip"],
]

matched_trial_types = ["T", "V", "VT", "RV", "RVT", "RV_slip", "RVT_slip"]

predicted_fc_matched = fit_and_print_results(
    data=matched_data,
    dataset_name="matched",
    dataset_label="Visual flow + mismatch matched dataset + T",
    trial_types=matched_trial_types,
    get_predicted_function=get_predicted_fold_changes_matched_dataset,
    actual_data=matched_dataset,
    b0=b0,
    b1=b1,
    b2=b2,
)

# ===============================================
# Create DataFrame and plot results
# -----------------------------------------------

# Prepare data for plotting
data_fold_changes = [matched_dataset[tt] for tt in matched_trial_types]
prediction_fold_changes = list(predicted_fc_matched)

# Create the DataFrame
df = pd.DataFrame(
    {
        "trial_type": matched_trial_types * 2,
        "group": ["data"] * len(matched_trial_types)
        + ["predictions"] * len(matched_trial_types),
        "fold_change": data_fold_changes + prediction_fold_changes,
    }
)

# Plot the data using seaborn
sns.barplot(data=df, x="trial_type", y="fold_change", hue="group")

# Customize the plot (optional)
plt.title("Comparison of Data and Model Predictions")
plt.xlabel("Trial Type")
plt.ylabel("Fold Change")

# Display the plot
plt.show()
