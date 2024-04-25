import pandas as pd
import seaborn as sns
from fitting_methods import (
    fit_fold_changes_to_data,
    get_predicted_fold_changes_matched_dataset,
    get_predicted_fold_changes_passive_same_luminance,
    get_predicted_fold_changes_visual_flow,
)
from models import arithmetic_sum_model

#  ===============================================
#  Mean fold changes across datasets
#  visual flow, 39 clusters
visual_flow = {}
visual_flow["RV"] = 2.52
visual_flow["RVT"] = 2.42
visual_flow["V"] = 1.88
visual_flow["VT"] = 2.13

#  passive same luminance, 151 clusters
passive_same_luminance = {}
passive_same_luminance["T"] = 1.44
passive_same_luminance["V"] = 1.16
passive_same_luminance["VT"] = 1.76

# Re-balanced fold changes, together in the same dataset
matched_dataset = {}
matched_dataset["T"] = 1.74
matched_dataset["V"] = 1.88
matched_dataset["VT"] = 2.13
matched_dataset["RV"] = 2.52
matched_dataset["RVT"] = 2.42
matched_dataset["RV_slip"] = 2.61
matched_dataset["RVT_slip"] = 3.045

# Weights of the arithmetic sum model, fitted on the passive same luminance
# dataset
b0 = 0.08
b1 = 0.74
b2 = 0.53

#  ===============================================
#  Fitting parameters to visual flow dataset
print("===============================================")
print("Visual flow dataset")

best_result = fit_fold_changes_to_data(
    data=[
        visual_flow["V"],
        visual_flow["VT"],
        visual_flow["RV"],
        visual_flow["RVT"],
    ]
)

k, c, w1, w2, w3 = best_result.x

print(f"fc(v) = {k:.2f} * v + {c:.2f}")
print(f"v = {w1:.2f} (VF > 0) + {w2:.2f} (T - R > 0)(T - R) + {w3:.2f} R")

predicted_fc = get_predicted_fold_changes_visual_flow(best_result.x)


print(f"V:   {predicted_fc[0]:.2f}, mean from data: {visual_flow['V']}")
print(f"VT:  {predicted_fc[1]:.2f}, mean from data: {visual_flow['VT']}")
print(f"RV:  {predicted_fc[2]:.2f}, mean from data: {visual_flow['RV']}")
print(f"RVT: {predicted_fc[3]:.2f}, mean from data: {visual_flow['RVT']}")


#  ===============================================
#  Fitting parameters to passive_same_luminance dataset
print("===============================================")
print("Passive same luminance dataset")


best_result = fit_fold_changes_to_data(
    data=[
        passive_same_luminance["V"],
        passive_same_luminance["VT"],
        passive_same_luminance["T"],
    ],
    dataset="passive_same_luminance",
)


k, c, w1, w2, w3 = best_result.x

print(f"fc(v) = {k:.2f} * v + {c:.2f}")
print(f"v = {w1:.2f} (VF > 0) + {w2:.2f} (T - R > 0)(T - R) + {w3:.2f} R")

predicted_fc = get_predicted_fold_changes_passive_same_luminance(best_result.x)


print(
    f"V:  {predicted_fc[0]:.2f}, "
    + "mean from data: {passive_same_luminance['V']}"
)
print(
    f"VT: {predicted_fc[1]:.2f}, "
    + "mean from data: {passive_same_luminance['VT']}"
)
print(
    f"T:  {predicted_fc[2]:.2f}, "
    + "mean from data: {passive_same_luminance['T']}"
)


arithmetic_sum = {}
arithmetic_sum["VT"] = arithmetic_sum_model(
    fc_T_VStatic=predicted_fc[2],
    fc_VF=predicted_fc[0],
    b0=b0,
    b1=b1,
    b2=b2,
)

print(
    "VT as predicted with the arithmetic sum + rate model: "
    + f"{arithmetic_sum['VT']:.2f}"
)


#  ===============================================
# matched dataset
print("===============================================")
print("Visual flow + mismatch matched dataset + T")


best_result = fit_fold_changes_to_data(
    data=[
        matched_dataset["T"],
        matched_dataset["V"],
        matched_dataset["VT"],
        matched_dataset["RV"],
        matched_dataset["RVT"],
        matched_dataset["RV_slip"],
        matched_dataset["RVT_slip"],
    ],
    dataset="matched",
)


k, c, w1, w2, w3 = best_result.x

print(f"fc(v) = {k:.2f} * v + {c:.2f}")
print(f"v = {w1:.2f} (VF > 0) + {w2:.2f} (T - R > 0)(T - R) + {w3:.2f} R")

predicted_fc = get_predicted_fold_changes_matched_dataset(best_result.x)

print(f"T:   {predicted_fc[0]:.2f}, mean from data: {matched_dataset['T']}")
print(f"V:   {predicted_fc[1]:.2f}, mean from data: {matched_dataset['V']}")
print(f"VT:  {predicted_fc[2]:.2f}, mean from data: {matched_dataset['VT']}")
print(f"RV:  {predicted_fc[3]:.2f}, mean from data: {matched_dataset['RV']}")
print(f"RVT: {predicted_fc[4]:.2f}, mean from data: {matched_dataset['RVT']}")
print(
    f"RV_slip: {predicted_fc[5]:.2f}, "
    + "mean from data: {matched_dataset['RV_slip']}"
)
print(
    f"RVT_slip: {predicted_fc[6]:.2f}, "
    + "mean from data: {matched_dataset['RVT_slip']}"
)


arithmetic_sum = {}
arithmetic_sum["VT"] = arithmetic_sum_model(
    fc_T_VStatic=predicted_fc[0],
    fc_VF=predicted_fc[1],
    b0=b0,
    b1=b1,
    b2=b2,
)

print(
    "VT as predicted with the arithmetic sum + rate model: "
    + f"{arithmetic_sum['VT']:.2f}"
)

df = pd.DataFrame(
    {
        "trial_type": matched_dataset.keys(),
        "data": matched_dataset.values(),
        "predictions": predicted_fc,
    },
)
df = pd.melt(
    df,
    id_vars="trial_type",
    value_vars=["data", "predictions"],
    var_name="group",
    value_name="fold_change",
)
sns.barplot(df, x="trial_type", y="fold_change", hue="group")
