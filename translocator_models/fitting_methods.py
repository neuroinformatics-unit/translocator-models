import numpy as np
from models import rate_based_model
from scipy.optimize import minimize


def single_fit(params, data, dataset="visual_flow"):
    def objective_function(params):
        if dataset == "visual_flow":
            predicted_fc = get_predicted_fold_changes_visual_flow(params)
        elif dataset == "passive_same_luminance":
            predicted_fc = get_predicted_fold_changes_passive_same_luminance(
                params
            )
        elif dataset == "matched":
            predicted_fc = get_predicted_fold_changes_matched_dataset(params)
        residuals = data - predicted_fc
        return np.sum(np.abs(residuals.ravel()))

    bounds = (
        (0, 3),  # k
        (1, 1),  # c  = 1 in our definition
        (1, 1),  # w1 = 1 in our definition
        (0, 3),  # w2
        (0, 3),  # w3
    )
    result = minimize(
        objective_function,
        params,
        bounds=bounds,
        method="Nelder-Mead",
    )
    return result


def get_predicted_fold_changes_visual_flow(params):
    k, c, w1, w2, w3 = params

    V = rate_based_model(
        VF=1,
        T=0,
        R=0,
        k=k,
        c=c,
        w1=w1,
        w2=w2,
        w3=w3,
    )

    VT = rate_based_model(
        VF=1,
        T=1,
        R=0,
        k=k,
        c=c,
        w1=w1,
        w2=w2,
        w3=w3,
    )

    RV = rate_based_model(
        VF=1,
        T=0,
        R=1,
        k=k,
        c=c,
        w1=w1,
        w2=w2,
        w3=w3,
    )

    RVT = rate_based_model(
        VF=1,
        T=1,
        R=1,
        k=k,
        c=c,
        w1=w1,
        w2=w2,
        w3=w3,
    )

    return np.asarray([V, VT, RV, RVT])


def get_predicted_fold_changes_passive_same_luminance(params):
    k, c, w1, w2, w3 = params

    V = rate_based_model(
        VF=1,
        T=0,
        R=0,
        k=k,
        c=c,
        w1=w1,
        w2=w2,
        w3=w3,
    )

    VT = rate_based_model(
        VF=1,
        T=1,
        R=0,
        k=k,
        c=c,
        w1=w1,
        w2=w2,
        w3=w3,
    )

    T = rate_based_model(
        VF=0,
        T=1,
        R=0,
        k=k,
        c=c,
        w1=w1,
        w2=w2,
        w3=w3,
    )

    return np.asarray([V, VT, T])


def get_predicted_fold_changes_matched_dataset(params):
    k, c, w1, w2, w3 = params

    V = rate_based_model(
        VF=1,
        T=0,
        R=0,
        k=k,
        c=c,
        w1=w1,
        w2=w2,
        w3=w3,
    )

    VT = rate_based_model(
        VF=1,
        T=1,
        R=0,
        k=k,
        c=c,
        w1=w1,
        w2=w2,
        w3=w3,
    )

    RV = rate_based_model(
        VF=1,
        T=0,
        R=1,
        k=k,
        c=c,
        w1=w1,
        w2=w2,
        w3=w3,
    )

    RVT = rate_based_model(
        VF=1,
        T=1,
        R=1,
        k=k,
        c=c,
        w1=w1,
        w2=w2,
        w3=w3,
    )

    RVT_slip = rate_based_model(
        VF=2,
        T=2,
        R=1,
        k=k,
        c=c,
        w1=w1,
        w2=w2,
        w3=w3,
    )

    RV_slip = rate_based_model(
        VF=2,
        T=0,
        R=1,
        k=k,
        c=c,
        w1=w1,
        w2=w2,
        w3=w3,
    )

    T = rate_based_model(
        VF=0,
        T=1,
        R=0,
        k=k,
        c=c,
        w1=w1,
        w2=w2,
        w3=w3,
    )

    return np.asarray([T, V, VT, RV, RVT, RV_slip, RVT_slip])


def fit_fold_changes_to_data(
    data,
    dataset="visual_flow",
    initial_params=np.asarray([0.85, 1, 1, 0.6, 1]),
):
    result = single_fit(initial_params, data, dataset)

    return result
