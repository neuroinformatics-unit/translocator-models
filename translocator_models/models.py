def adaptive_response(stimulus, threshold):
    """Compute adaptive response to a given stimulus."""
    adaptive_stim = 0
    if stimulus > threshold:
        adaptive_stim = 1
    return adaptive_stim


def rate_based_model(
    VF, T, R, w1=1, w2=0.6, w3=1, t1=0.5, t2=0.5, k=0.85, c=1
):
    """Predict fold change of the population response from the baseline"""

    VF_adaptive = adaptive_response(VF, t1)

    T_mismatch = adaptive_response(T - R, t2)

    v = w1 * VF_adaptive + w2 * T_mismatch * (T - R) + w3 * R

    fc = v * k + c

    return fc


def arithmetic_sum_model(fc_T_VStatic, fc_VF, b0, b1, b2):
    """Predict VT fold change given V and T using the weights fitten on
    all clusters of the passive_same_luminance dataset."""

    return b0 + b1 * fc_T_VStatic + b2 * fc_VF
