def adaptive_response(stimulus: float, threshold: float) -> int:
    """
    Compute an adaptive response based on a given stimulus and threshold.

    Checks if the stimulus surpasses the specified threshold and returns an
    adaptive response. If the stimulus exceeds the threshold, the function
    returns 1 (active state); otherwise, it returns 0.

    Parameters
    ----------
    stimulus : float
        The input stimulus value.
    threshold : float
        The threshold level to trigger the response.

    Returns
    -------
    int
        1 if the stimulus is greater than the threshold; else 0.
    """
    adaptive_stim = 0
    if stimulus > threshold:
        adaptive_stim = 1
    return adaptive_stim


def rate_based_model(
    VF: float,
    T: float,
    R: float,
    w1: float = 1,
    w2: float = 0.6,
    w3: float = 1,
    t1: float = 0.5,
    t2: float = 0.5,
    k: float = 0.85,
    c: float = 1,
) -> float:
    """
    Rate-based fold change (fc) model of the population firing rate.

    Calculates the fold change in population firing rate for visual flow (VF),
    translation (T), and running (R) stimuli. VF and T are processed as
    adaptive responses: the response is 1 if the stimulus surpasses a certain
    threshold and 0 otherwise. The model combines these adaptive responses
    with weights (`w1`, `w2`, `w3`) and thresholds (`t1`, `t2`). The combined
    response is then scaled and offset by constants `k` and `c` to obtain the
    fold change.

    Parameters
    ----------
    VF : float
        The visual flow stimulus value.
    T : float
        The translation stimulus value.
    R : float
        The running stimulus value.
    w1 : float, optional
        Weight for VF adaptive response. Default is 1.
    w2 : float, optional
        Weight for T-R mismatch term. Default is 0.6.
    w3 : float, optional
        Weight for running stimulus. Default is 1.
    t1 : float, optional
        Threshold for VF adaptive response. Default is 0.5.
    t2 : float, optional
        Threshold for T-R mismatch adaptive response. Default is 0.5.
    k : float, optional
        Scaling factor for the response. Default is 0.85.
    c : float, optional
        Constant offset for the response. Default is 1.

    Returns
    -------
    float
        Fold change in population response.

    Notes
    -----
    - The function uses `adaptive_response` to process VF and the difference
      between T and R.
    - The combined response `v` is computed using the formula:
      `v = w1 * VF_adaptive + w2 * T_mismatch * (T - R) + w3 * R`
    - The final fold change `fc` is calculated as `fc = v * k + c`.
    """
    # Compute adaptive response for VF
    VF_adaptive = adaptive_response(VF, t1)

    # Compute adaptive response for the difference between T and R
    T_mismatch = adaptive_response(T - R, t2)

    # Compute the combined response v based on adaptive responses and weights
    v = w1 * VF_adaptive + w2 * T_mismatch * (T - R) + w3 * R

    # Scale and offset the response to obtain the fold change
    fc = v * k + c

    return fc


def arithmetic_sum_model(
    fc_T_VStatic: float,
    fc_VF: float,
    b0: float,
    b1: float,
    b2: float,
) -> float:
    """
    Predict fold change in the visual cortex using an arithmetic sum model.

    This model predicts the fold change (fc) in response to translation stimuli
    with constant luminance (`fc_T_VStatic`) and visual flow (`fc_VF`). It uses
    empirical fold changes from the `passive_same_luminance` dataset and
    combines them with weights and a constant term.

    Parameters
    ----------
    fc_T_VStatic : float
        Fold change from translation with constant luminance.
    fc_VF : float
        Fold change from visual flow.
    b0 : float
        Constant term for the model.
    b1 : float
        Weight for the `fc_T_VStatic` term.
    b2 : float
        Weight for the `fc_VF` term.

    Returns
    -------
    float
        Predicted VT fold change.

    Notes
    -----
    The model uses the formula:
    `fc = b0 + b1 * fc_T_VStatic + b2 * fc_VF`
    """
    return b0 + b1 * fc_T_VStatic + b2 * fc_VF
