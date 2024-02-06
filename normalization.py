def min_max_normalize(data):
    """
    Applies min-max normalization to a list of numerical values.
    Normalization formula: (x - min) / (max - min)
    """
    min_val = min(data)
    max_val = max(data)
    if max_val - min_val == 0:  # Prevent division by zero
        return [0] * len(data)
    return [(x - min_val) / (max_val - min_val) for x in data]

def run_min_max_norm(fit_scores, refit_scores):
    # Initialize containers for normalized scores
    normalized_fit_scores = []
    normalized_refit_scores = []

    # Iterate through each dataset's scores to apply normalization
    for fit_score, refit_score in zip(fit_scores, refit_scores):
        combined_scores = fit_score + refit_score  # Combine fit and refit scores for normalization
        normalized_scores = min_max_normalize(combined_scores)  # Normalize

        # Separate the normalized scores back into fit and refit, maintaining their original order
        normalized_fit = normalized_scores[:len(fit_score)]
        normalized_refit = normalized_scores[len(fit_score):]

        normalized_fit_scores.append(normalized_fit)
        normalized_refit_scores.append(normalized_refit)

    # Check the first elements of normalized scores to verify the process
    return normalized_fit_scores, normalized_refit_scores
