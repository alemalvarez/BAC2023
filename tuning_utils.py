import scoring_utils

def get_cutoff(model, X, y) -> float:
    """Function that gets optimum cutoff for a model.

    Args:
        model (_type_): The model for which to find the cutoff.
        X (_type_): Data to to predictions.
        y (_type_): Ground truth.

    Returns:
        float: The best cutoff.
    """
    cutoffs = [.4, .45, .5, .525, .55, .575, .6]

    best_cutoff = 0
    best_profit = 0

    for cutoff in cutoffs:
        yprob = model.predict_proba(X)[:,1]
        profit = scoring_utils.get_profit(X, y, (yprob>cutoff))
        print(f"Cutoff: {cutoff} , Profit: ${profit:,.2f}")
        if profit > best_profit:
            best_cutoff = cutoff
            best_profit = profit

    return best_cutoff