import numpy as np

from scipy.stats import t
from regressionDesign import RegressionDesign
from functools import cmp_to_key
from itertools import product
import statsmodels.api as sm


class PredictionEngine:

    @staticmethod
    def tune_regression(games_cache, constants_cache, user_preferences):
        rd = RegressionDesign(False, True, games_cache, constants_cache, user_preferences)
        aicc, regr_best_rd = rd.get_best_aicc()
        return regr_best_rd

    @staticmethod
    def tune_classifier(games_cache, constants_cache, user_preferences):
        rd = RegressionDesign(True, True, games_cache, constants_cache, user_preferences)
        aicc, lgst_best_rd = rd.get_best_aicc()
        return lgst_best_rd

    @staticmethod
    def get_predictions(regr_rd: RegressionDesign, lgst_rd, games_cache, constants_cache, term_and_tag_effects_dict):
        if term_and_tag_effects_dict is None:
            term_and_tag_effects_dict = {
                "tags_list_must_have": [],
                "tags_list_kill": [],
                "tags_list_plus": [],
                "tags_list_minus": [],
                "terms_list_plus": [],
                "terms_list_minus": []
            }

        regr_preds_original, regr_mdl = best_imputed_data(regr_rd, constants_cache)
        lgst_preds_original, lgst_mdl = best_imputed_data(lgst_rd, constants_cache)

        lgst_preds_original = smooth_lgst_preds(lgst_preds_original, len(lgst_rd.fit_ids))

        y_actuals = regr_rd.y_fit
        y_hats = regr_preds_original[regr_rd.fit_locs]

        df = len(y_actuals) - len(regr_rd.interaction_set) - 1
        residuals = y_actuals - y_hats
        rse = np.sqrt(np.sum(residuals ** 2) / df)

        x_is_a_game = games_cache.game_data["IsAGame"]
        zero_out = ~x_is_a_game | regr_rd.fit_locs | lgst_rd.fit_locs
        regr_preds_original[zero_out] = -np.inf
        lgst_preds_original[zero_out] = 0

        zeroed_by_tag_and_term = get_ignored_games(games_cache, term_and_tag_effects_dict)
        regr_preds_original[zeroed_by_tag_and_term] = -np.inf
        lgst_preds_original[zeroed_by_tag_and_term] = 0

        regr_preds_original = add_regr_bonus(regr_preds_original, games_cache, term_and_tag_effects_dict,
                                             constants_cache)
        lgst_preds_original = add_lgst_bonus(lgst_preds_original, games_cache, term_and_tag_effects_dict,
                                             constants_cache)

        max_to_sort = constants_cache.MAX_GAMES_TO_SORT
        cutoff = min(max_to_sort, np.sum(~np.isinf(regr_preds_original)))
        min_score = np.quantile(regr_preds_original, 1 - cutoff / len(regr_preds_original))
        if np.isinf(min_score):
            min_score = np.min(regr_preds_original[~np.isinf(regr_preds_original)])

        print(f"min score: {min_score}")

        lgst_preds_original[regr_preds_original < min_score] = 0
        regr_preds_original[regr_preds_original < min_score] = -np.inf

        max_regr = max(regr_rd.y_fit)
        regr_preds_original[regr_preds_original > max_regr] = max_regr

        idx = np.where(~np.isinf(regr_preds_original))[0]
        data = [[regr_preds_original[i], lgst_preds_original[i], i] for i in idx]
        sorted_preds = get_sorted_predictions(data, rse, df)
        for i in range(100):
            l = sorted_preds[i][2]
            print(games_cache.game_data.iloc[l]["name"])


def smooth_lgst_preds(lgst_preds_original, n):
    a_vals = n * lgst_preds_original
    return (a_vals + 1) / (n + 2)  # applies a simple form of laplace smoothing based on the size of the fitting set,
    # works like a jeffrey's prior for small n


def get_ignored_games(games_cache, term_and_tag_effects_dict):
    """
    Parameters:
        games_cache: An object containing the game data as a pandas DataFrame
        term_and_tag_effects_dict: A dictionary-like structure with `tags_list_must_have` and `tags_list_kill` as keys

    Returns:
        zeroed: A boolean NumPy array indicating ignored games
    """
    # Initialize a boolean array with False for all rows
    zeroed = np.zeros(len(games_cache.game_data), dtype=bool)

    # Handle tags_list_must_have
    for tag in term_and_tag_effects_dict["tags_list_must_have"]:
        col_values = games_cache.game_data[tag]  # Access the column corresponding to the tag
        zeroed[col_values == 0] = True  # Mark rows where the column value is 0 as True

    # Handle tags_list_kill
    for tag in term_and_tag_effects_dict["tags_list_kill"]:
        col_values = games_cache.game_data[tag]  # Access the column corresponding to the tag
        zeroed[col_values != 0] = True  # Mark rows where the column value is non-zero as True

    return zeroed


# Function: add_regr_bonus
def add_regr_bonus(regr_preds_original, games_cache, term_and_tag_effects_dict, constants_cache):
    good_rows = ~np.isinf(regr_preds_original)
    raw_bonus, n_terms = get_raw_bonus(games_cache, term_and_tag_effects_dict, constants_cache)

    sd_regr_preds = np.std(regr_preds_original[good_rows])
    sd_bonus = np.std(raw_bonus[good_rows])

    if sd_bonus > 0:
        regr_preds = regr_preds_original + constants_cache.TAG_TERM_BONUS_MULTIPLIER * n_terms * raw_bonus * sd_regr_preds / sd_bonus
    else:
        regr_preds = regr_preds_original

    return regr_preds


def add_lgst_bonus(lgst_preds_original, games_cache, term_and_tag_effects_dict, constants_cache):
    # Identify rows with valid predictions
    good_rows = lgst_preds_original != 0

    # Get raw bonus from helper function
    raw_bonus, _ = get_raw_bonus(games_cache, term_and_tag_effects_dict, constants_cache)

    # Convert logistic predictions to logits
    logits = np.log(lgst_preds_original / (1 - lgst_preds_original))

    # Calculate standard deviations
    sd_lgst_logits = np.std(logits[good_rows])
    sd_bonus = np.std(raw_bonus[good_rows])

    if sd_bonus > 0:
        # Update logits with the scaled bonus
        lgst_logits = logits + constants_cache.TAG_TERM_BONUS_MULTIPLIER * raw_bonus * sd_lgst_logits / sd_bonus

        # Convert logits back to probabilities
        lgst_preds = 1 / (1 + np.exp(-lgst_logits))
    else:
        # No change if bonus cannot be applied
        lgst_preds = lgst_preds_original

    return lgst_preds


# Function: get_raw_bonus
def get_raw_bonus(games_cache, term_and_tag_effects_dict, constants_cache):
    game_data = games_cache.game_data

    # Process plus columns
    plus_cols = game_data[term_and_tag_effects_dict["tags_list_plus"]] > 0
    plus_cols = (plus_cols - plus_cols.mean(axis=0)) / plus_cols.std(axis=0)

    # Process minus columns
    minus_cols = game_data[term_and_tag_effects_dict["tags_list_minus"]] > 0
    minus_cols = (minus_cols - minus_cols.mean(axis=0)) / minus_cols.std(axis=0)

    # Raw bonus calculation
    raw_bonus = plus_cols.sum(axis=1) - minus_cols.sum(axis=1)

    # Add term bonuses
    raw_bonus += (
            get_term_bonuses(games_cache, term_and_tag_effects_dict["terms_list_plus"], constants_cache) -
            get_term_bonuses(games_cache, term_and_tag_effects_dict["terms_list_minus"], constants_cache)
    )

    n_terms = (
            len(term_and_tag_effects_dict["tags_list_plus"]) +
            len(term_and_tag_effects_dict["tags_list_minus"]) +
            len(term_and_tag_effects_dict["terms_list_plus"]) +
            len(term_and_tag_effects_dict["terms_list_minus"])
    )

    return raw_bonus, n_terms


# Function: get_term_bonuses
def get_term_bonuses(games_cache, terms_list, constants_cache):
    bonus = np.zeros(len(games_cache.game_data))
    game_data = games_cache.game_data

    for term in terms_list:
        if term == constants_cache.REGR_TERM_DIFF:
            bonus_line = game_data["diffs_procedural"]
        elif term == constants_cache.REGR_TERM_LOG_LENGTH:
            lengths = game_data["length"]
            lengths = np.maximum(lengths, 0.1)  # Replace NaN or < 0.1 with 0.1
            bonus_line = np.log(lengths)
        elif term == constants_cache.REGR_TERM_LOGITS:
            positives = game_data["positive"]
            negatives = game_data["negative"]
            bonus_line = RegressionDesign.get_logits(positives, negatives, constants_cache)
        elif term == constants_cache.REGR_TERM_YEAR:
            year = game_data["year"]
            year.fillna(year.mean(), inplace=True)
            bonus_line = year
        elif term == constants_cache.REGR_TERM_LOG_TOTAL_RATINGS:
            positives = game_data["positive"]
            negatives = game_data["negative"]
            total_ratings = positives + negatives
            total_ratings = np.maximum(total_ratings, 1)  # Replace values < 1 with 1
            bonus_line = np.log(total_ratings)
        else:
            raise ValueError(f"Unknown term {term}")

        # Standardize the bonus_line
        bonus += (bonus_line - bonus_line.mean()) / bonus_line.std()

    return bonus


def get_sorted_predictions(data, se, df):
    # Define a pairwise comparison function

    comparator = create_comparator(se, df)

    # Convert the pairwise comparison function to a key function
    key_function = cmp_to_key(comparator)

    # Sort the list
    sorted_data = sorted(data, key=key_function)

    # Print the sorted list
    return sorted_data


def create_comparator(se, df):
    def compare_pairs(a, b):
        regr_a = a[0]
        lgst_a = a[1]

        regr_b = b[0]
        lgst_b = b[1]
        if np.isinf(regr_a) and np.isinf(regr_b):
            return 0
        elif np.isinf(regr_b) or (regr_a > regr_b and lgst_a > lgst_b):
            return -1
        elif np.isinf(regr_a) or (regr_a < regr_b and lgst_a < lgst_b):
            return 1
        elif regr_a == regr_b and lgst_a == lgst_b:
            return 0
        elif regr_a >= regr_b and lgst_a >= lgst_b:
            return -1
        elif regr_a <= regr_b and lgst_a <= lgst_b:
            return 1
        else:
            # Calculate win probabilities
            a_wins = lgst_a * (1 - lgst_b)
            b_wins = lgst_b * (1 - lgst_a)
            p_both_active = lgst_a * lgst_b

            if p_both_active > 0:
                # Compute t-value
                t_value = -(regr_a - regr_b) / (np.sqrt(2) * se)

                # Compute probability using the cumulative distribution function
                p_a_wins = 1 - t.cdf(t_value, df)
                a_wins += p_both_active * p_a_wins
                b_wins += p_both_active * (1 - p_a_wins)

            # Final comparison
            return -1 if a_wins > b_wins else 1

    return compare_pairs


def best_imputed_data(rd, constants_cache):
    """
    Finds the best imputed data based on the highest mean predicted value.

    Parameters:
        rd: A regression design object with a method `get_finished_model(impute_sequence)`.

    Returns:
        values_best (np.ndarray): Best values based on imputation.
        mdl_best: The best model based on the imputed data.
    """
    n = constants_cache.IMPUTE_SEQ_LENGTH

    # Generate all binary sequences of length n
    binary_sequences = np.array(list(product([0, 1], repeat=n)))

    best_value = -np.inf
    values_best = None
    mdl_best = None

    for impute_sequence in binary_sequences:
        # Get the model and associated data for the imputation sequence
        mdl, xs_mdl, x_fit, fit_ids = rd.get_finished_model(impute_sequence)

        # Ensure fit_ids is sorted
        if not np.all(np.diff(fit_ids) >= 0):
            raise ValueError("fit_ids not sorted")

        # Compute values
        if xs_mdl is None or len(xs_mdl) == 0:
            # Use model coefficients if no xs_mdl
            values = mdl.coef_
        else:
            if rd.is_logistic:
                values = mdl.predict(sm.add_constant(xs_mdl))
            else:
                # Predict values using the model
                values = mdl.predict(xs_mdl)

        # Calculate the average value
        avg_value = np.mean(values)

        # Check if this is the best sequence
        if avg_value > best_value:
            values_best = values
            best_value = avg_value
            mdl_best = mdl

    return values_best, mdl_best
