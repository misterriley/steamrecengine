import copy
import math

import numpy as np
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from constantsCache import ConstantsCache


class RegressionDesign:
    def __init__(self, is_logistic, calc_plsr, games_cache, constants_cache: ConstantsCache, user_preferences):
        """
        Constructor for the RegressionDesign class.

        Parameters:
        is_logistic (bool): Determines if logistic regression is used.
        calc_plsr (bool): Whether to calculate PLSR.
        dbw: Database wrapper object to fetch data.
        """
        self.constants_cache = constants_cache
        self.interaction_set = [[1]]
        self.is_logistic = is_logistic

        if user_preferences is not None:
            # Load fit_table and y_fit based on logistic or rating preferences
            if self.is_logistic:
                fit_table = [
                    pref for pref in user_preferences
                    if pref['status'] == constants_cache.STATUS_MORE or pref['status'] == constants_cache.STATUS_LESS
                ]
                self.y_fit = np.array([pref['status'] for pref in fit_table])
            else:
                fit_table = [
                    pref for pref in user_preferences
                    if pref['rating'] is not None
                ]
                self.y_fit = np.array([pref['rating'] for pref in fit_table])

            # Store fit_ids and ensure they are sorted
            self.fit_ids = [pref['game_id'] for pref in fit_table]
            if not np.all(np.diff(self.fit_ids) >= 0):
                raise ValueError("self.fit_ids is not sorted")

        if games_cache is not None:
            # Compute fit_locs
            self.fit_locs = games_cache.game_data['game_id'].isin(self.fit_ids)

            # Load latent dimensions
            lds_dims = games_cache.latent_data[self.fit_locs].values
            game_data = games_cache.game_data

            # Load additional data from game_data
            self.diffs = game_data['diffs_procedural'].values
            positives = game_data['positive'].values
            negatives = game_data['negative'].values
            lengths = game_data['length'].values
            self.log_lengths = np.log(np.maximum(lengths, .1))
            self.years = game_data['year'].values
            total_ratings = positives + negatives
            self.log_total_ratings = np.log(np.maximum(total_ratings, 1))
            self.logits = RegressionDesign.get_logits(positives, negatives, constants_cache)

            # Calculate PLSR if required
        if calc_plsr:
            if games_cache is not None:
                self.plsr_pas, self.plsr_betas = self.get_plsr(lds_dims, games_cache)
            else:
                raise ValueError("games_cache must exist if calc_plsr is true")

        # Set interaction lengths
        self.max_interaction_length = 10
        self.max_repeat_length = 10
        if self.is_logistic:
            self.max_interaction_length = 10
            self.max_repeat_length = 10

    def add_interaction(self, interaction):
        # Sort the interaction
        interaction = sorted(interaction)

        for interaction_index, interaction_at_index in enumerate(self.interaction_set):
            skip_this = False
            for place_index, val in enumerate(interaction):
                if place_index >= len(interaction_at_index):
                    skip_this = True
                    break

                val_at_index = interaction_at_index[place_index]

                if val > val_at_index:
                    skip_this = True
                    break
                elif val < val_at_index:
                    skip_this = False
                    break
                # If values are equal, continue to the next index

            if not skip_this:
                # Insert interaction in the correct position
                self.interaction_set.insert(interaction_index, interaction)
                return

        # Append the interaction if no insertion point was found
        self.interaction_set.append(interaction)

    def do_regression(self, tried_models):
        """Translate the `do_regression` MATLAB function."""
        # Get a unique description of the model
        dsc = self.get_desc()

        # Initialize AICc to infinity
        aicc = math.inf

        # Check if the model has already been tried
        if tried_models and dsc in tried_models:
            return aicc, tried_models

        # Add the current model description to the tried_models list
        tried_models.append(dsc)

        # Get the finished model
        mdl, x, x_fit, fit_ids = self.get_finished_model()
        if mdl is None:  # Check for an empty model
            return aicc, tried_models

        # Calculate AICc
        k = len(self.interaction_set) + 1  # Number of parameters
        n = len(self.y_fit)  # Number of data points
        if n <= k + 1:  # Avoid division by zero or invalid AICc calculation
            return aicc, tried_models

        if self.is_logistic:
            ll = mdl.llf
        else:
            y_pred = mdl.predict(x_fit)
            residuals = self.y_fit - y_pred
            sigma_squared = np.mean(residuals ** 2)
            ll = -n / 2 * np.log(2 * np.pi * sigma_squared) - np.sum(residuals**2) / (2 * sigma_squared)

        aicc = (
                2 * k
                - 2 * ll
                + (2 * k ** 2 + 2 * k) / (n - k - 1)
        )

        # Print the current state
        self.print_state()

        return aicc, tried_models

    def get_finished_model(self, impute_sequence=None):
        """
        Perform model building and return the finished model, along with relevant data.
        """
        mdl = None

        if impute_sequence is not None:
            # Impute data with the given sequence
            x = self.impute_with_sequence(impute_sequence)
            if x is None and self.interaction_set:
                return mdl, None, None, None
        else:
            # Default imputation sequences
            x_max = self.impute_with_sequence(np.ones(self.constants_cache.IMPUTE_SEQ_LENGTH))
            x_min = self.impute_with_sequence(np.zeros(self.constants_cache.IMPUTE_SEQ_LENGTH))
            if (x_max is None or x_min is None) and self.interaction_set:
                return mdl, None, None, None
            x = x_max  # Use x_max by default

        if x is not None:
            # Extract the rows for fitting based on fit_locs
            x_fit = x[self.fit_locs, :]
            # Perform logistic or linear regression based on `is_logistic`
            if self.is_logistic:
                try:
                    mdl = sm.GLM(self.y_fit, sm.add_constant(x_fit),
                                 family=sm.families.Binomial(link=sm.families.links.Logit())).fit()
                except ValueError as e:
                    raise RuntimeError(f"Error during logistic regression fitting: {e}")
            else:
                mdl = LinearRegression()
                try:
                    mdl.fit(x_fit, self.y_fit)
                except ValueError as e:
                    raise RuntimeError(f"Error during linear regression fitting: {e}")
        else:
            x_fit = None
            # Use only `y_fit` as a table to fit an intercept-only model
            t_y = pd.DataFrame({"y_fit": self.y_fit})
            mdl = LinearRegression()
            mdl.fit(t_y, np.ones(len(t_y)))  # Fitting intercept-only

        fit_ids = self.fit_ids
        if not np.all(np.diff(fit_ids) >= 0):  # Check if sorted
            raise ValueError("fit_ids not sorted")

        return mdl, x, x_fit, fit_ids

    def impute_with_sequence(self, impute_sequence):
        """
        Translates the MATLAB `impute_with_sequence` function.
        """
        # Call `construct_xs_by_parts` to compute `x`
        x = RegressionDesign.construct_xs_by_parts(
            self.interaction_set,
            impute_sequence,
            self.plsr_pas,
            self.diffs,
            self.log_lengths,
            self.log_total_ratings,
            self.logits,
            self.years,
            self.constants_cache
        )

        # Check if `x` is empty or contains invalid values
        if x.size == 0 or np.any(np.abs(np.imag(x)) > 0) or np.any(np.isinf(x)) or np.any(np.isnan(x)):
            x = None  # Return `None` for invalid cases

        return x

    def contains_any_reference(self, index):
        """
        Check if any element in `index` is found in any interaction in `interaction_set`.
        """
        return any(index in interaction for interaction in self.interaction_set)

    def contains_interaction(self, interaction):
        """
        Check if the given `interaction` exists in `interaction_set`.
        """
        # Sort the input interaction for comparison
        if isinstance(interaction, list):
            test_val = sorted(interaction)
        elif isinstance(interaction, set):
            test_val = sorted(interaction)
        else:
            raise ValueError("Interaction must be a list or set.")

        # Iterate through the interaction set
        for current_interaction in self.interaction_set:
            # Check if lengths match and sorted values are equal
            if len(current_interaction) == len(test_val) and sorted(current_interaction) == test_val:
                return True

        return False

    def get_desc(self):
        """
        Generate a description string of the `interaction_set`.
        """
        s = ""
        for interaction in self.interaction_set:
            s += "/" + str(interaction)
        return s  # Convert to string and return

    def clone(self):
        """
        Clone the current object and return a new instance with the same properties.
        """
        # Create a new instance of the same class
        cln = RegressionDesign(self.is_logistic, False, None, None, None)

        # Copy attributes to the new instance
        cln.interaction_set = copy.deepcopy(self.interaction_set)
        cln.plsr_pas = self.plsr_pas
        cln.plsr_betas = self.plsr_betas
        cln.max_interaction_length = self.max_interaction_length
        cln.max_repeat_length = self.max_repeat_length
        cln.fit_locs = self.fit_locs
        cln.y_fit = self.y_fit
        cln.diffs = self.diffs
        cln.log_lengths = self.log_lengths
        cln.years = self.years
        cln.log_total_ratings = self.log_total_ratings
        cln.logits = self.logits
        cln.fit_ids = self.fit_ids
        cln.constants_cache = self.constants_cache

        return cln

    def print_state(self):
        """
        Print the state of the interaction set.
        """
        for interaction in self.interaction_set:
            print(f"\t{str(interaction)}")

    def test_model(self, aicc_best, tried_models):
        """
        Test a model and update the best AICc if the current model is better.
        """
        # Perform regression and get the AICc for the current model
        aicc, tried_models = self.do_regression(tried_models)

        # Check if the new model is better
        if aicc < aicc_best:
            # Update the best AICc, the object, and the dirty flag
            aicc_best, obj_best, is_dirty = self.do_new_best_aicc(aicc)
        else:
            # Log the result if the current model is not better
            print(f"Boo! Haman! aicc = {aicc:.6f} (best = {aicc_best:.6f})")

            # Maintain the existing best object and state
            obj_best = None
            is_dirty = False

        return aicc_best, obj_best, is_dirty, tried_models

    def do_new_best_aicc(self, aicc):
        """
        Update the best AICc value and related states.

        Parameters:
        aicc (float): The new best AICc value.

        Returns:
        tuple: (aicc_best, obj_best, is_dirty)
        """
        print(f"Yay! Mordecai! aicc = {aicc:.6f}")

        # Update best AICc and related states
        aicc_best = aicc
        obj_best = self  # Return the current object as the best one
        is_dirty = True

        return aicc_best, obj_best, is_dirty

    def sanity_check(self):
        """
        Perform a sanity check on the `interaction_set`.
        """
        is_sane = True

        # Check if PLSR_TERM_ORD is referenced
        if not self.contains_any_reference(self.constants_cache.REGR_TERM_PLSR):
            is_sane = False

        # Check all interactions in the set
        for interaction in self.interaction_set:
            if not is_sane:
                break

            # Remove each element from the interaction and check the reduced interaction
            for delete_index in range(len(interaction)):
                down_one = interaction[:delete_index] + interaction[delete_index + 1:]  # Remove one element
                if down_one and not self.contains_interaction(down_one):
                    is_sane = False
                    break

        return is_sane

    def get_best_aicc(self):
        """
        Determine the best AICc by iterating through interaction sets.
        """
        tried_models = []
        aicc_best, obj_best, is_dirty, tried_models = self.test_model(float("inf"), tried_models)
        print(f"Initial aicc = {aicc_best:.6f}")

        while is_dirty:
            is_dirty = False
            ref_obj = obj_best

            # Combine interactions
            for x_index, x_interaction in enumerate(ref_obj.interaction_set):
                for y_index, y_interaction in enumerate(ref_obj.interaction_set[x_index:], start=x_index):
                    compound_interaction = x_interaction + y_interaction

                    # Check if the compound interaction is valid
                    is_good = (
                                      len(set(compound_interaction)) > 1 and len(
                                  compound_interaction) <= ref_obj.max_interaction_length
                              ) or (
                                      len(set(compound_interaction)) == 1 and len(
                                  compound_interaction) <= ref_obj.max_repeat_length
                              )

                    if not is_good:
                        continue

                    # Add the compound interaction and test the model
                    if not ref_obj.contains_interaction(compound_interaction):
                        cln = ref_obj.clone()
                        cln.add_interaction(compound_interaction)
                        if not cln.sanity_check():
                            continue

                        aicc, tried_models = cln.do_regression(tried_models)
                        if aicc < aicc_best:
                            aicc_best, obj_best, is_dirty = cln.do_new_best_aicc(aicc)
                            break
                        else:
                            print(f"Boo! Haman! aicc = {aicc:.6f} (best = {aicc_best:.6f})")

                if is_dirty:
                    break

            # Remove interactions
            ref_obj = obj_best
            for interaction_index in range(len(ref_obj.interaction_set)):
                cln = ref_obj.clone()
                del cln.interaction_set[interaction_index]
                if not cln.sanity_check():
                    continue

                aicc, tried_models = cln.do_regression(tried_models)
                if aicc < aicc_best:
                    aicc_best, obj_best, is_dirty = cln.do_new_best_aicc(aicc)
                    break
                else:
                    print(f"Boo! Haman! aicc = {aicc:.6f} (best = {aicc_best:.6f})")

            # Add new terms
            ref_obj = obj_best
            for interaction_index in range(1, self.constants_cache.N_REGR_TERMS + 1):
                if not ref_obj.contains_interaction([interaction_index]):
                    cln = ref_obj.clone()
                    cln.add_interaction([interaction_index])
                    if not cln.sanity_check():
                        continue

                    aicc, tried_models = cln.do_regression(tried_models)
                    if aicc < aicc_best:
                        aicc_best, obj_best, is_dirty = cln.do_new_best_aicc(aicc)
                        break
                    else:
                        print(f"Boo! Haman! aicc = {aicc:.6f} (best = {aicc_best:.6f})")

        print(f"Exiting with best aicc = {aicc_best:.6f}")
        return aicc_best, obj_best

    def get_plsr(self, lds_dims, games_cache):
        """
        Perform Partial Least Squares Regression.

        Parameters:
        obj: An object with `y_fit` (dependent variable).
        lds_dims: The independent variable matrix for PLSR.
        latent_array: The latent data array.

        Returns:
        pas: Predicted values using the latent array and regression coefficients.
        beta: Coefficients obtained from PLS regression.
        """
        # Perform Partial Least Squares Regression with one component
        pls = PLSRegression(n_components=1, scale=False)
        pls.fit(lds_dims, self.y_fit)

        # Get the regression coefficients
        beta = pls.coef_

        pas = pls.predict(games_cache.latent_data)

        return pas, beta

    @staticmethod
    def get_fake_row_latent(tag_names, tag, repeats, V, mu, lds_dims, latent_array, IDF):
        """
        Add new rows corresponding to a fake tag latent space projection.

        Parameters:
        tag_names (list): List of tag names.
        tag (str): Tag to find and process.
        repeats (int): Number of times to repeat the fake row latent.
        V (numpy.ndarray): Matrix for latent space transformation.
        mu (numpy.ndarray): Mean vector for centering.
        lds_dims (numpy.ndarray): Existing latent dimensions to be updated.
        latent_array (numpy.ndarray): Existing latent array to be updated.
        IDF (numpy.ndarray): Inverse Document Frequency weights.

        Returns:
        lds_dims: Updated latent dimensions.
        latent_array: Updated latent array.
        """
        # Locate the tag index
        if tag not in tag_names:
            raise ValueError(f"Tag {tag} not found in dataset")

        tag_loc = np.array(tag_names) == tag

        # Compute the fake row
        fake_row = tag_loc.astype(float) * IDF

        # Center the fake row
        fake_row_centered = fake_row - mu

        # Project the fake row into latent space
        fake_row_latent = fake_row_centered @ V

        # Repeat the latent row for the given number of repeats
        new_rows = np.tile(fake_row_latent, (repeats, 1))

        # Append to existing arrays
        lds_dims = np.vstack((lds_dims, new_rows))
        latent_array = np.vstack((latent_array, new_rows))

        return lds_dims, latent_array

    @staticmethod
    def construct_xs_by_parts(interactions, impute_sequence, plsr_pas, diffs_procedural,
                              log_lengths, log_total_ratings, logits, years, constants):
        """
        Constructs the X matrix by processing interaction terms.

        Parameters:
        interactions (list of lists): List of interactions, where each interaction is a list of terms.
        impute_sequence (list): Sequence of imputation bits.
        plsr_pas (numpy.ndarray): PLSR processed data.
        diffs_procedural (numpy.ndarray): Procedural differences data.
        log_lengths (numpy.ndarray): Logarithm of lengths.
        log_total_ratings (numpy.ndarray): Logarithm of total ratings.
        logits (numpy.ndarray): Logits data.
        years (numpy.ndarray): Years data.

        Returns:
        numpy.ndarray: Constructed X matrix.
        """
        x = np.zeros((len(plsr_pas), len(interactions)))

        for interaction_index, interaction in enumerate(interactions):
            col = None
            for term in interaction:
                impute_bit = 0

                # Select the appropriate data based on the term
                if term == constants.REGR_TERM_PLSR:
                    y = plsr_pas
                elif term == constants.REGR_TERM_DIFF:
                    y = diffs_procedural
                elif term == constants.REGR_TERM_LOG_LENGTH:
                    y = log_lengths
                    impute_bit = impute_sequence[0]
                elif term == constants.REGR_TERM_LOGIT:
                    y = logits
                elif term == constants.REGR_TERM_YEAR:
                    y = years
                    impute_bit = impute_sequence[1]
                elif term == constants.REGR_TERM_LOG_TOTAL_RATINGS:
                    y = log_total_ratings
                else:
                    raise ValueError(f"Unknown term in switch: {term}")

                # Imputation and thresholding
                thresh = 0.1
                y_mean = np.nanmean(y)
                if impute_bit == 1:
                    set_val = np.quantile(y[~np.isnan(y)], 1 - thresh)
                else:
                    set_val = np.quantile(y[~np.isnan(y)], thresh)
                y[np.isnan(y)] = set_val

                # Center the data and calculate the column
                centered_y = y - y_mean
                if col is None:
                    col = centered_y
                else:
                    col = col * centered_y

            # Assign the computed column to the X matrix
            x[:, interaction_index] = col

        return x

    @staticmethod
    def get_logits(positives, negatives, constants):
        """
        Compute logits based on positives and negatives.

        Parameters:
        positives (numpy.ndarray): Array of positive counts.
        negatives (numpy.ndarray): Array of negative counts.

        Returns:
        numpy.ndarray: Array of logits.
        """
        # Calculate log_n with safeguards for invalid log values
        log_n = np.log(np.maximum(positives + negatives, 2))

        # Calculate probabilities
        numerator = positives + constants.ALPHA + log_n * constants.NUM_SLOPE
        denominator = positives + constants.ALPHA + negatives + constants.BETA + log_n * (
                constants.NUM_SLOPE + constants.DENOM_SLOPE)
        ps = numerator / denominator

        # Calculate logits
        logits = np.log(ps / (1 - ps))

        return logits
